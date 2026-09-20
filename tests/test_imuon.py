"""CPU numerical and training-integration regressions for intrinsic Muon.

Run: python -m unittest discover -s tests -p 'test_imuon.py' -v
"""

import argparse
import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from optimizer import imuon


ROOT = Path(__file__).resolve().parents[1]


def reference_polar(X):
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    keep = S > S[0] * max(X.shape) * torch.finfo(X.dtype).eps
    return U[:, keep] @ Vh[keep]


def inverse_sqrt(C):
    values, vectors = torch.linalg.eigh(C)
    return (vectors * values.rsqrt()) @ vectors.T


def reference_step(A, B, GA, GB, lr):
    """F.3's symmetric Gram-root equations, independent of the QR code."""
    CA = inverse_sqrt(A @ A.T)
    CB = inverse_sqrt(B.T @ B)
    return (A - lr * CB @ reference_polar(CB @ GA),
            B - lr * reference_polar(GB @ CA) @ CA)


class ToyModel(nn.Module):
    def __init__(self, A, B):
        super().__init__()
        self.layer = nn.Module()
        self.layer.base_layer = nn.Linear(A.shape[1], B.shape[0], bias=False).double()
        self.layer.base_layer.weight.requires_grad_(False)
        self.layer.lora_A = nn.ModuleDict({
            'server': nn.Linear(A.shape[1], A.shape[0], bias=False).double()})
        self.layer.lora_B = nn.ModuleDict({
            'server': nn.Linear(B.shape[1], B.shape[0], bias=False).double()})
        self.layer.scaling = {'server': 2.0}
        with torch.no_grad():
            self.A.copy_(A)
            self.B.copy_(B)

    @property
    def A(self):
        return self.layer.lora_A['server'].weight

    @property
    def B(self):
        return self.layer.lora_B['server'].weight


class IMuonTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.A = torch.randn(3, 9, dtype=torch.float64)
        self.B = torch.randn(7, 3, dtype=torch.float64)
        self.G = torch.randn(7, 9, dtype=torch.float64)

    def test_qr_matches_gram_root_equations_and_simultaneous_step(self):
        for scaling in (0.5, 1.0, 4.0):
            A, B = self.A.clone(), self.B.clone()
            GA, GB = scaling * B.T @ self.G, scaling * self.G @ A.T
            expected = reference_step(A, B, GA, GB, 0.07)
            state = {}
            step_sq = imuon.imuon_step(A, B, GA, GB, state, 0.07, polar_method='svd')
            torch.testing.assert_close(A, expected[0], atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(B, expected[1], atol=1e-12, rtol=1e-12)
            self.assertEqual(state, {})
            self.assertAlmostEqual(step_sq, ((A-self.A).square().sum()
                                            + (B-self.B).square().sum()).item())

    def test_exact_step_respects_change_of_factor_basis(self):
        N = torch.tensor([[2., .3, -.2], [0., .7, .4], [.1, 0., 1.4]], dtype=torch.float64)
        A, B = self.A.clone(), self.B.clone()
        transformed_A, transformed_B = N @ A, B @ torch.linalg.inv(N)
        for a, b in ((A, B), (transformed_A, transformed_B)):
            imuon.imuon_step(a, b, b.T @ self.G, self.G @ a.T, {}, .03,
                             polar_method='svd')
        torch.testing.assert_close(transformed_B @ transformed_A, B @ A,
                                   atol=1e-12, rtol=1e-12)

    def test_appendix_k_momentum_matches_dense_reference_over_multiple_steps(self):
        A, B = self.A.clone(), self.B.clone()
        MA, MB = torch.zeros_like(A), torch.zeros_like(B)
        state = {}
        for t in range(3):
            G = self.G + .1 * t
            GA, GB = B.T @ G, G @ A.T
            MA, MB = .95 * MA + GA, .95 * MB + GB
            ambient = (GB + .95 * MB) @ A + B @ (GA + .95 * MA)
            expected = reference_step(A, B, B.T @ ambient, ambient @ A.T, .02)
            imuon.imuon_step(A, B, GA, GB, state, .02, beta=.95, polar_method='svd')
            torch.testing.assert_close(A, expected[0], atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(B, expected[1], atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(state['M_A'], MA)
            torch.testing.assert_close(state['M_B'], MB)

    def test_ns_matches_svd_for_well_conditioned_inputs(self):
        for X in (self.A, self.B, self.A.float(), self.B.float()):
            torch.testing.assert_close(imuon.polar(X), imuon.polar(X, 'svd'),
                                       atol=2e-6, rtol=2e-6)

    def test_zero_and_rank_deficient_gradients_do_not_inject_null_directions(self):
        for method in ('ns', 'svd'):
            A, B = self.A.clone(), self.B.clone()
            self.assertEqual(imuon.imuon_step(A, B, torch.zeros_like(A),
                                             torch.zeros_like(B), {}, .1,
                                             polar_method=method), 0)
            torch.testing.assert_close(A, self.A, atol=0, rtol=0)
            torch.testing.assert_close(B, self.B, atol=0, rtol=0)
            X = self.B[:, :1] @ self.A[:1]
            torch.testing.assert_close(imuon.polar(X, method), reference_polar(X),
                                       atol=1e-12, rtol=1e-12)

    def test_singular_factors_are_rejected_without_parameter_changes(self):
        for zero in (False, True):
            A, B = self.A.clone(), self.B.clone()
            if zero:
                B.zero_()
            else:
                B[:, 2] = B[:, 0]
            original = B.clone()
            with self.assertRaisesRegex(ValueError, 'full-rank'):
                imuon.imuon_step(A, B, B.T @ self.G, self.G @ A.T, {}, .1)
            torch.testing.assert_close(A, self.A, atol=0, rtol=0)
            torch.testing.assert_close(B, original, atol=0, rtol=0)

    def test_bfloat16_storage_and_autocast_use_fp32_math(self):
        A, B = self.A.bfloat16(), self.B.bfloat16()
        G = self.G.float()
        GA, GB = B.float().T @ G, G @ A.float().T
        expected_A, expected_B = A.float(), B.float()
        imuon.imuon_step(expected_A, expected_B, GA, GB, {}, .1, polar_method='svd')
        with torch.autocast('cpu', dtype=torch.bfloat16):
            imuon.imuon_step(A, B, GA, GB, {}, .1, polar_method='svd')
        torch.testing.assert_close(A, expected_A.bfloat16(), atol=0, rtol=0)
        torch.testing.assert_close(B, expected_B.bfloat16(), atol=0, rtol=0)

    def test_round_averages_before_lmo_and_uses_current_scheduler_lr(self):
        model = ToyModel(self.A, self.B)
        options = dict(server_name='server', client_epoch=1, train_stats=True,
                       server_momentum=.95, imuon_polar='svd', imuon_ns_steps=10)
        graphs = SimpleNamespace(grad_norm=[])
        optimizer = torch.optim.SGD([model.A, model.B], lr=.1)
        scheduler = SimpleNamespace(step=lambda: optimizer.param_groups[0].update(
            lr=optimizer.param_groups[0]['lr'] / 2))
        reference_A, reference_B = self.A.clone(), self.B.clone()
        reference_state = {}
        old_base = model.layer.base_layer.weight.detach().clone()
        gradients = [self.G, .2 - .4 * self.G]

        def collect(*args, **kwargs):
            self.assertIs(options['local_update_ON'], False)
            self.assertEqual(kwargs['exclude_from_copy'], ('imuon_state',))
            for i, G in enumerate(gradients):
                args[10](i, {'layer.lora_A.server.weight': 2 * model.B.T @ G,
                             'layer.lora_B.server.weight': 2 * G @ model.A.T})
            return len(gradients)

        with patch.object(imuon, 'collect_client_grads', side_effect=collect), \
                contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2):
                G = 2 * sum(gradients) / len(gradients)
                imuon.imuon_step(reference_A, reference_B, reference_B.T @ G,
                                 G @ reference_A.T, reference_state,
                                 optimizer.param_groups[0]['lr'], .95, 'svd')
                imuon.federated_imuon(model, None, None, graphs, 'cpu', [],
                                      optimizer, scheduler, 1., options, {}, 0)
                torch.testing.assert_close(model.A, reference_A)
                torch.testing.assert_close(model.B, reference_B)
        self.assertEqual(len(graphs.grad_norm), 2)
        self.assertEqual(optimizer.param_groups[0]['lr'], .025)
        torch.testing.assert_close(model.layer.base_layer.weight, old_base, atol=0, rtol=0)

    def test_existing_initializer_produces_full_rank_and_preserves_effective_weight(self):
        # Load only this production helper so the test does not require PEFT.
        path = ROOT / 'arch/lora.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'init_lora_uniform_sv')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        model = ToyModel(self.A, self.B)
        original = model.layer.base_layer.weight.detach().clone()
        namespace['init_lora_uniform_sv'](model, 'server', 3, 6, .8)
        torch.testing.assert_close(model.layer.base_layer.weight + 2 * model.B @ model.A,
                                   original, atol=2e-7, rtol=2e-6)
        imuon.imuon_step(model.A, model.B, model.B.T @ self.G,
                         self.G @ model.A.T, {}, .01)

    def test_cli_accepts_imuon_and_main_dispatches_to_new_module(self):
        tree = ast.parse((ROOT / 'main.py').read_text())
        parser = argparse.ArgumentParser()
        flags = {'--fedlora_avg', '--momentum', '--imuon_polar', '--imuon_ns_steps'}
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
                 and isinstance(n.args[0], ast.Constant) and n.args[0].value in flags]
        for call in calls:
            exec(compile(ast.Expression(body=call), 'cli', 'eval'), {'parser': parser})
        args = parser.parse_args(['--fedlora_avg', 'imuon'])
        self.assertEqual((args.momentum, args.imuon_polar, args.imuon_ns_steps), (0., 'ns', 10))
        args = parser.parse_args(['--fedlora_avg', 'imuon', '--momentum', '0.95'])
        self.assertEqual(args.momentum, .95)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'federated_lora')
        namespace = dict(train_graphs=None, model_params=None)
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'dispatch', 'exec'), namespace)
        with patch.object(imuon, 'federated_imuon') as run:
            namespace['federated_lora'](None, None, None, None, None, None, None,
                                        1., {'fedlora_avg': 'imuon'}, 2)
            run.assert_called_once()
        from optimizer.load_optimizer import GRAD_ONLY_FEDLORA
        self.assertIn('imuon', GRAD_ONLY_FEDLORA)


if __name__ == '__main__':
    unittest.main()
