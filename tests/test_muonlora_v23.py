"""CPU regressions for the legacy v20-derived projected Muon split.

Load production definitions via AST to avoid importing PEFT, datasets and the
CLI training stack. The integration test executes federated_muonlora itself;
only client gradient collection, optimizer construction and GPU logging are
stubbed. Run: python -m unittest discover -s tests -p 'test_muonlora_v23.py' -v
"""

import ast
import contextlib
import io
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


NS = dict(torch=torch, np=np, math=math)
load_functions(ROOT / 'optimizer/fedlora.py', {
    '_projected_muon_factor_split', 'get_muonlora_hparams',
    '_advance_alternating_phase', '_factor_alignment_ef_state',
    '_factor_alignment_error_step', '_low_rank_fro_norm', 'federated_muonlora',
}, NS)
load_functions(ROOT / 'arch/lora.py', {'merge_to_base'}, NS)
split = NS['_projected_muon_factor_split']
hparams = NS['get_muonlora_hparams']


class ToyModel(nn.Module):
    def __init__(self, B, A):
        super().__init__()
        self.layer = nn.Module()
        self.layer.base_layer = nn.Linear(A.shape[1], B.shape[0], bias=False).double()
        self.layer.lora_A = nn.ModuleDict()
        self.layer.lora_B = nn.ModuleDict()
        for adapter in ('server', 'muon_update', 'orth_correction'):
            self.layer.lora_A[adapter] = nn.Linear(A.shape[1], A.shape[0], bias=False).double()
            self.layer.lora_B[adapter] = nn.Linear(B.shape[1], B.shape[0], bias=False).double()
        with torch.no_grad():
            self.layer.lora_A['server'].weight.copy_(A)
            self.layer.lora_B['server'].weight.copy_(B)
        self.set_adapter('server')

    def set_adapter(self, adapter):
        for name, param in self.named_parameters():
            param.requires_grad_(f'.{adapter}.' in name)

    def effective_weight(self, scaling):
        return (self.layer.base_layer.weight.detach()
                + scaling * self.layer.lora_B['server'].weight.detach()
                @ self.layer.lora_A['server'].weight.detach())


class MuonLoRAV23Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.gamma = 0.8
        self.B = self.gamma * torch.linalg.qr(torch.randn(7, 3, dtype=torch.float64))[0]
        self.A = self.gamma * torch.linalg.qr(torch.randn(5, 3, dtype=torch.float64))[0].T
        self.U = torch.linalg.qr(torch.randn(7, 3, dtype=torch.float64))[0]
        self.V = torch.linalg.qr(torch.randn(5, 3, dtype=torch.float64))[0].T

    def test_hparams_inherit_v20_and_cli_accepts_v23(self):
        self.assertEqual(hparams('muonlora_v23')[:16], hparams('muonlora_v20')[:16])
        self.assertEqual(hparams('muonlora_v23')[-2:], (True, True))
        self.assertFalse(hparams('muonlora_v20')[-2])
        tree = ast.parse((ROOT / 'main.py').read_text())
        argument = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.Call) and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == '--fedlora_avg')
        choices = next(ast.literal_eval(k.value) for k in argument.keywords if k.arg == 'choices')
        self.assertIn('muonlora_v23', choices)

    def test_shortcut_matches_gram_inverse_without_calling_pinv(self):
        for update_B in (False, True):
            exact = split(self.B, self.A, self.U, self.V, 0.04, update_B, False, self.gamma)
            with patch.object(torch.linalg, 'pinv', side_effect=AssertionError('pinv called')):
                fast = split(self.B, self.A, self.U, self.V, 0.04, update_B, True, self.gamma)
            for actual, expected in zip(fast, exact):
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_factor_move_and_residual_match_dense_formula(self):
        for update_B in (False, True):
            for shortcut in (False, True):
                for alpha in (0.0, 1.0, 30.0):
                    # Includes nonunit LoRA scaling and optional shape scaling.
                    s, lr = 0.5, 0.003
                    U = -lr / s * math.sqrt(7 / 5) * self.U
                    moved, left, right = split(self.B, self.A, U, self.V,
                                               alpha, update_B, shortcut, self.gamma)
                    dense = U @ self.V
                    if update_B:
                        expected = self.B + alpha * dense @ torch.linalg.pinv(self.A)
                        change = (moved - self.B) @ self.A
                    else:
                        expected = self.A + alpha * torch.linalg.pinv(self.B) @ dense
                        change = self.B @ (moved - self.A)
                    torch.testing.assert_close(moved, expected, atol=1e-12, rtol=1e-12)
                    torch.testing.assert_close(s * (change + left @ right),
                                               s * dense, atol=1e-12, rtol=1e-12)

    def test_exact_inverse_handles_nonorthogonal_and_rank_deficient_factors(self):
        for rank_deficient in (False, True):
            B = self.B @ torch.diag(torch.tensor([0.5, 1.2, 2.0], dtype=torch.float64))
            A = torch.diag(torch.tensor([1.3, 0.6, 2.0], dtype=torch.float64)) @ self.A
            if rank_deficient:
                B[:, 2] = B[:, 0]
                A[2] = A[0]
            for update_B in (False, True):
                moved, _, _ = split(B, A, self.U, self.V, 0.03, update_B, False, self.gamma)
                expected = (B + 0.03 * self.U @ self.V @ torch.linalg.pinv(A)
                            if update_B else A + 0.03 * torch.linalg.pinv(B) @ self.U @ self.V)
                torch.testing.assert_close(moved, expected, atol=1e-12, rtol=1e-12)

    def test_update_is_invariant_to_muon_basis(self):
        R = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
        for update_B in (False, True):
            for shortcut in (False, True):
                old = split(self.B, self.A, self.U, self.V, 0.03, update_B, shortcut, self.gamma)
                new = split(self.B, self.A, self.U @ R, R.T @ self.V,
                            0.03, update_B, shortcut, self.gamma)
                torch.testing.assert_close(old[0], new[0], atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(old[1] @ old[2], new[1] @ new[2], atol=1e-12, rtol=1e-12)

    def test_invalid_scale_is_rejected(self):
        for gamma in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                split(self.B, self.A, self.U, self.V, 1, True, True, gamma)

    def test_real_rounds_preserve_muon_step_and_transport_both_histories(self):
        for shortcut in (False, True):
            for scaled in (False, True):
                model = ToyModel(self.B, self.A)
                opts = dict(fedlora_avg='muonlora_v23', lora_init_scale=self.gamma,
                            server_name='server', output_layer_name='', client_num=1,
                            client_opt_name='sgd', client_epoch=1, client_partial=1,
                            client_momentum=0, client_weight_decay=0, lr_decay=1,
                            epochs_lr_decay=[], train_stats=False, server_momentum=0.95,
                            lora_rank=3, lora_alpha=1.5, muonlora_switch_interval=1,
                            muonlora_merge_alpha=30, muonlora_scaled=scaled,
                            model_name='meta-llama/Llama-3.2-3B')
                optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
                graphs = types.SimpleNamespace(loader_iter=0)
                scheduler = types.SimpleNamespace(step=lambda: None)
                main_stub, util_stub = types.ModuleType('main'), types.ModuleType('utilities')
                gradient = torch.randn(7, 5, dtype=torch.float64)

                def train(*args):
                    A = model.layer.lora_A['server'].weight.detach()
                    B = model.layer.lora_B['server'].weight.detach()
                    return model, {'layer.lora_A.server.weight': B.T @ gradient,
                                   'layer.lora_B.server.weight': gradient @ A.T}

                main_stub.train = train
                util_stub.get_gpu_memory = lambda: None
                util_stub.principal_angle = lambda *args: None
                captured = []

                def capture_split(*args):
                    captured.append((args[2] @ args[3]).clone())
                    return split(*args)

                def config(fedlora_avg_name):
                    values = list(hparams(fedlora_avg_name))
                    values[-1] = shortcut
                    return tuple(values)

                with patch.dict(sys.modules, main=main_stub, utilities=util_stub), \
                     patch.dict(NS, load_optimizer=lambda *args: (None, None, None),
                                get_muonlora_hparams=config,
                                _projected_muon_factor_split=capture_split), \
                     contextlib.redirect_stdout(io.StringIO()):
                    for step in range(1, 5):
                        old_A = model.layer.lora_A['server'].weight.detach().clone()
                        old_B = model.layer.lora_B['server'].weight.detach().clone()
                        before = model.effective_weight(0.5).clone()
                        NS['federated_muonlora'](
                            model, 'HF_CrossEntropy', None, 3, graphs, 'cpu',
                            [iter(())], optimizer, scheduler, 1.0, opts, {}, step)
                        self.assertEqual(opts['update_B'], step % 2 == 1)
                        torch.testing.assert_close(model.effective_weight(0.5) - before,
                                                   0.5 * captured[-1], atol=1e-12, rtol=1e-10)
                        A = model.layer.lora_A['server'].weight.detach()
                        B = model.layer.lora_B['server'].weight.detach()
                        torch.testing.assert_close(A @ A.T, self.gamma**2 * torch.eye(3, dtype=A.dtype))
                        torch.testing.assert_close(B.T @ B, self.gamma**2 * torch.eye(3, dtype=B.dtype))
                        torch.testing.assert_close(A if opts['update_B'] else B,
                                                   old_A if opts['update_B'] else old_B)
                self.assertEqual(len(captured), 4)
                self.assertEqual(len(opts['factor_ambient_ef']), 2)
                for state in opts['factor_ambient_ef'].values():
                    self.assertEqual(state['error'].dtype, torch.float64)
                    self.assertTrue(torch.isfinite(state['error']).all())


if __name__ == '__main__':
    unittest.main()
