"""CPU tests: python -m unittest discover -s tests -p 'test_signmuon_server.py' -v."""

import argparse
import ast
import contextlib
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

import optimizer.signmuon_server as sm
import optimizer.federated_train_single_step as collection
from optimizer.load_optimizer import GRAD_ONLY_FEDLORA


ROOT = Path(__file__).resolve().parents[1]


class SignMuonServerTests(unittest.TestCase):
    def test_sign_before_average_and_server_momentum(self):
        p = nn.Parameter(torch.zeros(2, 2))
        state = sm.SignMuonServerState([('w', p)])
        groups = [{'params': [p], 'lr': 0.1}]
        gradients = [torch.full_like(p, 100), torch.full_like(p, -1), torch.full_like(p, -2)]
        state.start_round()
        for client, g in enumerate(gradients):
            state.accumulate_client(client, {'w': g})
        with patch.object(sm, 'muon_direction', side_effect=lambda m, method: m.clone()) as muon:
            state.apply_step([('w', p)], groups, 3, 0.9, 'none')
            torch.testing.assert_close(muon.call_args.args[0], torch.full_like(p, -1 / 3))
            torch.testing.assert_close(p, torch.full_like(p, 0.1))
            state.start_round()
            state.accumulate_client(0, {'w': torch.ones_like(p)})
            state.apply_step([('w', p)], groups, 1, 0.9, 'none')
            torch.testing.assert_close(state.momentum['w'], torch.full_like(p, 0.7))
            torch.testing.assert_close(p, torch.zeros_like(p))

    def test_normalization_preserves_norm_and_sign_including_zeros(self):
        for direction in (torch.tensor([[3., -4.], [0., 0.]]),
                          torch.randn(7, 3), torch.zeros(2, 4), torch.tensor(2.)):
            result = sm.normalized_sign(direction)
            torch.testing.assert_close(result.norm(), direction.norm())
            torch.testing.assert_close(result.sign(), direction.sign())
            torch.testing.assert_close(sm.normalized_sign(direction, 'none'), direction.sign())

    def test_zero_and_rank_deficient_muon_do_not_invent_directions(self):
        for method in ('ns5', 'svd'):
            for shape in ((3, 5), (5, 3), (4,), ()):
                result = sm.muon_direction(torch.zeros(shape), method)
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(result.count_nonzero(), 0)
        matrix = torch.tensor([[4., 0.], [0., 0.]])
        torch.testing.assert_close(sm.muon_direction(matrix, 'svd'), torch.diag(torch.tensor([1., 0.])))

    def test_lmo_and_normalization_match_dense_reference(self):
        torch.manual_seed(4)
        matrix = torch.randn(5, 3)
        U, _, Vh = torch.linalg.svd(matrix, full_matrices=False)
        Q = U @ Vh
        torch.testing.assert_close(sm.muon_direction(matrix, 'svd'), Q)
        expected = Q.sign() * Q.norm() / Q.sign().norm()
        torch.testing.assert_close(sm.normalized_sign(Q), expected)
        for method in ('svd', 'ns5'):
            result = sm.muon_direction(matrix, method)
            self.assertEqual(result.shape, matrix.shape)
            torch.testing.assert_close(sm.normalized_sign(result).norm(), result.norm())

    def test_parameter_dtype_group_lrs_and_decoupled_weight_decay(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            p = nn.Parameter(torch.ones(3, dtype=dtype))
            q = nn.Parameter(torch.ones(2, dtype=dtype))
            named = [('p', p), ('q', q)]
            state = sm.SignMuonServerState(named)
            state.accumulate_client(0, {'p': torch.ones_like(p), 'q': -torch.ones_like(q)})
            groups = [{'params': [p], 'lr': 0.125, 'weight_decay': 0.5},
                      {'params': [q], 'lr': 0.25, 'weight_decay': 0.0}]
            state.apply_step(named, groups, 1, 0, method='svd')
            torch.testing.assert_close(p, torch.full_like(p, 0.8125))
            torch.testing.assert_close(q, torch.full_like(q, 1.25))
            self.assertEqual(p.dtype, dtype)
            # server buffers track the model dtype, as ef14muon's do
            self.assertEqual(state.momentum['p'].dtype, dtype)
            self.assertEqual(state.votes['p'].dtype, dtype)

    def test_votes_stay_exact_and_precision_warnings_fire(self):
        # a sum of +-1 over client_num clients is integral, and bf16 holds
        # every integer to 256, so low-width votes are exact, not approximate
        p = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
        state = sm.SignMuonServerState([('w', p)])
        for client in range(200):
            state.accumulate_client(client, {'w': torch.ones_like(p)})
        torch.testing.assert_close(state.votes['w'].float(), torch.full((4,), 200.))

        def warnings(beta, clients):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                state.check_precision(beta, clients)
            return buffer.getvalue()

        self.assertEqual(warnings(0.9, 8), '')
        self.assertIn('too close to 1', warnings(0.999, 8))
        self.assertIn('no longer exact', warnings(0.9, 1000))
        # fp32 parameters have no practical ceiling on either knob
        wide = sm.SignMuonServerState([('w', nn.Parameter(torch.zeros(4)))])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            wide.check_precision(0.999, 1000)
        self.assertEqual(buffer.getvalue(), '')

    def test_invalid_configuration_and_nonfinite_gradients(self):
        p = nn.Parameter(torch.zeros(2, 2))
        state = sm.SignMuonServerState([('w', p)])
        with self.assertRaises(FloatingPointError):
            state.accumulate_client(0, {'w': torch.full_like(p, float('nan'))})
        for beta in (-0.1, 1.0, float('nan')):
            with self.assertRaises(ValueError):
                state.apply_step([('w', p)], [{'params': [p], 'lr': 1}], 1, beta)
        with self.assertRaises(ValueError):
            state.apply_step([('w', p)], [], 0, 0)
        with self.assertRaises(ValueError):
            sm.normalized_sign(p, 'invalid')
        with self.assertRaises(ValueError):
            sm.muon_direction(p, 'invalid')

    def test_cli_options_flow_to_dispatch_and_real_client_collection(self):
        tree = ast.parse((ROOT / 'main.py').read_text())
        wanted = {'--fedlora_avg', '--signmuon_normalization', '--signmuon_lmo'}
        parser = argparse.ArgumentParser()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and node.value.args and isinstance(node.value.args[0], ast.Constant)
                    and node.value.args[0].value in wanted):
                exec(compile(ast.Module(body=[node], type_ignores=[]), 'main.py', 'exec'), {'parser': parser})
        defaults = parser.parse_args(['--fedlora_avg', 'signmuon-server'])
        self.assertEqual(defaults.signmuon_normalization, 'rms')
        self.assertEqual(defaults.signmuon_lmo, 'ns5')
        args = parser.parse_args(['--fedlora_avg', 'signmuon-server',
                                  '--signmuon_normalization', 'none', '--signmuon_lmo', 'svd'])
        opts = dict(lora_rank=-1, client_epoch=1, client_num=4, client_partial=0.5,
                    server_momentum=0, train_stats=True, client_opt_name='sgd',
                    client_momentum=0, client_weight_decay=0, lr_decay=1,
                    epochs_lr_decay=[], use_model_grad=True)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == 'opt_params' and isinstance(target.slice, ast.Constant)
                    and target.slice.value in ('fedlora_avg', 'signmuon_normalization', 'signmuon_lmo')):
                exec(compile(ast.Module(body=[node], type_ignores=[]), 'main.py', 'exec'),
                     {'opt_params': opts, 'args': args})
        self.assertIn('signmuon-server', GRAD_ONLY_FEDLORA)
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = Mock()
        graphs = types.SimpleNamespace(loader_iter=0, grad_norm=[])
        namespace = dict(train_graphs=graphs, model_params={}, lora_rank=-1)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'federated_lora')
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'main.py', 'exec'), namespace)
        gradients = [torch.tensor([[2., -3.], [-4., 5.]]), torch.tensor([[7., -1.], [-1., 9.]])]
        loaders = [iter(gradients), gradients]
        main_stub = types.ModuleType('main')

        def train(model, loss_name, criterion, device, loader, optimizer, scheduler, epoch, options):
            self.assertFalse(options['local_update_ON'])
            self.assertNotIn('signmuon_server_state', options)
            torch.testing.assert_close(model.weight, torch.zeros_like(model.weight))
            return model, {'weight': next(loader)}

        main_stub.train = train
        with patch.dict(sys.modules, main=main_stub), \
             patch.object(collection, 'load_optimizer', return_value=(None, None, None)), \
             patch.object(sm, 'select_client_ids', return_value=[1, 3]), \
             patch.object(optimizer, 'step', side_effect=AssertionError('double optimizer step')), \
             contextlib.redirect_stdout(io.StringIO()):
            namespace['federated_lora'](model, None, None, 'cpu', loaders, optimizer,
                                        scheduler, 1, opts, 1)
        expected_mean = (gradients[0].sign() + gradients[1].sign()) / 2
        torch.testing.assert_close(opts['signmuon_server_state'].momentum['weight'], expected_mean)
        expected_step = -0.1 * sm.muon_direction(expected_mean, 'svd').sign()
        torch.testing.assert_close(model.weight, expected_step)
        self.assertEqual(graphs.loader_iter, 2)
        self.assertEqual(len(graphs.grad_norm), 1)
        scheduler.step.assert_called_once()


if __name__ == '__main__':
    unittest.main()
