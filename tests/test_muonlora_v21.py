"""Numerical regressions for MuonLoRA v21.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_muonlora_v21.py' -v
"""

import unittest

import torch
from torch import nn

from optimizer.muonlora import (
    MuonLoRAV21Config,
    _approximate_muon_factors,
    _init_state,
    _retract_A,
    _retract_B,
    adapter_layers,
    adapter_metadata,
    muonlora_v21_step,
)


class ToyLoRALayer(nn.Module):
    """A tiny layer with PEFT's parameter layout and orientation metadata."""

    def __init__(self, m, n, rank, transposed):
        super().__init__()
        self.fan_in_fan_out = transposed
        self.scaling = {"server": 0.5}
        self.base_layer = nn.Linear(m, n, bias=False) if transposed else nn.Linear(n, m, bias=False)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = nn.ModuleDict({"server": nn.Linear(n, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"server": nn.Linear(rank, m, bias=False)})

    def effective_weight(self):
        W = self.base_layer.weight.detach().double()
        W = W.T if self.fan_in_fan_out else W
        A = self.lora_A["server"].weight.detach().double()
        B = self.lora_B["server"].weight.detach().double()
        return W + self.scaling["server"] * B @ A


class MuonLoRAV21Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def test_qr_retraction_preserves_signed_orthonormal_frames(self):
        radius = 1.3
        Q = torch.linalg.qr(torch.randn(8, 3, dtype=torch.float64))[0]
        for signs in ([1., 1., 1.], [-1., 1., -1.]):
            B = radius * Q * torch.tensor(signs)
            torch.testing.assert_close(_retract_B(B, radius), B, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(_retract_A(B.T, radius), B.T, atol=1e-12, rtol=1e-12)
            perturbation = 1e-8 * torch.randn_like(B)
            self.assertLess((_retract_B(B + perturbation, radius) - B).norm().item(), 1e-6)

    def test_qr_retraction_preserves_column_and_row_spaces(self):
        B = torch.randn(8, 3, dtype=torch.float64)
        result = _retract_B(B, 1.3)
        torch.testing.assert_close(result.T @ result, 1.3**2 * torch.eye(3, dtype=B.dtype))
        torch.testing.assert_close(result @ result.T / 1.3**2, B @ torch.linalg.pinv(B))
        torch.testing.assert_close(_retract_A(B.T, 1.3), result.T)

    def test_muon_removes_zero_and_negligible_singular_directions(self):
        B = torch.eye(4, dtype=torch.float64)[:, :2]
        M_A = torch.zeros(2, 3, dtype=torch.float64)
        M_A[0, 0] = 1.0
        expected = torch.zeros(4, 3, dtype=torch.float64)
        expected[0, 0] = 1.0
        for small in (0., 1e-20):
            M_A[1, 1] = small
            U, V = _approximate_muon_factors(B, M_A, B)
            self.assertEqual(U.shape, (4, 2))
            self.assertEqual(V.shape, (2, 3))
            torch.testing.assert_close(U @ V, expected, atol=1e-14, rtol=1e-14)
        U, V = _approximate_muon_factors(B, torch.zeros_like(M_A), B)
        self.assertEqual(torch.count_nonzero(U @ V).item(), 0)

    def test_muon_full_rank_matches_dense_matrix_sign(self):
        B = torch.linalg.qr(torch.randn(8, 3, dtype=torch.float64))[0]
        M_A = torch.randn(3, 7, dtype=torch.float64)
        M_B = torch.randn(8, 3, dtype=torch.float64)
        H = M_B @ torch.linalg.pinv(B.T @ M_B) @ M_A
        left, _, right = torch.linalg.svd(H, full_matrices=False)
        expected = left[:, :3] @ right[:3]
        U, V = _approximate_muon_factors(B, M_A, M_B)
        torch.testing.assert_close(U @ V, expected, atol=1e-11, rtol=1e-11)

    def test_effective_update_in_both_base_orientations(self):
        for m, n in ((5, 5), (7, 5)):
            for transposed in (False, True):
                for active, both in (("A", False), ("B", False), ("A", True)):
                    with self.subTest(shape=(m, n), transposed=transposed, active=active, both=both):
                        layer = ToyLoRALayer(m, n, 2, transposed)
                        model = nn.ModuleDict({"layer": layer})
                        layers = adapter_layers(model, "server")
                        scalings, orientations = adapter_metadata(model, "server")
                        self.assertEqual(orientations["layer.base_layer"], transposed)
                        params = dict(model.named_parameters())
                        state, _ = _init_state(params, layers)
                        old = layer.effective_weight()
                        old_A = layer.lora_A["server"].weight.detach().clone()
                        old_B = layer.lora_B["server"].weight.detach().clone()
                        gA = {"layer.base_layer": torch.randn(2, n, dtype=torch.float64)}
                        gB = {"layer.base_layer": torch.randn(m, 2, dtype=torch.float64)}
                        U, V = _approximate_muon_factors(old_B.double(), gA["layer.base_layer"], gB["layer.base_layer"])
                        muonlora_v21_step(
                            params, layers, scalings, state, gA, gB,
                            0.03, 0.9, 2., 1., active, True,
                            MuonLoRAV21Config(update_both_factors=both),
                            fan_in_fan_out=orientations)
                        expected = old - 0.03 * (m / n)**0.5 * U @ V
                        torch.testing.assert_close(layer.effective_weight(), expected, atol=4e-7, rtol=4e-7)
                        if not both:
                            fixed = layer.lora_A["server"].weight if active == "B" else layer.lora_B["server"].weight
                            torch.testing.assert_close(fixed, old_A if active == "B" else old_B, atol=0, rtol=0)

    def test_initialization_preserves_effective_weight_in_both_orientations(self):
        from arch.lora import init_lora_uniform_sv

        for m, n in ((5, 5), (7, 5)):
            for transposed in (False, True):
                with self.subTest(shape=(m, n), transposed=transposed):
                    layer = ToyLoRALayer(m, n, 2, transposed)
                    before = layer.base_layer.weight.detach().double().clone()
                    before = before.T if transposed else before
                    init_lora_uniform_sv(nn.ModuleDict({"layer": layer}), "server", 2, 1., 1.3)
                    torch.testing.assert_close(layer.effective_weight(), before, atol=4e-7, rtol=4e-7)


if __name__ == "__main__":
    unittest.main()
