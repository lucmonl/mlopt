"""Numerical regressions for MuonLoRA v22's normalized factor update.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_muonlora_v22.py' -v

v21 steps the active factor along the raw momentum, so the frame rotation is
proportional to the gradient magnitude and collapses as training converges.
v22 steps along the same momentum rescaled to Frobenius norm sqrt(rank), so the
rotation rate no longer depends on how large the gradients are.
"""

import unittest

import torch
from torch import nn

import optimizer.muonlora as ml

from optimizer.muonlora import (
    MuonLoRAV21Config,
    _approximate_muon_factors,
    _factor_directions,
    _init_state,
    adapter_layers,
    adapter_metadata,
    get_muonlora_config,
    muonlora_v21_step,
)


class ToyLoRALayer(nn.Module):
    """A tiny layer with PEFT's parameter layout and orientation metadata."""

    def __init__(self, m, n, rank):
        super().__init__()
        self.fan_in_fan_out = False
        self.scaling = {"server": 0.5}
        self.base_layer = nn.Linear(n, m, bias=False)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = nn.ModuleDict({"server": nn.Linear(n, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"server": nn.Linear(rank, m, bias=False)})

    def effective_weight(self):
        W = self.base_layer.weight.detach().double()
        A = self.lora_A["server"].weight.detach().double()
        B = self.lora_B["server"].weight.detach().double()
        return W + self.scaling["server"] * B @ A


BASE = "layer.base_layer"


def _factor_step_norm(direction, gradient_scale, active="B", rounds=1, decay=1.0,
                      fresh_directions=False, ratio=False):
    """Frobenius norm of the adapter move.

    With ``ratio`` the last round's move is returned relative to the first, which
    is what shows whether the frame keeps rotating as gradients decay.
    ``fresh_directions`` redraws the gradient each round so the frame is chasing a
    moving target rather than converging onto one fixed direction.
    """
    torch.manual_seed(11)
    m, n, rank = 7, 5, 2
    layer = ToyLoRALayer(m, n, rank)
    model = nn.ModuleDict({"layer": layer})
    layers = adapter_layers(model, "server")
    scalings, orientations = adapter_metadata(model, "server")
    params = dict(model.named_parameters())
    state, _ = _init_state(params, layers)
    config = MuonLoRAV21Config(factor_direction=direction)

    generator = torch.Generator().manual_seed(5)
    base_gA = torch.randn(rank, n, dtype=torch.float64, generator=generator)
    base_gB = torch.randn(m, rank, dtype=torch.float64, generator=generator)
    history = []
    for step in range(rounds):
        if fresh_directions:
            base_gA = torch.randn(rank, n, dtype=torch.float64, generator=generator)
            base_gB = torch.randn(m, rank, dtype=torch.float64, generator=generator)
        shrink = decay ** step
        gA = {BASE: base_gA * gradient_scale * shrink}
        gB = {BASE: base_gB * gradient_scale * shrink}
        before_A = params["layer.lora_A.server.weight"].detach().clone()
        before_B = params["layer.lora_B.server.weight"].detach().clone()
        muonlora_v21_step(
            params, layers, scalings, state, gA, gB,
            0.03, 0.9, 2.0, 1.0, active, True, config,
            fan_in_fan_out=orientations)
        history.append(
            ((params["layer.lora_A.server.weight"] - before_A).norm().item() ** 2
             + (params["layer.lora_B.server.weight"] - before_B).norm().item() ** 2) ** 0.5)
    if not ratio:
        return history[-1]
    # Round 0 is a transient: the momentum is still zero and the factors have
    # not been retracted yet, so it is several times any later step. Measure the
    # decay from round 1 onward instead.
    return (sum(history[-5:]) / 5) / history[1]


class MuonLoRAV22Tests(unittest.TestCase):
    def test_v22_selects_the_normalized_momentum_direction(self):
        self.assertEqual(
            get_muonlora_config("muonlora_v22").factor_direction, "normalized_momentum")
        self.assertEqual(get_muonlora_config("muonlora_v21").factor_direction, "momentum")
        with self.assertRaises(NotImplementedError):
            get_muonlora_config("muonlora_v999")

    def test_muon_factors_are_scale_invariant(self):
        """U, V do not depend on how large the momenta are."""
        B = torch.linalg.qr(torch.randn(8, 3, dtype=torch.float64))[0]
        M_A = torch.randn(3, 7, dtype=torch.float64)
        M_B = torch.randn(8, 3, dtype=torch.float64)
        U, V = _approximate_muon_factors(B, M_A, M_B)
        for scale in (1e-4, 1e4):
            U_s, V_s = _approximate_muon_factors(B, scale * M_A, scale * M_B)
            torch.testing.assert_close(U_s @ V_s, U @ V, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(
            V.norm(), torch.tensor(3.0, dtype=torch.float64).sqrt(), atol=1e-10, rtol=1e-10)

    def test_v22_frame_rotation_is_independent_of_gradient_scale(self):
        # rounds=2 so the measured move is a real gradient step. Round 0 is the
        # retraction pulling the random init onto the sphere, which is identical
        # for every gradient scale and would make this pass vacuously.
        small = _factor_step_norm("normalized_momentum", 1e-3, rounds=2)
        large = _factor_step_norm("normalized_momentum", 1e3, rounds=2)
        self.assertAlmostEqual(small, large, places=10)
        self.assertGreater(small, 1e-3)

    def test_direction_norm_is_scale_free_only_for_the_normalized_modes(self):
        """The mechanism behind the fix, isolated from the retraction.

        v21's step direction is the raw momentum, so as gradients shrink the
        frame rotation shrinks with them and the adapter subspace freezes.
        """
        torch.manual_seed(0)
        m, n, rank = 8, 6, 3
        B = torch.linalg.qr(torch.randn(m, rank, dtype=torch.float64))[0]
        M_A = torch.randn(rank, n, dtype=torch.float64)
        M_B = torch.randn(m, rank, dtype=torch.float64)
        expected = torch.tensor(float(rank), dtype=torch.float64).sqrt()

        baseline = {}
        for scale in (1e-6, 1.0, 1e6):
            U, V = _approximate_muon_factors(B, scale * M_A, scale * M_B)
            for mode in ("muon", "normalized_momentum", "momentum"):
                config = MuonLoRAV21Config(factor_direction=mode)
                d_A, d_B = _factor_directions(
                    config, scale * M_A, scale * M_B,
                    scale * M_A, scale * M_B, U, V)
                if mode == "momentum":
                    baseline.setdefault(mode, d_A.norm() / scale)
                    # grows without bound with the gradient magnitude
                    torch.testing.assert_close(
                        d_A.norm() / scale, baseline[mode], atol=1e-8, rtol=1e-8)
                else:
                    torch.testing.assert_close(d_A.norm(), expected, atol=1e-10, rtol=1e-10)
                    torch.testing.assert_close(d_B.norm(), expected, atol=1e-10, rtol=1e-10)

    def test_v21_frame_rotation_tracks_gradient_scale(self):
        """The behaviour v22 is meant to fix, in the small-step regime.

        Large gradients are not a useful probe here: the retraction bounds the
        move by the manifold diameter, so v21's step saturates instead of
        scaling. The regime that matters is the converged one, where steps are
        small and v21's rotation decays linearly with the gradient.
        """
        # rounds=2 for the same reason as the v22 case above.
        small = _factor_step_norm("momentum", 1e-5, rounds=2)
        large = _factor_step_norm("momentum", 1e-3, rounds=2)
        self.assertGreater(large / small, 50.0)

    def test_v22_frame_keeps_moving_as_gradients_decay(self):
        """With a moving target and shrinking gradients, v21 freezes; v22 does not."""
        v21_ratio = _factor_step_norm(
            "momentum", 1.0, rounds=30, decay=0.7, fresh_directions=True, ratio=True)
        v22_ratio = _factor_step_norm(
            "normalized_momentum", 1.0, rounds=30, decay=0.7,
            fresh_directions=True, ratio=True)
        # Measured: v21 0.029, v22 0.381 -- a 13x improvement. v22 is not
        # perfectly flat because normalizing fixes the magnitude but the
        # *direction* of M still stabilizes as the heavy-ball average smooths,
        # so consecutive steps grow more parallel and the retraction cancels
        # more of them. factor_direction="muon" scores 0.769 for that reason.
        self.assertLess(v21_ratio, 0.10)
        self.assertGreater(v22_ratio, 0.25)
        self.assertGreater(v22_ratio / v21_ratio, 5.0)

    def test_v22_keeps_the_two_momenta_separate(self):
        """Transported M drives the factor step; untransported N feeds U, V."""
        self.assertTrue(get_muonlora_config("muonlora_v22").muon_from_unaligned_momentum)
        self.assertFalse(get_muonlora_config("muonlora_v21").muon_from_unaligned_momentum)

        torch.manual_seed(37)
        m, n, rank = 7, 5, 2
        layer = ToyLoRALayer(m, n, rank)
        model = nn.ModuleDict({"layer": layer})
        layers = adapter_layers(model, "server")
        scalings, orientations = adapter_metadata(model, "server")
        params = dict(model.named_parameters())
        config = get_muonlora_config("muonlora_v22")
        state, _ = _init_state(params, layers, config)
        self.assertIn("N_A", state["layers"][BASE])

        beta, eta, multiplier = 0.9, 0.03, 2.0
        grads = [(torch.randn(rank, n, dtype=torch.float64),
                  torch.randn(m, rank, dtype=torch.float64)) for _ in range(2)]

        # Round 1 moves B, so round 2's M_A is transported while N_A is not.
        muonlora_v21_step(
            params, layers, scalings, state, {BASE: grads[0][0]}, {BASE: grads[0][1]},
            eta, beta, multiplier, 1.0, "B", True, config, fan_in_fan_out=orientations)

        N_A_prev = state["layers"][BASE]["N_A"].clone()
        N_B_prev = state["layers"][BASE]["N_B"].clone()
        B_round2 = params["layer.lora_B.server.weight"].detach().double().clone()
        before = layer.effective_weight()

        captured = {}
        original = ml._factor_directions

        def spy(cfg, M_A, M_B, g_A, g_B, U, V):
            captured["M_A"] = M_A.clone()
            captured["M_B"] = M_B.clone()
            return original(cfg, M_A, M_B, g_A, g_B, U, V)

        ml._factor_directions = spy
        try:
            muonlora_v21_step(
                params, layers, scalings, state, {BASE: grads[1][0]}, {BASE: grads[1][1]},
                eta, beta, multiplier, 1.0, "B", True, config,
                fan_in_fan_out=orientations)
        finally:
            ml._factor_directions = original

        # N follows plain heavy-ball, with no transport applied.
        expected_N_A = beta * N_A_prev + grads[1][0]
        expected_N_B = beta * N_B_prev + grads[1][1]
        torch.testing.assert_close(state["layers"][BASE]["N_A"], expected_N_A)
        torch.testing.assert_close(state["layers"][BASE]["N_B"], expected_N_B)

        # The factor step was handed the transported momenta, not N.
        torch.testing.assert_close(captured["M_A"], state["layers"][BASE]["M_A"])
        torch.testing.assert_close(captured["M_B"], state["layers"][BASE]["M_B"])
        self.assertGreater(
            (captured["M_A"] - expected_N_A).norm().item(), 1e-6,
            "transport was a no-op here, so this test would not discriminate")

        # The base weight moved along U, V built from the *unaligned* momenta.
        U, V = _approximate_muon_factors(B_round2, expected_N_A, expected_N_B)
        expected = before - eta * (m / n) ** 0.5 * U @ V
        torch.testing.assert_close(layer.effective_weight(), expected, atol=4e-7, rtol=4e-7)

        # ...and would not have matched had they been built from the aligned ones.
        U_aligned, V_aligned = _approximate_muon_factors(
            B_round2, state["layers"][BASE]["M_A"], state["layers"][BASE]["M_B"])
        self.assertGreater((U_aligned @ V_aligned - U @ V).norm().item(), 1e-6)

    def test_v21_keeps_a_single_momentum(self):
        torch.manual_seed(37)
        m, n, rank = 7, 5, 2
        layer = ToyLoRALayer(m, n, rank)
        model = nn.ModuleDict({"layer": layer})
        layers = adapter_layers(model, "server")
        scalings, orientations = adapter_metadata(model, "server")
        params = dict(model.named_parameters())
        config = get_muonlora_config("muonlora_v21")
        state, _ = _init_state(params, layers, config)
        self.assertNotIn("N_A", state["layers"][BASE])
        muonlora_v21_step(
            params, layers, scalings, state,
            {BASE: torch.randn(rank, n, dtype=torch.float64)},
            {BASE: torch.randn(m, rank, dtype=torch.float64)},
            0.03, 0.9, 2.0, 1.0, "B", True, config, fan_in_fan_out=orientations)
        self.assertNotIn("N_A", state["layers"][BASE])

    def test_v22_preserves_the_effective_update_identity(self):
        for active in ("A", "B"):
            with self.subTest(active=active):
                torch.manual_seed(37)
                m, n, rank = 7, 5, 2
                layer = ToyLoRALayer(m, n, rank)
                model = nn.ModuleDict({"layer": layer})
                layers = adapter_layers(model, "server")
                scalings, orientations = adapter_metadata(model, "server")
                params = dict(model.named_parameters())
                state, _ = _init_state(params, layers)
                old = layer.effective_weight()
                old_B = layer.lora_B["server"].weight.detach().clone()
                gA = {BASE: torch.randn(rank, n, dtype=torch.float64)}
                gB = {BASE: torch.randn(m, rank, dtype=torch.float64)}
                U, V = _approximate_muon_factors(old_B.double(), gA[BASE], gB[BASE])
                muonlora_v21_step(
                    params, layers, scalings, state, gA, gB,
                    0.03, 0.9, 2.0, 1.0, active, True,
                    get_muonlora_config("muonlora_v22"),
                    fan_in_fan_out=orientations)
                expected = old - 0.03 * (m / n) ** 0.5 * U @ V
                torch.testing.assert_close(
                    layer.effective_weight(), expected, atol=4e-7, rtol=4e-7)


if __name__ == "__main__":
    unittest.main()
