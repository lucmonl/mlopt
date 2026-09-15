"""MuonLoRA algorithms from v21 onward.

This module is intentionally independent of the historical implementation in
``optimizer/fedlora.py``.  The old versions remain frozen there; new versions
share the small, explicit building blocks below.

v22 is v21 with one building block replaced: the active factor still steps along
its own momentum, but that momentum is rescaled to Frobenius norm sqrt(r) first.
v21 stepped along the raw ``M``, so the frame rotation was proportional to the
gradient magnitude; as training converges the gradients shrink, the adapter frame
freezes, and the rank-r update is left confined to a stale subspace.  Normalizing
fixes the rotation rate without changing the direction v21 chose to move along.

v22 also splits the momentum in two.  The transported (aligned) momenta M_A and
M_B drive the factor step, where the frame correction belongs: A and B move
inside the current frame.  A second pair N_A and N_B accumulates the same
gradients with no transport, and the Muon reconstruction U, V is built from
those instead -- it approximates an ambient weight-space direction, which does
not depend on the adapter frame.

``factor_direction="muon"`` is kept as an ablation: it steps along the
orthonormalized Muon factor instead, which is what v14 did.

v21 keeps v14's approximate-Muon base-weight target and alternating factor
schedule, but changes how the server adapter moves.  If A is active,

    A+ = Retract(A - eta_f M_A),    B+ = B,

and symmetrically for B.  Here M_A/M_B are transported heavy-ball factor
momenta and eta_f = server_lr * muonlora_merge_alpha.  The base weight receives

    Delta W_base = Delta W_muon - s (B+ A+ - B A),

so the effective layer W_base + s B A moves by exactly Delta W_muon, up to the
model's storage precision.  This separates the useful adapter-frame evolution
from the approximate-Muon update applied to the represented dense weight.
"""

from dataclasses import dataclass, replace
import math

import torch

from optimizer.federated_train_single_step import collect_client_grads
from optimizer.riemannion import riemann_layers as adapter_layers


STATE_DTYPE = torch.float64


@dataclass(frozen=True)
class MuonLoRAV21Config:
    """New v21 design choices.

    These toggles are deliberately configured here rather than exposed as a
    growing collection of CLI flags.  Future versions can replace one building
    block at a time while retaining the same federated wrapper.
    """

    # v21 itself is one-sided.  --muonlora_update_both_factors turns the
    # simultaneous-factor extension on for any version; a version that wants it
    # unconditionally sets True here instead.  The CLI flag can only enable it,
    # so a version's own True is never silently overridden.
    update_both_factors: bool = False

    # Direction for the factor step.  False is a useful no-momentum ablation.
    # Only consulted when factor_direction is "momentum".
    use_factor_momentum: bool = True

    # Keep a second, untransported heavy-ball momentum and build the Muon
    # reconstruction (U, V) from it.  The transported momenta still drive the
    # factor step: the frame correction belongs to the A/B update, which moves
    # inside the current frame, not to the reconstruction of the ambient
    # weight-space direction.
    muon_from_unaligned_momentum: bool = False

    # Which direction the active factor steps along:
    #   "momentum"            raw heavy-ball momentum M (v21)
    #   "gradient"            this round's averaged factor gradient
    #   "normalized_momentum" M rescaled to Frobenius norm sqrt(r) (v22)
    #   "muon"                orthonormalized Muon factor, ||.||_F = sqrt(r); v14's rule
    # The last two are scale-free, so the frame keeps rotating at a fixed rate
    # as gradients shrink.
    factor_direction: str = "momentum"

    # Keep active factors at the initialization radius after their raw step.
    retract_updated_factors: bool = True


_VERSION_CONFIGS = {
    "muonlora_v21": MuonLoRAV21Config(),
    "muonlora_v22": MuonLoRAV21Config(
        factor_direction="normalized_momentum",
        muon_from_unaligned_momentum=True,
    ),
}


def get_muonlora_config(fedlora_avg_name):
    """Single source of truth for each new version's design toggles."""
    try:
        return _VERSION_CONFIGS[fedlora_avg_name]
    except KeyError:
        raise NotImplementedError(
            "{} has no MuonLoRA config; known versions: {}".format(
                fedlora_avg_name, ", ".join(sorted(_VERSION_CONFIGS))))


def get_muonlora_v21_hparams():
    """Back-compatible accessor for v21's toggles."""
    return get_muonlora_config("muonlora_v21")


@torch.no_grad()
def _rescaled(M, rank):
    """M scaled to Frobenius norm sqrt(rank).  A zero input stays zero."""
    norm = M.norm()
    if norm.item() == 0.0:
        return M
    return M * (math.sqrt(rank) / norm)


@torch.no_grad()
def _factor_directions(config, M_A, M_B, grad_A, grad_B, U, V):
    """Direction the active factor steps along, per config.factor_direction.

    Returns (direction_A, direction_B) shaped like (A, B).  The "muon" and
    "normalized_momentum" forms have a Frobenius norm that does not depend on
    the gradient magnitude, which is what keeps the adapter frame moving once
    the loss flattens.
    """
    mode = config.factor_direction
    if mode == "muon":
        return V, U
    if mode == "normalized_momentum":
        rank = M_A.shape[0]
        return _rescaled(M_A, rank), _rescaled(M_B, rank)
    if mode == "gradient" or not config.use_factor_momentum:
        return grad_A, grad_B
    if mode != "momentum":
        raise ValueError("unknown factor_direction {!r}".format(mode))
    return M_A, M_B


def adapter_metadata(model, server_name):
    """Read per-layer scaling and base-weight orientation from PEFT.

    In particular, square Conv1D weights cannot be identified by shape.
    """
    scalings, fan_in_fan_out = {}, {}
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and server_name in scaling:
            base = name + ".base_layer"
            scalings[base] = float(scaling[server_name])
            fan_in_fan_out[base] = bool(getattr(module, "fan_in_fan_out", False))
    return scalings, fan_in_fan_out


def _current_factor(server_epoch, switch_interval):
    """Return the active side without mutable/off-by-one phase state.

    The first block updates B, matching v14.  The phase is a pure function of
    the epoch, so resuming from a checkpoint cannot advance it twice.
    """
    if switch_interval <= 0:
        raise ValueError("muonlora_v21 needs --muonlora_switch_interval > 0")
    return "B" if ((server_epoch - 1) // switch_interval) % 2 == 0 else "A"


@torch.no_grad()
def _transport_A_momentum(M_A, B_previous, B_current):
    """Express B_previous M_A in the current B frame (v14 transport)."""
    gram = B_previous.T @ B_previous
    change = B_current.T @ B_previous @ torch.linalg.pinv(gram, rcond=1e-6)
    return change @ M_A


@torch.no_grad()
def _transport_B_momentum(M_B, A_previous, A_current):
    """Express M_B A_previous in the current A frame (v14 transport)."""
    gram = A_previous @ A_previous.T
    change = torch.linalg.pinv(gram, rcond=1e-6) @ A_previous @ A_current.T
    return M_B @ change


@torch.no_grad()
def _approximate_muon_factors(B, M_A, M_B):
    """Return factors of the minimum-norm matrix sign of the reconstruction.

    v14 reconstructs a compatible ambient matrix through

        M_B (B^T M_B)^+ M_A,

    then uses two thin QR factorizations and an r-by-r SVD.  Unlike v14,
    numerically zero singular directions are masked instead of receiving an
    arbitrary unit update.  Factor shapes stay (m, r) and (r, n); masked
    columns of the left factor are zero.
    """
    projected = B.T @ M_B
    projected_inv = torch.linalg.pinv(projected)
    N = (projected_inv @ M_A).T
    if M_B.norm().item() == 0.0 or N.norm().item() == 0.0:
        return torch.zeros_like(M_B), torch.zeros_like(M_A)

    Q_M, R_M = torch.linalg.qr(M_B, mode="reduced")
    Q_N, R_N = torch.linalg.qr(N, mode="reduced")
    U_core, singular_values, Vh_core = torch.linalg.svd(
        R_M @ R_N.T, full_matrices=False)
    # Use the ambient matrix dimensions and the computation dtype for the
    # numerical-rank cutoff, as for a dense SVD of the reconstruction.
    rtol = max(M_B.shape[0], M_A.shape[1]) * torch.finfo(singular_values.dtype).eps
    keep = singular_values > rtol * singular_values[0]
    U = (Q_M @ U_core) * keep.to(Q_M.dtype).unsqueeze(0)
    return U, (Q_N @ Vh_core.T).T


@torch.no_grad()
def _qr_positive_diagonal(X):
    """Q from thin QR with nonnegative diag(R).

    For full-column-rank inputs diag(R) is positive, making this a local QR
    retraction: an already orthonormal input is unchanged.  A zero diagonal
    uses sign +1 rather than zero, so Q retains orthonormal columns even at a
    rank-deficient candidate (where a smooth QR retraction is undefined).
    """
    Q, R = torch.linalg.qr(X, mode="reduced")
    signs = torch.where(R.diagonal() < 0, -1.0, 1.0).to(Q.dtype)
    return Q * signs.unsqueeze(0)


@torch.no_grad()
def _retract_A(A, radius):
    return radius * _qr_positive_diagonal(A.T).T


@torch.no_grad()
def _retract_B(B, radius):
    return radius * _qr_positive_diagonal(B)


def _init_state(params, layers, config=None):
    state = {"last_updated": (), "layers": {}}
    bytes_used = 0
    for base, (name_A, name_B) in layers.items():
        A = params[name_A].data.to(STATE_DTYPE)
        B = params[name_B].data.to(STATE_DTYPE)
        state["layers"][base] = {
            "M_A": torch.zeros_like(A),
            "M_B": torch.zeros_like(B),
            # A pre-update snapshot is retained.  On the next round the
            # difference to the current factor is exactly last round's move.
            "previous_A": A.clone(),
            "previous_B": B.clone(),
        }
        if config is not None and config.muon_from_unaligned_momentum:
            state["layers"][base]["N_A"] = torch.zeros_like(A)
            state["layers"][base]["N_B"] = torch.zeros_like(B)
        bytes_used += sum(
            value.numel() * value.element_size()
            for value in state["layers"][base].values() if torch.is_tensor(value))
    return state, bytes_used


@torch.no_grad()
def _apply_effective_update(base_weight, U, V, eta, muon_scale,
                            A_old, B_old, A_new, B_new, scaling,
                            fan_in_fan_out, active_side=None):
    """Apply the desired Muon update with mandatory adapter compensation.

    Subtracting s(B_new A_new - B_old A_old) ensures that changing the
    adapters preserves the intended effective Muon update.

    When a single factor moved, that difference is s B (A_new - A_old) (or
    s (B_new - B_old) A).  Subtracting the small factor difference *first* and
    multiplying once is both cheaper and far more accurate than differencing
    two adapter products whose own norm dwarfs the step: the two-product form
    carries the full ||s B A|| through the FP32 accumulator and cancels it
    away, costing ~20x the relative error of this form.

    Accumulation is FP32 to avoid a full dense FP64 temporary; the final result
    is cast once to the base parameter dtype.  A/B and optimizer state remain
    FP64 until the actual adapter assignment.  The factor differences are
    formed in FP64, so only the outer product is rounded.

    Returns the squared Frobenius norm of the part of the intended update that
    the base parameter's own dtype could not represent.  This, not the FP32
    accumulation above, is what actually limits the effective-update identity:
    in a bf16 run the step is added to a weight whose ULP can exceed it, and
    then most of the round is silently discarded.  Measuring it is the point of
    the `storage_residual` term in the round log.
    """
    delta = torch.mm(U.float(), V.float())
    delta.mul_(-eta * muon_scale)
    if active_side == "A":
        delta.addmm_(B_old.float(), (A_new - A_old).float(),
                     beta=1.0, alpha=-scaling)
    elif active_side == "B":
        delta.addmm_((B_new - B_old).float(), A_old.float(),
                     beta=1.0, alpha=-scaling)
    else:
        # Both factors moved: no single small difference exists.
        delta.addmm_(B_old.float(), A_old.float(), beta=1.0, alpha=scaling)
        delta.addmm_(B_new.float(), A_new.float(), beta=1.0, alpha=-scaling)

    stored_delta = delta.T if fan_in_fan_out else delta
    if base_weight.shape != stored_delta.shape:
        raise ValueError(
            "base weight {} is incompatible with update {} "
            "(fan_in_fan_out={})".format(
                tuple(base_weight.shape), tuple(stored_delta.shape),
                fan_in_fan_out))
    previous = base_weight.clone()
    base_weight.add_(stored_delta.to(base_weight.dtype))
    # W_after - W_before is exact in the base dtype: the two are far closer than
    # a factor of two, so Sterbenz applies.  This is the update the model
    # actually received, not an estimate of it.
    previous.neg_().add_(base_weight)
    # stored_delta is dead after this point, so it doubles as the residual
    # buffer.  The mixed-dtype in-place subtract casts elementwise and never
    # materializes an FP32 copy of the base weight.
    return stored_delta.sub_(previous).pow_(2).sum(dtype=torch.float64).item()


@torch.no_grad()
def muonlora_v21_step(params, layers, scalings, state, grad_A, grad_B,
                       eta, beta, factor_step_multiplier, radius,
                       active_factor, aspect_scaled, config, *, fan_in_fan_out):
    """Run the server-side v21 transformation; exposed for focused tests."""
    last_updated = set(state["last_updated"])
    update_sides = {"A", "B"} if config.update_both_factors else {active_factor}
    factor_eta = eta * factor_step_multiplier

    step_sq = 0.0
    factor_step_sq = 0.0
    storage_residual_sq = 0.0
    for base, (name_A, name_B) in layers.items():
        A_param, B_param = params[name_A], params[name_B]
        base_name = base + ".weight"
        if base_name not in params:
            raise KeyError("missing base parameter {}".format(base_name))
        if base not in scalings:
            raise KeyError("missing PEFT adapter scaling for {}".format(base))
        if base not in fan_in_fan_out:
            raise KeyError("missing PEFT base-weight orientation for {}".format(base))

        layer_state = state["layers"][base]
        A_old = A_param.data.to(STATE_DTYPE)
        B_old = B_param.data.to(STATE_DTYPE)
        M_A_old = layer_state["M_A"]
        M_B_old = layer_state["M_B"]

        # Transport only the momentum whose partner changed on the last step.
        if "B" in last_updated:
            M_A_old = _transport_A_momentum(
                M_A_old, layer_state["previous_B"], B_old)
        if "A" in last_updated:
            M_B_old = _transport_B_momentum(
                M_B_old, layer_state["previous_A"], A_old)

        M_A = beta * M_A_old + grad_A[base]
        M_B = beta * M_B_old + grad_B[base]
        layer_state["M_A"], layer_state["M_B"] = M_A, M_B
        layer_state["previous_A"] = A_old.clone()
        layer_state["previous_B"] = B_old.clone()

        if config.muon_from_unaligned_momentum:
            # The same heavy-ball recursion without the frame transport.  U, V
            # approximate an ambient weight-space direction, which does not
            # depend on the adapter frame, so they are built from these; the
            # factor step below still uses the transported M_A / M_B.  Created
            # lazily so a checkpoint written before this existed still loads.
            if "N_A" not in layer_state:
                layer_state["N_A"] = torch.zeros_like(M_A)
                layer_state["N_B"] = torch.zeros_like(M_B)
            N_A = beta * layer_state["N_A"] + grad_A[base]
            N_B = beta * layer_state["N_B"] + grad_B[base]
            layer_state["N_A"], layer_state["N_B"] = N_A, N_B
            muon_A, muon_B = N_A, N_B
        else:
            muon_A, muon_B = M_A, M_B

        U, V = _approximate_muon_factors(B_old, muon_A, muon_B)
        muon_scale = math.sqrt(U.shape[0] / V.shape[1]) if aspect_scaled else 1.0

        direction_A, direction_B = _factor_directions(
            config, M_A, M_B, grad_A[base], grad_B[base], U, V)
        A_candidate = A_old - factor_eta * direction_A if "A" in update_sides else A_old
        B_candidate = B_old - factor_eta * direction_B if "B" in update_sides else B_old

        if config.retract_updated_factors:
            A_candidate = _retract_A(A_candidate, radius) if "A" in update_sides else A_candidate
            B_candidate = _retract_B(B_candidate, radius) if "B" in update_sides else B_candidate

        # Compensation must use the exact values that will be stored, including
        # fp16/bf16 rounding, or the effective-update identity is already stale.
        A_stored = A_candidate.to(A_param.dtype)
        B_stored = B_candidate.to(B_param.dtype)
        A_new = A_stored.to(STATE_DTYPE)
        B_new = B_stored.to(STATE_DTYPE)

        storage_residual_sq += _apply_effective_update(
            params[base_name].data, U, V, eta, muon_scale,
            A_old, B_old, A_new, B_new, scalings[base], fan_in_fan_out[base],
            active_side=None if config.update_both_factors else active_factor)
        A_param.data.copy_(A_stored)
        B_param.data.copy_(B_stored)

        # Low-rank Frobenius identities avoid another dense diagnostic tensor.
        step_gram = (U.T @ U) * (V @ V.T).T
        step_sq += (eta * muon_scale) ** 2 * step_gram.sum().item()
        factor_step_sq += (A_new - A_old).pow(2).sum().item()
        factor_step_sq += (B_new - B_old).pow(2).sum().item()

    state["last_updated"] = tuple(sorted(update_sides))
    return step_sq ** 0.5, factor_step_sq ** 0.5, storage_residual_sq ** 0.5


def run_muonlora_round(model, loss_name, criterion, train_graphs, device,
                       train_loaders, server_optimizer,
                       server_lr_scheduler, client_lr, opt_params,
                       model_params, server_epoch):
    """Collect averaged factor gradients and run one MuonLoRA step.

    Shared by every version in ``_VERSION_CONFIGS``; the version only selects
    which building blocks the step uses.
    """
    version = opt_params["fedlora_avg"]
    if opt_params["client_epoch"] != 1:
        raise ValueError("{} needs --client_epoch 1".format(version))
    if opt_params.get("lora_freeze_a", False):
        raise ValueError(
            "{} updates both sides over time; do not use --lora_freeze_a".format(version))

    config = get_muonlora_config(version)
    if opt_params.get("muonlora_update_both_factors", False):
        # The flag enables only; a version whose config already asks for both
        # factors keeps them regardless.
        config = replace(config, update_both_factors=True)
    # v21's key, so an in-flight v21 run still restores from its checkpoint.
    state_key = version
    opt_state_key = version + "_state"
    opt_params["local_update_ON"] = False
    server_name = opt_params["server_name"]
    model.set_adapter(server_name)
    params = dict(model.named_parameters())
    layers = adapter_layers(model, server_name)
    scalings, fan_in_fan_out = adapter_metadata(model, server_name)
    if not layers:
        raise ValueError("{} found no server LoRA layers".format(version))

    covered = {name for pair in layers.values() for name in pair}
    missing = [name for name, param in model.named_parameters()
               if param.requires_grad and name not in covered]
    if missing:
        raise ValueError(
            "{} only steps LoRA pairs; unhandled trainable parameters: {}"
            .format(version, missing[:5]))

    checkpoint_state = server_optimizer.state.get(state_key)
    if opt_state_key not in opt_params:
        if checkpoint_state is None:
            state, bytes_used = _init_state(params, layers, config)
            server_optimizer.state[state_key] = state
        else:
            state = checkpoint_state
            bytes_used = sum(
                value.numel() * value.element_size()
                for layer_state in state["layers"].values()
                for value in layer_state.values()
                if torch.is_tensor(value))
            print("[{}] restored momentum/transport state from optimizer checkpoint".format(version))
        opt_params[opt_state_key] = state
        print("[{}] {} layers, FP64 state {:.3f} GB, config={}".format(
            version, len(layers), bytes_used / 1024 ** 3, config))
    state = opt_params[opt_state_key]
    # Keep the in-process reference and optimizer checkpoint reference unified.
    server_optimizer.state[state_key] = state

    grad_A = {base: torch.zeros_like(params[names[0]], dtype=STATE_DTYPE)
              for base, names in layers.items()}
    grad_B = {base: torch.zeros_like(params[names[1]], dtype=STATE_DTYPE)
              for base, names in layers.items()}

    def _accumulate(client_id, model_grad):
        for base, (name_A, name_B) in layers.items():
            grad_A[base].add_(model_grad[name_A].to(STATE_DTYPE))
            grad_B[base].add_(model_grad[name_B].to(STATE_DTYPE))

    client_num = collect_client_grads(
        model, loss_name, criterion, train_graphs, device, train_loaders,
        client_lr, opt_params, model_params, server_epoch, _accumulate,
        exclude_from_copy=(opt_state_key,))
    for base in layers:
        grad_A[base].div_(client_num)
        grad_B[base].div_(client_num)

    eta = server_optimizer.param_groups[0]["lr"]
    if any(group["lr"] != eta for group in server_optimizer.param_groups):
        raise ValueError("{} expects one server learning rate".format(version))
    beta = float(opt_params["server_momentum"])
    factor_multiplier = float(opt_params["muonlora_merge_alpha"])
    radius = float(opt_params["lora_init_scale"])
    if config.retract_updated_factors and radius <= 0:
        raise ValueError(
            "{} retraction needs --lora_init_scale > 0".format(version))
    # The alternation schedule is meaningless once both factors move every
    # round, and demanding a valid switch interval for it would be a trap.
    active = None if config.update_both_factors else _current_factor(
        server_epoch, int(opt_params["muonlora_switch_interval"]))

    server_optimizer.zero_grad()
    step_norm, factor_step_norm, storage_residual = muonlora_v21_step(
        params, layers, scalings, state, grad_A, grad_B,
        eta, beta, factor_multiplier, radius, active,
        bool(opt_params.get("muonlora_scaled", False)), config,
        fan_in_fan_out=fan_in_fan_out)
    # Fraction of the intended update the base dtype could not store.  A value
    # approaching 1 means the round is being quantized away rather than applied,
    # which no amount of server-side precision can fix -- eta or the base dtype
    # has to change.
    residual_fraction = storage_residual / step_norm if step_norm > 0 else float("nan")
    print("[{}] epoch={} active={} dir={} eta={:.3e} factor_eta={:.3e} "
          "||Delta W_mu||_F={:.6f} ||Delta(A,B)||_F={:.6f} "
          "storage_residual={:.6f} ({:.1%} of ||Delta W_mu||_F)".format(
              version, server_epoch, "+".join(state["last_updated"]),
              config.factor_direction, eta,
              eta * factor_multiplier, step_norm, factor_step_norm,
              storage_residual, residual_fraction))

    if opt_params.get("train_stats", False):
        train_graphs.grad_norm.append(step_norm)
    if server_lr_scheduler is not None:
        server_lr_scheduler.step()
    for group in server_optimizer.param_groups:
        print("server lr", group["lr"])


# Back-compatible name for the shared wrapper.
federated_muonlora_v21 = run_muonlora_round
