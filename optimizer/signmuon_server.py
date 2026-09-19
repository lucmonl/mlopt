"""Memoryless sign compression on both sides of a server Muon step.

For each trainable tensor, with K participating clients:
    gbar = sum_j sign(g_j) / K
    M = beta * M + gbar
    Q = Muon(M)                     # NS5 by default; exact SVD optional
    D = sign(Q) * ||Q||_F / ||sign(Q)||_F
    W = (1 - lr * weight_decay) * W - lr * D

The default final normalization preserves Q's RMS/Frobenius norm, including
when some entries are zero. It does NOT preserve orthogonality or spectral
norm. 'none' sends unscaled signs instead. Zero input produces a zero step.
Vectors/scalars use sign(M) in place of the matrix Muon operation.

Only server heavy-ball momentum is retained: no client momentum, local steps,
or error feedback. Votes and momentum are kept at the model dtype, matching
ef14muon's server buffers; the step itself is formed in FP32 and rounded once
into the parameter. NS5 uses the existing BF16 Muon kernel, so widening the
buffers would not buy a more accurate LMO -- only accumulation headroom, which
`SignMuonServerState.check_precision` bounds and warns about. These are
simulated sign messages in dense tensors, not a bit-packed implementation.

Use --apply_lora --lora_rank -1 --fedlora_avg signmuon-server to train the same
dense target weights as ef14muon, without constructing any LoRA adapters.
"""

import math

import torch

from optimizer.ef14muon import newton_schulz5
from optimizer.federated_train_single_step import collect_client_grads, select_client_ids


@torch.no_grad()
def normalized_sign(direction, normalization="rms"):
    """Return FP32 signs, optionally scaled to preserve the input Frobenius norm."""
    if normalization not in ("rms", "none"):
        raise ValueError(f"unknown signmuon normalization: {normalization}")
    direction = direction.float()
    signs = direction.sign()
    if normalization == "rms":
        # clamp protects the all-zero case without creating a nonzero step.
        signs.mul_(direction.norm() / signs.norm().clamp_min(1.0))
    return signs


@torch.no_grad()
def muon_direction(momentum, method="ns5"):
    """Muon direction with a zero-preserving, rank-aware SVD reference."""
    if method not in ("ns5", "svd"):
        raise ValueError(f"unknown signmuon LMO: {method}")
    if not torch.isfinite(momentum).all():
        raise FloatingPointError("signmuon-server received nonfinite momentum")
    if momentum.ndim < 2:
        return momentum.sign().float()
    if momentum.ndim != 2:
        raise ValueError("signmuon-server supports matrices, vectors and scalars only")
    if method == "ns5":
        return newton_schulz5(momentum).float()
    # Unlike a bare U @ Vh, zero singular directions must not generate an
    # arbitrary update when client votes cancel or the momentum is low rank.
    U, S, Vh = torch.linalg.svd(momentum.float(), full_matrices=False)
    tolerance = max(momentum.shape) * torch.finfo(S.dtype).eps * S.max()
    return (U * (S > tolerance).to(U.dtype)) @ Vh


class SignMuonServerState:
    def __init__(self, named):
        # Buffers keep the model dtype, as ef14muon's BUF_DTYPE does. The LMO is
        # evaluated in bf16 either way (newton_schulz5 casts its input), so fp32
        # here would only buy accumulation headroom -- and it would give this
        # baseline a more precise server state than the one it is compared
        # against, which is a different experiment.
        #
        # Votes are exact at any of these widths: a sum of at most client_num
        # values in {-1, 0, 1} is an integer, and bf16 represents every integer
        # up to 256. The momentum recursion is not exact; see check_precision.
        self.momentum = {name: torch.zeros_like(p) for name, p in named}
        self.votes = {name: torch.zeros_like(p) for name, p in named}

    @torch.no_grad()
    def check_precision(self, beta, client_count):
        """Warn when the buffer width cannot resolve this beta or client count.

        M converges to roughly gbar/(1 - beta), so an increment gbar is rounded
        away once it falls under half an ulp there, i.e. once 1 - beta drops
        below about the dtype's eps. bf16 (eps 2^-7) is therefore good to about
        beta = 0.99 and silently freezes the momentum near 0.999; fp32 has no
        practical ceiling. The vote sum stays integral while client_count fits
        in the significand, i.e. up to 2/eps -- 256 for bf16, 2048 for fp16.
        """
        for dtype in {t.dtype for t in self.momentum.values()}:
            eps = torch.finfo(dtype).eps
            if 1.0 - beta <= eps:
                print(f"[signmuon-server] WARNING: momentum {beta} is too close to 1 "
                      f"for {dtype}: increments below half an ulp of the steady "
                      f"state are rounded away, so the buffer stalls. Use "
                      f"momentum < {1.0 - eps:.4f} or widen the buffers.")
            if client_count > 2.0 / eps:
                print(f"[signmuon-server] WARNING: {client_count} clients exceeds the "
                      f"{2.0 / eps:.0f} integers {dtype} represents exactly, so the "
                      f"vote sum is no longer exact.")

    @torch.no_grad()
    def start_round(self):
        for votes in self.votes.values():
            votes.zero_()

    @torch.no_grad()
    def accumulate_client(self, client_id, gradients):
        for name, votes in self.votes.items():
            gradient = gradients[name]
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"nonfinite gradient from client {client_id}: {name}")
            # Sign BEFORE averaging: local magnitudes must not affect votes.
            votes.add_(gradient.sign().to(votes))

    @torch.no_grad()
    def apply_step(self, named, param_groups, client_count, beta,
                   normalization="rms", method="ns5"):
        if client_count <= 0:
            raise ValueError("signmuon-server requires at least one participating client")
        if not 0 <= beta < 1:
            raise ValueError("signmuon-server requires momentum in [0, 1)")
        if normalization not in ("rms", "none") or method not in ("ns5", "svd"):
            raise ValueError("invalid signmuon normalization or LMO")
        groups = {id(p): group for group in param_groups for p in group["params"]}
        if any(id(p) not in groups for _, p in named):
            raise ValueError("server optimizer must contain every trainable parameter")
        muon_norm_sq = sent_norm_sq = 0.0
        for name, param in named:
            group = groups[id(param)]
            lr = float(group["lr"])
            weight_decay = float(group.get("weight_decay", 0.0))
            momentum = self.momentum[name]
            momentum.mul_(beta).add_(self.votes[name], alpha=1.0 / client_count)
            direction = muon_direction(momentum, method)
            update = normalized_sign(direction, normalization)
            muon_norm_sq += lr**2 * float(direction.square().sum())
            sent_norm_sq += lr**2 * float(update.square().sum())
            # Form decay and update together in FP32, then round once into the
            # parameter's original dtype. No server_optimizer.step(): that
            # would apply its momentum/preconditioning a second time.
            new_value = param.float().mul(1.0 - lr * weight_decay).add_(update, alpha=-lr)
            param.copy_(new_value)
        return math.sqrt(muon_norm_sq), math.sqrt(sent_norm_sq)


def federated_signmuon_server(model, loss_name, criterion, train_graphs, device,
                              train_loaders, server_optimizer, server_lr_scheduler,
                              client_lr, opt_params, model_params, server_epoch):
    """Collect local signs at the current shared model and take one server step."""
    if opt_params.get("lora_rank") != -1:
        raise ValueError("signmuon-server requires --lora_rank -1 (dense target weights)")
    if opt_params["client_epoch"] != 1:
        raise ValueError("signmuon-server requires --client_epoch 1: one gradient per client")
    beta = opt_params["server_momentum"]
    if not 0 <= beta < 1:
        raise ValueError("signmuon-server requires momentum in [0, 1)")
    normalization = opt_params.get("signmuon_normalization", "rms")
    method = opt_params.get("signmuon_lmo", "ns5")
    if normalization not in ("rms", "none") or method not in ("ns5", "svd"):
        raise ValueError("invalid signmuon normalization or LMO")
    selected = select_client_ids(opt_params)
    if len(selected) == 0:
        raise ValueError("signmuon-server requires at least one participating client")
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named or any(p.ndim > 2 for _, p in named):
        raise ValueError("signmuon-server needs trainable matrices/vectors/scalars")
    opt_params["local_update_ON"] = False
    state_key = "signmuon_server_state"
    if state_key not in opt_params:
        state = SignMuonServerState(named)
        opt_params[state_key] = state
        widths = sorted({str(t.dtype).removeprefix("torch.") for t in state.momentum.values()})
        print(f"[signmuon-server] LMO={method}, normalization={normalization}, "
              f"server buffers at model dtype ({'/'.join(widths)}); "
              "no client state or error feedback")
        state.check_precision(beta, len(selected))
    state = opt_params[state_key]
    state.start_round()
    count = collect_client_grads(
        model, loss_name, criterion, train_graphs, device, train_loaders,
        client_lr, opt_params, model_params, server_epoch, state.accumulate_client,
        exclude_from_copy=(state_key,), client_selected=selected)
    server_optimizer.zero_grad(set_to_none=True)
    muon_norm, update_norm = state.apply_step(
        named, server_optimizer.param_groups, count, beta, normalization, method)
    print(f"[signmuon-server] clients={count}, Muon step norm={muon_norm:.6f}, "
          f"normalized-sign step norm={update_norm:.6f} (before parameter rounding)")
    if opt_params.get("train_stats", False):
        train_graphs.grad_norm.append(muon_norm)
    if server_lr_scheduler is not None:
        server_lr_scheduler.step()
    for group in server_optimizer.param_groups:
        print("server lr", group["lr"])
