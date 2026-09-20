"""Intrinsic Muon for LoRA (arXiv:2605.09238, Appendix F.3).

For A in R^{r x n}, B in R^{m x r}, let A.T = QA RA and B = QB RB.
Without momentum, factor gradients supply the two projected gradients:

    HB = G_B RA^-1,             HA = RB^-T G_A
    D_B = polar(HB) RA^-T,      D_A = RB^-1 polar(HA)
    A+ = A - lr D_A,            B+ = B - lr D_B

Both directions use the old factors. tau is absorbed into lr. There is no
POLoRA curvature state, damping, factor-norm rescaling or direction rescaling.
Full-rank factors are required; --lora_init_scale > 0 supplies them using the
existing base-compensated initialization (the initial effective weight stays
unchanged up to storage precision).

The existing --momentum sets beta: 0 implements F.3 directly; positive values
implement Appendix K (paper: 0.95), with Nesterov factor
buffers followed by the ambient direction Mhat = Gtilde_B A + B Gtilde_A.
Only Mhat QA and QB.T Mhat are evaluated, without forming an m x n tensor.
This is deliberately different from feeding factor momenta directly into F.3.

QR, solves, polar iterations, gradient accumulation and momentum use fp32
(fp64 for fp64 inputs); adapter storage retains its original dtype. PEFT's
positive alpha/r scaling is already in the factor gradients. We optimize
f(A,B)=loss(W_base+s BA), with no extra division of lr by s. Use alpha=rank
for the paper's X=BA convention. The server optimizer only supplies the lr
and scheduler; momentum is applied here using opt_params["server_momentum"].
The server optimizer's own step and weight decay are not applied.
"""

import math

import torch

from optimizer.federated_train_single_step import collect_client_grads
from optimizer.riemannion import riemann_layers as adapter_layers


def _work_dtype(tensor):
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


@torch.no_grad()
def polar(X, method="ns", ns_steps=10):
    """Thin polar factor; zero singular directions remain zero.

    ns uses F.2's cubic Newton--Schulz polynomial, multiplying on the short
    side. svd is the numerical reference, with null singular values dropped.
    """
    if method not in ("ns", "svd"):
        raise ValueError("imuon polar method must be 'ns' or 'svd'")
    if ns_steps < 1:
        raise ValueError("imuon_ns_steps must be positive")
    X = X.to(_work_dtype(X))
    if not bool(torch.isfinite(X).all()):
        raise ValueError("imuon received a non-finite projected gradient")
    # Scaling first avoids overflow/underflow in the Frobenius norm.
    scale = X.abs().max()
    if scale == 0:
        return torch.zeros_like(X)
    X = X / scale
    if method == "svd":
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        keep = S > S[0] * max(X.shape) * torch.finfo(X.dtype).eps
        return (U * keep.to(X.dtype)) @ Vh
    transposed = X.shape[0] > X.shape[1]
    Y = X.T if transposed else X
    Y = Y / torch.linalg.vector_norm(Y)
    for _ in range(ns_steps):
        Y = 1.5 * Y - 0.5 * ((Y @ Y.T) @ Y)
    return Y.T if transposed else Y


def _full_rank_qr(X):
    """Thin QR with positive diagonal and a numerical full-rank check."""
    if not bool(torch.isfinite(X).all()):
        raise ValueError("imuon factors must be finite")
    Q, R = torch.linalg.qr(X, mode="reduced")
    singular = torch.linalg.svdvals(R)
    if singular[-1] <= singular[0] * max(X.shape) * torch.finfo(X.dtype).eps:
        raise ValueError(
            "imuon requires full-rank LoRA factors; use --lora_init_scale > 0 "
            "and reduce lr if factors become singular during training")
    signs = R.diagonal().sign()
    return Q * signs, signs.unsqueeze(1) * R


@torch.no_grad()
def imuon_step(A, B, G_A, G_B, state, lr, beta=0.0,
               polar_method="ns", ns_steps=10):
    """Update one pair in place and return squared factor-step Frobenius norm."""
    if not math.isfinite(lr) or lr < 0:
        raise ValueError("imuon lr must be finite and nonnegative")
    if not 0 <= beta < 1:
        raise ValueError("imuon momentum must be in [0, 1)")
    if (A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]
            or not 0 < A.shape[0] <= min(A.shape[1], B.shape[0])
            or G_A.shape != A.shape or G_B.shape != B.shape):
        raise ValueError("imuon requires compatible thin LoRA factors and gradients")
    dtype = _work_dtype(A)
    # Disable surrounding AMP: QR and its products must use the work dtype.
    with torch.autocast(device_type=A.device.type, enabled=False):
        a, b = A.to(dtype), B.to(dtype)
        ga, gb = G_A.to(dtype), G_B.to(dtype)
        QA, RA = _full_rank_qr(a.T)
        QB, RB = _full_rank_qr(b)
        if beta:
            if "M_A" not in state:
                state["M_A"] = torch.zeros_like(a)
                state["M_B"] = torch.zeros_like(b)
            state["M_A"].mul_(beta).add_(ga)
            state["M_B"].mul_(beta).add_(gb)
            ga = ga + beta * state["M_A"]
            gb = gb + beta * state["M_B"]
            # Project Appendix K's Mhat without allocating the ambient matrix.
            HB = gb @ RA.T + b @ (ga @ QA)
            HA = (QB.T @ gb) @ a + RB @ ga
        else:
            HB = torch.linalg.solve_triangular(RA.T, gb.T, upper=False).T
            HA = torch.linalg.solve_triangular(RB.T, ga, upper=False)
        DA = torch.linalg.solve_triangular(
            RB, polar(HA, polar_method, ns_steps), upper=True)
        DB = torch.linalg.solve_triangular(
            RA, polar(HB, polar_method, ns_steps).T, upper=True).T
        next_A, next_B = a - lr * DA, b - lr * DB
        # Check after casting too, so fp16 overflow cannot corrupt the adapter.
        next_A, next_B = next_A.to(A.dtype), next_B.to(B.dtype)
        if not bool(torch.isfinite(next_A).all() & torch.isfinite(next_B).all()):
            raise ValueError("imuon produced non-finite factors; reduce the learning rate")
        delta_sq = ((next_A.to(dtype) - a).square().sum()
                    + (next_B.to(dtype) - b).square().sum()).item()
        A.copy_(next_A)
        B.copy_(next_B)
    return delta_sq


def federated_imuon(model, loss_name, criterion, train_graphs, device, train_loaders,
                    server_optimizer, server_lr_scheduler, client_lr, opt_params,
                    model_params, server_epoch):
    """Average shared-adapter gradients, then take one intrinsic server step."""
    if opt_params["client_epoch"] != 1:
        raise ValueError("imuon requires --client_epoch 1")
    opt_params["local_update_ON"] = False
    params = dict(model.named_parameters())
    layers = adapter_layers(model, opt_params["server_name"])
    covered = {name for names in layers.values() for name in names}
    if (not layers or any(p.requires_grad and name not in covered
                          for name, p in params.items())
            or any(not params[name].requires_grad for name in covered)):
        raise ValueError("imuon requires only complete, trainable server LoRA pairs")
    state = opt_params.setdefault("imuon_state", {base: {} for base in layers})
    grads = {name: torch.zeros_like(params[name], dtype=_work_dtype(params[name]))
             for name in covered}

    def accumulate(client_id, model_grad):
        for name, buf in grads.items():
            buf.add_(model_grad[name].to(buf.dtype))

    client_num = collect_client_grads(
        model, loss_name, criterion, train_graphs, device, train_loaders,
        client_lr, opt_params, model_params, server_epoch, accumulate,
        exclude_from_copy=("imuon_state",))
    if client_num <= 0:
        raise ValueError("imuon requires at least one participating client")
    # Respect the scheduler's current lr, including separate parameter groups.
    lrs = {id(p): group["lr"] for group in server_optimizer.param_groups
           for p in group["params"]}
    step_sq = 0.0
    for base, (name_A, name_B) in layers.items():
        A, B = params[name_A], params[name_B]
        if id(A) not in lrs or id(B) not in lrs or lrs[id(A)] != lrs[id(B)]:
            raise ValueError("imuon needs the same server lr for both factors of each pair")
        step_sq += imuon_step(
            A, B, grads[name_A] / client_num, grads[name_B] / client_num,
            state[base], lrs[id(A)], opt_params["server_momentum"],
            opt_params["imuon_polar"], opt_params["imuon_ns_steps"])
    print("[imuon] {} layers, ||step||_F={:.6f}".format(len(layers), step_sq ** 0.5))
    if opt_params.get("train_stats", False):
        train_graphs.grad_norm.append(step_sq ** 0.5)
    if server_lr_scheduler is not None:
        server_lr_scheduler.step()
