"""Compressed distributed Muon: layer-wise sparsification on both links.

Uplink
------
Each client sparsifies its own gradient **per tensor** (layer-wise top-k or
rand-k, `--compressor` / `--sketch_size`) and sends only the surviving entries.
Every layer therefore appears in every message with its own budget
k_i = ratio * numel_i, rather than competing for one global budget.

Server
------
One dense per-layer buffer -- momentum `m`, O(d) -- and **no per-client state**:

    ghat = (1/n) sum_j C(grad f_j)      # compressed client messages   [w2s]
    m    = rho * m + ghat               # momentum
    Delta_i = lr * NS5(m_i)             # Muon LMO, per matrix
    W_i -= C(Delta_i)                   # sparsified broadcast         [s2w]

There is **no error feedback anywhere**: both compressions are memoryless, so
whatever either of them discards is discarded permanently.  Two consequences,
stated plainly because neither is compensated:

* `ghat` is a biased estimate of the mean gradient -- the regime Beznosikov
  et al. (2020, Example 1) show can diverge for biased compression without EF;
* the applied step is a sparsified, hence no longer orthogonal, version of the
  LMO direction. Because `NS5` deliberately spreads mass evenly across entries,
  top-k on its output is close to random-k and removes most of the step's
  magnitude; the printed `retained` ratio measures exactly this.

(An earlier version kept a server-side error buffer `e` with `D = C(e)`,
`e -= D`. That recursion is self-consistent -- `D` is a subset of `e`, so drain
is contained in fill -- but it was correcting nothing, since `NS5` and the s2w
compression both sit between `D` and the weights. Note the *other* recursion,
`e -= C(NS5(e))`, is not self-consistent at all: `NS5` is magnitude-destroying,
so fill is gradient-scaled while drain is dimension-scaled and the buffer
diverges or oscillates.)

Layer selection
---------------
This operates on the *dense* weight matrices of whichever modules the LoRA
config would have adapted (``target_modules`` in
``arch/lora.add_adapters_dataset``).  Run it with ``--apply_lora --lora_rank -1``:
that path calls ``add_ft``, which only flips ``requires_grad`` on those modules.
No LoRA factors, no PEFT adapters, no merging are involved.
"""

import math

import torch

from optimizer.compressor import compress, compressed_bits, resolve_k
from optimizer.federated_train_single_step import collect_client_grads

# Quintic Newton-Schulz coefficients from Jordan et al.'s Muon.
NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5

# Everything -- client gradients, server buffers, compression, the LMO -- runs
# in bf16, matching the model dtype, so nothing is silently promoted. The one
# exception is `svd_orth`: torch.linalg.svd has no bf16 kernel on either CPU or
# CUDA, so it upcasts internally and casts the result back. Scalar reductions
# used only for printed diagnostics stay in fp32; summing squares over millions
# of entries in bf16 would report noise.
BUF_DTYPE = torch.bfloat16
VALUE_BITS = 16   # bf16 payload on the wire, must track BUF_DTYPE


@torch.no_grad()
def newton_schulz5(G, steps=NS_STEPS, eps=1e-7):
    """Approximate the spectral-norm LMO: return an (approximately) orthogonal
    matrix sharing G's singular vectors, i.e. U V^T for G = U S V^T.

    The standard Muon quintic iteration, computed in bfloat16 like the reference
    implementation.  It is scale invariant, so the magnitude of G is irrelevant
    to the result.
    """
    assert G.ndim == 2, "spectral LMO needs a matrix, got shape {}".format(tuple(G.shape))
    a, b, c = NS_COEFFS
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        # keep the iteration on the smaller Gram matrix
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


@torch.no_grad()
def svd_orth(G):
    """Exact spectral-norm LMO: return U V^T for the thin SVD G = U S V^T.

    Same target as `newton_schulz5`, without the polynomial approximation -- use
    it to check how far the quintic iteration actually is from the true
    orthogonal factor. Costs a full SVD per call, so it is a diagnostic, not a
    drop-in for the training loop.

    This is the one place the bf16 pipeline is broken, and not by choice:
    torch.linalg.svd has no bfloat16 kernel on CPU ("linalg_svd_cpu" not
    implemented for 'BFloat16') or on CUDA, so the factorization runs in fp32
    and the result is cast back to G's dtype.

    Like the NS iteration this is scale invariant and ignores S entirely, so a
    rank-deficient G still comes back with min(m, n) orthonormal directions --
    the ones belonging to zero singular values are arbitrary but harmless, since
    the LMO only ever multiplies them by zero downstream.
    """
    assert G.ndim == 2, "spectral LMO needs a matrix, got shape {}".format(tuple(G.shape))
    X = G.float()
    try:
        U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    except Exception:
        # cuSOLVER gesvd occasionally fails to converge; the CPU path does not
        U, _, Vh = torch.linalg.svd(X.cpu(), full_matrices=False)
        U, Vh = U.to(X.device), Vh.to(X.device)
    return (U @ Vh).to(G.dtype)


@torch.no_grad()
def lmo_step(chunk, method="ns5"):
    """Unit-radius LMO direction for one parameter.

    Matrices use the spectral-norm ball (Muon / Scion hidden layers); anything
    that is not 2-D uses the l_inf ball, whose LMO is sign(.), as the paper does
    for embedding and head layers.

    `method` picks how the spectral LMO is evaluated: "ns5" is the quintic
    Newton-Schulz approximation used for training, "svd" the exact U V^T.

    The result keeps the input dtype (bf16 in the pipeline); `svd_orth` upcasts
    internally only because torch.linalg.svd has no bf16 kernel.
    """
    if chunk.ndim == 2:
        scale = 1
        if method == "svd":
            return svd_orth(chunk) * scale
        return newton_schulz5(chunk).to(chunk.dtype) * scale
    return torch.sign(chunk)


class ServerState:
    """Per-layer momentum.  No error buffer and no per-client state."""

    def __init__(self, shapes, compressor, ratio, momentum, device,
                 dtype=BUF_DTYPE):
        self.compressor = compressor
        self.ratio = ratio
        self.momentum = momentum
        self.shapes = shapes
        self.dtype = dtype
        self.round_sum = {n: torch.zeros(s, device=device, dtype=dtype)
                          for n, s in shapes.items()}
        self.moment = {n: torch.zeros(s, device=device, dtype=dtype)
                       for n, s in shapes.items()}
        self.uplink_bits = 0
        # both the "dense" reference and the sparse payloads are bf16 values
        self.dense_bits = sum(int(torch.tensor(s).prod()) * VALUE_BITS
                              for s in shapes.values())

    def state_bytes(self):
        elem = torch.empty(0, dtype=self.dtype).element_size()
        return elem * 2 * sum(int(torch.tensor(s).prod()) for s in self.shapes.values())

    def start_round(self):
        for t in self.round_sum.values():
            t.zero_()

    @torch.no_grad()
    def accumulate_client(self, model_grad):
        """Compress this client's gradient tensor by tensor and add it in."""
        bits = 0
        for name, buf in self.round_sum.items():
            g = model_grad[name].to(self.dtype)
            c, nnz = compress(g, self.compressor, self.ratio)
            buf += c
            bits += compressed_bits(nnz, g.numel(), value_bits=VALUE_BITS)
        self.uplink_bits = bits   # identical for every client

    @torch.no_grad()
    def extract(self, client_num, lr):
        """Momentum only.  Returns the per-layer LMO input, scaled by lr."""
        updates = {}
        for name, m in self.moment.items():
            m.mul_(self.momentum).add_(self.round_sum[name] / client_num)
            updates[name] = lr * m
        return updates


def federated_ef14muon(model, loss_name, criterion, train_graphs, device, train_loaders,
                       server_optimizer, server_lr_scheduler, client_lr, opt_params,
                       model_params, server_epoch):
    from utilities import get_gpu_memory

    # clients differentiate the shared model and never take a local step
    opt_params["local_update_ON"] = False

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    shapes = {n: tuple(p.shape) for n, p in named}
    d = sum(p.numel() for _, p in named)

    compressor = opt_params.get("compressor", "topk")
    ratio = opt_params["sketch_size"]
    if ratio is None or ratio <= 0:
        compressor = "none"

    if "ef14_state" not in opt_params:
        state = ServerState(shapes, compressor, ratio,
                            opt_params["server_momentum"], device)
        opt_params["ef14_state"] = state
        opt_params["ef14_bits"] = 0
        if compressor == "none":
            print("[ef14muon] no compression: dense distributed Muon baseline")
        else:
            ks = [resolve_k(int(torch.tensor(s).prod()), ratio) for s in shapes.values()]
            print("[ef14muon] layer-wise {}: ratio={} -> k per tensor in [{}, {}]".format(
                compressor, ratio, min(ks), max(ks)))
        print("[ef14muon] {} trainable tensors, {} coordinates, "
              "server state {:.2f} GB, per-client state 0 GB".format(
                  len(named), d, state.state_bytes() / 1024 ** 3))
    state = opt_params["ef14_state"]

    server_lr = 0
    for group in server_optimizer.param_groups:
        server_lr = group["lr"]

    # ---- client loop (shared with the FedAvg path) -------------------------
    state.start_round()
    client_num = collect_client_grads(
        model, loss_name, criterion, train_graphs, device, train_loaders,
        client_lr, opt_params, model_params, server_epoch,
        lambda cid, g: state.accumulate_client(g),
        exclude_from_copy=("ef14_state",))
    get_gpu_memory()

    opt_params["ef14_bits"] += state.uplink_bits
    print("[ef14muon] w2s bits/client this round: {} ({:.4f} x dense)".format(
        state.uplink_bits, state.uplink_bits / state.dense_bits))

    # ---- server: momentum ---------------------------------------------------
    updates = state.extract(client_num, server_lr)

    # ---- LMO step, per weight matrix ---------------------------------------
    # The LMO output is dense by construction (orthogonalization spreads the
    # update evenly across singular directions), so it is sparsified per tensor
    # before it goes out on the wire. This compression is memoryless on purpose:
    # whatever is dropped here is gone, there is no buffer to retry it.
    update_norm = 0.0
    sent_norm = 0.0
    s2w_bits = 0
    empty = 0
    for name, param in named:
        chunk = updates[name]
        if chunk.count_nonzero() == 0:
            empty += 1
            continue
        step = server_lr * lmo_step(chunk, "svd")
        # norms are printed diagnostics only, so they are reduced in fp32
        update_norm += step.float().norm().item() ** 2
        step, nnz = compress(step, compressor, ratio)
        s2w_bits += compressed_bits(nnz, step.numel(), value_bits=VALUE_BITS)
        sent_norm += step.float().norm().item() ** 2
        param.data -= step.to(param.dtype)

    opt_params["ef14_s2w_bits"] = opt_params.get("ef14_s2w_bits", 0) + s2w_bits
    print("[ef14muon] s2w bits this round: {} ({:.4f} x dense)".format(
        s2w_bits, s2w_bits / state.dense_bits))
    print("[ef14muon] LMO step norm: {:.6f} -> sent {:.6f} (retained {:.3f}) | "
          "{} empty tensors".format(
              update_norm ** 0.5, sent_norm ** 0.5,
              (sent_norm / update_norm) ** 0.5 if update_norm else 0.0, empty))

    if opt_params.get("train_stats", False):
        train_graphs.grad_norm.append(update_norm ** 0.5)

    if server_lr_scheduler is not None:
        server_lr_scheduler.step()
    for group in server_optimizer.param_groups:
        print("server lr", group["lr"])
