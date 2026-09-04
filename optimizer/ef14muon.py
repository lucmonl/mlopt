"""Compressed distributed Muon: layer-wise sparsification on both links.

``federated_ef14muon`` runs one round of either of two variants, selected by
``--fedlora_avg``.  They share everything -- the client loop, the layer-wise
compressors, the LMO, the bit accounting -- and differ only in what they
remember; ``ServerState.error_feedback`` is the switch.

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

ef21muon: the same pipeline with error feedback
-----------------------------------------------
``--fedlora_avg ef21muon`` replaces both memoryless compressions with
Algorithm 1 / Algorithm 3 of "Error Feedback for Muon and Friends"
(Gruntkowska et al., ICLR 2026), taken literally:

    X^{k+1}   = LMO_{B(X^k, t^k)}(G^k)                       server, dense
    S^k       = C^k(X^{k+1} - W^k)                           s2w message
    W^{k+1}   = W^k + S^k                                    model shift
    M_j^{k+1} = (1-beta) M_j^k + beta grad f_j(W^{k+1}; xi)  client momentum
    R_j^{k+1} = C_j^k(M_j^{k+1} - G_j^k)                     w2s message
    G_j^{k+1} = G_j^k + R_j^{k+1}                            client estimator
    G^{k+1}   = G^k + (1/n) sum_j R_j^{k+1}                  server estimator

Two consequences are worth stating, because both change what the surrounding
code sees.  The **model holds W, not X**: clients differentiate the shifted
iterate, which is exactly what the s2w error feedback of EF21-P maintains,
while X lives on the server as a separate dense buffer (with
``--compressor none`` the two coincide and the method reduces to
Gluon/Scion/Muon).  And the **momentum moves to the clients** -- beta =
1 - ``--momentum``, so ``--momentum 0.9`` is the paper's EMA decay 0.9 -- so
there is no server momentum buffer at all.  Per-client state is paged to disk;
see ``ClientStore``.

Layer selection
---------------
This operates on the *dense* weight matrices of whichever modules the LoRA
config would have adapted (``target_modules`` in
``arch/lora.add_adapters_dataset``).  Run it with ``--apply_lora --lora_rank -1``:
that path calls ``add_ft``, which only flips ``requires_grad`` on those modules.
No LoRA factors, no PEFT adapters, no merging are involved.
"""

import math
import os

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


class ClientStore:
    """Per-client (M_j, G_j) kept on disk, one file per client.  EF21 only.

    EF21 needs a persistent gradient estimator G_j *and* a momentum M_j for
    every client, i.e. 2 n d numbers, which does not fit in device memory for
    any interesting (n, d).  Tensors are stored on CPU in `BUF_DTYPE` under
    `<root>/ef21_state/`; `load` moves one client's pair to the device and
    `save` moves it back, so exactly one pair is ever resident.  A client that
    has never been seen returns None, which the caller reads as "initialize me
    this round".

    Sizing this is not optional at scale: n clients cost n * 2 * d * 2 bytes on
    disk and *twice* that in traffic every round (one read and one write per
    participating client).  For 64 clients over the 705M attention parameters of
    Llama-3.2-3B that is 168 GB resident and 336 GB of I/O per round -- past any
    home-directory quota, and slower than the 64 backward passes it serves.
    Set `--ef21_state_dir` to a scratch/project filesystem, cut `--client_num`,
    or drop the momentum (see `store_moment`).

    With beta = 1 (`--momentum 0.0`) the momentum is just the current gradient,
    M_j^{k+1} = grad f_j(W^{k+1}), so there is nothing to carry between rounds:
    `store_moment` is False and the footprint halves.
    """

    def __init__(self, root, shapes, device, store_moment=True, dtype=BUF_DTYPE):
        self.root = root
        self.shapes = shapes
        self.device = device
        self.store_moment = store_moment
        self.dtype = dtype
        os.makedirs(self.root, exist_ok=True)
        self.seen = set()

    def path(self, client_id):
        return os.path.join(self.root, "client_{}.pt".format(int(client_id)))

    def load(self, client_id):
        if int(client_id) not in self.seen:
            return None
        blob = torch.load(self.path(client_id), map_location="cpu", weights_only=True)
        moment = ({n: t.to(self.device, self.dtype) for n, t in blob["m"].items()}
                  if self.store_moment else None)
        return (moment,
                {n: t.to(self.device, self.dtype) for n, t in blob["g"].items()})

    def save(self, client_id, moment, estimator):
        blob = {"g": {n: t.detach().to("cpu", self.dtype) for n, t in estimator.items()}}
        if self.store_moment:
            blob["m"] = {n: t.detach().to("cpu", self.dtype) for n, t in moment.items()}
        torch.save(blob, self.path(client_id))
        self.seen.add(int(client_id))

    def bytes_per_client(self):
        elem = torch.empty(0, dtype=self.dtype).element_size()
        return elem * (2 if self.store_moment else 1) * sum(
            int(torch.tensor(s).prod()) for s in self.shapes.values())

    def disk_bytes(self):
        return self.bytes_per_client() * max(len(self.seen), 1)


class ServerState:
    """Server buffers for both variants.

    EF14 (`error_feedback=False`): a round accumulator and per-layer momentum,
    O(d), no per-client state.

    EF21 (`error_feedback=True`): the true iterate X and the estimator G, plus
    per-client (M_j, G_j) paged through a `ClientStore`.  W is deliberately not
    stored -- it *is* the model, so the client backward pass differentiates the
    shifted iterate, which is what the s2w error feedback of EF21-P maintains.
    Peak resident state stays O(d): X, G, the model, and one loaded pair.
    """

    def __init__(self, named, compressor, ratio, momentum, device,
                 error_feedback=False, s2w_compressor=None, store_root=None,
                 dtype=BUF_DTYPE):
        self.compressor = compressor
        self.s2w_compressor = compressor if s2w_compressor is None else s2w_compressor
        self.ratio = ratio
        self.momentum = momentum
        self.error_feedback = error_feedback
        self.shapes = {n: tuple(p.shape) for n, p in named}
        self.dtype = dtype
        self.uplink_bits = 0
        # both the "dense" reference and the sparse payloads are bf16 values
        self.dense_bits = sum(int(torch.tensor(s).prod()) * VALUE_BITS
                              for s in self.shapes.values())
        if error_feedback:
            # beta in (0, 1] of Algorithm 1; --momentum 0.9 is its EMA decay
            self.beta = 1.0 - momentum
            assert 0 < self.beta <= 1, "ef21muon needs --momentum in [0, 1)"
            # X^0 = W^0 = the current model weights
            self.iterate = {n: p.detach().to(dtype).clone() for n, p in named}
            self.estimator = {n: torch.zeros(s, device=device, dtype=dtype)
                              for n, s in self.shapes.items()}
            # beta == 1 makes M_j^{k+1} = grad f_j(W^{k+1}): nothing to carry
            self.store = ClientStore(store_root, self.shapes, device,
                                     store_moment=self.beta < 1, dtype=dtype)
            self.initialized = False
        else:
            self.round_sum = {n: torch.zeros(s, device=device, dtype=dtype)
                              for n, s in self.shapes.items()}
            self.moment = {n: torch.zeros(s, device=device, dtype=dtype)
                           for n, s in self.shapes.items()}

    def state_bytes(self):
        """Resident device state: (round_sum, m) for EF14, (X, G, M_j, G_j) for EF21."""
        elem = torch.empty(0, dtype=self.dtype).element_size()
        d = sum(int(torch.tensor(s).prod()) for s in self.shapes.values())
        return elem * (4 if self.error_feedback else 2) * d

    def start_round(self):
        # EF21's G is *persistent* -- it is the whole point of the recursion
        # G^{k+1} = G^k + mean_j R_j and must not be cleared between rounds.
        if not self.error_feedback:
            for t in self.round_sum.values():
                t.zero_()

    @torch.no_grad()
    def accumulate_client(self, client_id, model_grad, client_num):
        """One client's contribution to the server's LMO input.

        EF14: compress the raw gradient tensor by tensor and add it in --
        memoryless, nothing is retried.

        EF21 (lines 9-11 of Algorithm 1): read (M_j, G_j) from disk, take the
        momentum step, compress the *residual* M_j - G_j, and write them back.
        R_j / n goes straight into the server estimator -- R_j depends only on
        this client's own state, so no round accumulator is needed.  The divisor
        is the *total* client count, not the number selected this round: G is by
        definition the average of all n estimators G_j, and a client that does
        not participate leaves its G_j, hence its share, unchanged.
        """
        bits = 0
        if not self.error_feedback:
            for name, buf in self.round_sum.items():
                g = model_grad[name].to(self.dtype)
                c, nnz = compress(g, self.compressor, self.ratio)
                buf += c
                bits += compressed_bits(nnz, g.numel(), value_bits=VALUE_BITS)
            self.uplink_bits = bits   # identical for every client
            return

        loaded = self.store.load(client_id)
        if loaded is None:
            # first sight of this client: M_j^0 = G_j^0 = grad f_j(X^0; xi^0),
            # the initialization Theorem 5 assumes, transmitted in full
            moment = {n: model_grad[n].detach().to(self.dtype).clone()
                      for n in self.shapes}
            estimator = {n: t.clone() for n, t in moment.items()}
            for name, t in estimator.items():
                self.estimator[name] += t / client_num
                bits += compressed_bits(t.numel(), t.numel(), value_bits=VALUE_BITS)
        else:
            moment, estimator = loaded
            for name in self.shapes:
                g = model_grad[name].to(self.dtype)
                if moment is None:   # beta == 1: M_j is the gradient itself
                    moment_i = g
                else:
                    moment_i = moment[name].mul_(1 - self.beta).add_(g, alpha=self.beta)
                residual, nnz = compress(moment_i - estimator[name],
                                         self.compressor, self.ratio)
                estimator[name] += residual
                self.estimator[name] += residual / client_num
                bits += compressed_bits(nnz, residual.numel(), value_bits=VALUE_BITS)
        self.store.save(client_id, moment, estimator)
        del moment, estimator
        self.uplink_bits = bits   # identical for every client

    @torch.no_grad()
    def extract(self, client_num):
        """The per-layer LMO input G^k.

        EF14 folds the round into the server momentum m = rho m + ghat.  EF21
        has no server momentum -- the momentum is client-side, inside M_j -- so
        the estimator is already G^k.
        """
        if self.error_feedback:
            return self.estimator
        for name, m in self.moment.items():
            m.mul_(self.momentum).add_(self.round_sum[name] / client_num)
        return self.moment

    @torch.no_grad()
    def apply_step(self, named, updates, radius):
        """LMO step, s2w compression, and the write to the model.

        EF14: the LMO output is dense by construction (orthogonalization spreads
        the update evenly across singular directions), so it is sparsified per
        tensor before it goes out on the wire, and the weights receive exactly
        the sparsified step.  This compression is memoryless on purpose:
        whatever is dropped here is gone, there is no buffer to retry it.

        EF21 (lines 3-6): the step lands on the server's own iterate,
        X^{k+1} = X^k + t LMO(G^k), and what goes on the wire is the *shift*
        S^k = C(X^{k+1} - W^k), applied to the model as W^{k+1} = W^k + S^k.
        The undelivered part stays in X - W and is retried next round.

        Returns (||LMO step||, ||message||, s2w bits, empty tensor count).
        """
        step_norm = 0.0
        sent_norm = 0.0
        bits = 0
        empty = 0
        for name, param in named:
            chunk = updates[name]
            if chunk.count_nonzero() == 0:
                empty += 1
                continue
            # LMO_{B(0,1)}(G) = -NS5(G) / -sign(G), hence the subtraction
            step = radius * lmo_step(chunk, "svd")
            # norms are printed diagnostics only, so they are reduced in fp32
            step_norm += step.float().norm().item() ** 2
            if self.error_feedback:
                self.iterate[name] -= step.to(self.dtype)
                message, nnz = compress(
                    self.iterate[name] - param.detach().to(self.dtype),
                    self.s2w_compressor, self.ratio)
                sent_norm += message.float().norm().item() ** 2
                param.data += message.to(param.dtype)
            else:
                message, nnz = compress(step, self.s2w_compressor, self.ratio)
                sent_norm += message.float().norm().item() ** 2
                param.data -= message.to(param.dtype)
            bits += compressed_bits(nnz, message.numel(), value_bits=VALUE_BITS)
        return step_norm ** 0.5, sent_norm ** 0.5, bits, empty


def federated_ef14muon(model, loss_name, criterion, train_graphs, device, train_loaders,
                       server_optimizer, server_lr_scheduler, client_lr, opt_params,
                       model_params, server_epoch):
    """One round of EF14-Muon, or of EF21-Muon when fedlora_avg is "ef21muon"."""
    from utilities import get_gpu_memory

    error_feedback = opt_params.get("fedlora_avg") == "ef21muon"
    tag = "ef21muon" if error_feedback else "ef14muon"
    state_key = "ef21_state" if error_feedback else "ef14_state"

    # clients differentiate the shared model (W under EF21) and never step
    opt_params["local_update_ON"] = False

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    d = sum(p.numel() for _, p in named)

    compressor = opt_params.get("compressor", "topk")
    ratio = opt_params["sketch_size"]
    if ratio is None or ratio <= 0:
        compressor = "none"
    # C^k in the paper is a free choice; Theorems 4 and 6 need C^k = I, and the
    # reported experiments run with it, so the s2w side can be turned off alone.
    s2w_compressor = ("none" if error_feedback and opt_params.get("ef21_s2w") == "none"
                      else compressor)

    if state_key not in opt_params:
        store_root = None
        if error_feedback:
            # --ef21_state_dir first: the client store is far larger than the
            # checkpoints it sits next to and usually belongs on scratch
            root = opt_params.get("ef21_state_dir") or opt_params.get("checkpoint_dir")
            if root is None:
                import tempfile
                root = tempfile.mkdtemp(prefix="ef21muon_")
                print("[{}] no checkpoint_dir in opt_params; paging client "
                      "state to {}".format(tag, root))
            store_root = os.path.join(root, "ef21_state")
        state = ServerState(named, compressor, ratio, opt_params["server_momentum"],
                            device, error_feedback=error_feedback,
                            s2w_compressor=s2w_compressor, store_root=store_root)
        opt_params[state_key] = state
        opt_params[tag + "_bits"] = 0
        if compressor == "none":
            print("[{}] no compression: dense distributed Muon baseline".format(tag))
        else:
            ks = [resolve_k(int(torch.tensor(s).prod()), ratio)
                  for s in state.shapes.values()]
            print("[{}] layer-wise {}: ratio={} -> k per tensor in [{}, {}] | "
                  "s2w compressor: {}".format(tag, compressor, ratio, min(ks), max(ks),
                                              s2w_compressor))
        print("[{}] {} trainable tensors, {} coordinates, resident server state "
              "{:.2f} GB".format(tag, len(named), d, state.state_bytes() / 1024 ** 3))
        if error_feedback:
            # size this now, not after the first round has half-filled the disk
            per = state.store.bytes_per_client()
            n = opt_params["client_num"]
            print("[{}] beta={:.3f}, client store: {:.2f} GB x {} clients = "
                  "{:.1f} GB at {}, {:.1f} GB of I/O per round".format(
                      tag, state.beta, per / 1024 ** 3, n, n * per / 1024 ** 3,
                      store_root, 2 * n * per / 1024 ** 3))
        else:
            print("[{}] per-client state 0 GB".format(tag))
    state = opt_params[state_key]

    server_lr = 0
    for group in server_optimizer.param_groups:
        server_lr = group["lr"]

    # ---- client loop (shared with the FedAvg path) -------------------------
    # One call = one round.  Algorithm 1 iterates [LMO step] -> [client work];
    # this is the same loop rotated by one, which lets round 0 double as the
    # initialization M_j^0 = G_j^0 = grad f_j(X^0; xi^0) at X^0 = W^0.
    state.start_round()
    total_clients = opt_params["client_num"]
    client_num = collect_client_grads(
        model, loss_name, criterion, train_graphs, device, train_loaders,
        client_lr, opt_params, model_params, server_epoch,
        lambda cid, g: state.accumulate_client(cid, g, total_clients),
        exclude_from_copy=(state_key,))
    get_gpu_memory()

    if error_feedback and not state.initialized:
        state.initialized = True
        print("[{}] initialized M_j = G_j = grad f_j(X^0) for {} client(s); "
              "client state on disk: {:.2f} GB".format(
                  tag, len(state.store.seen), state.store.disk_bytes() / 1024 ** 3))
    opt_params[tag + "_bits"] += state.uplink_bits
    print("[{}] w2s bits/client this round: {} ({:.4f} x dense) | {}/{} clients "
          "participating".format(tag, state.uplink_bits,
                                 state.uplink_bits / state.dense_bits,
                                 client_num, total_clients))

    # ---- server: LMO input, then the step and the s2w message ---------------
    updates = state.extract(client_num)
    step_norm, sent_norm, s2w_bits, empty = state.apply_step(named, updates, server_lr)

    opt_params[tag + "_s2w_bits"] = opt_params.get(tag + "_s2w_bits", 0) + s2w_bits
    print("[{}] s2w bits this round: {} ({:.4f} x dense)".format(
        tag, s2w_bits, s2w_bits / state.dense_bits))
    # under EF21 this ratio is not a retention rate: W lags X by the accumulated
    # primal error, so the message carries that backlog as well as this step.
    print("[{}] LMO step norm: {:.6f} -> sent {:.6f} (ratio {:.3f}) | "
          "{} empty tensors".format(
              tag, step_norm, sent_norm,
              sent_norm / step_norm if step_norm else 0.0, empty))

    if opt_params.get("train_stats", False):
        train_graphs.grad_norm.append(step_norm)

    if server_lr_scheduler is not None:
        server_lr_scheduler.step()
    for group in server_optimizer.param_groups:
        print("server lr", group["lr"])
