"""EF14-Muon: FetchSGD-style server-side error feedback with a Muon LMO step.

Motivation
----------
EF21-Muon (Gruntkowska et al., "Error Feedback for Muon and Friends", ICLR 2026)
compresses the worker-to-server direction with per-worker error feedback: each
worker j stores a momentum buffer M_j and a gradient estimator G_j and sends
C(M_j - G_j).  That is provably convergent, but the state is 2*n*d floats, which
is infeasible to *simulate* on one machine for a large model and many clients.

This module implements the FetchSGD arrangement instead (Rothchild et al.,
arXiv:2007.07682): momentum and error accumulation are moved to the *server* and
kept in sketch space, so there is **no per-client state at all**.  The reason
this is sound is linearity of the Count Sketch,

    sum_j S(g_j) = S(sum_j g_j),

so aggregating per-client sketches loses nothing; the lossy step is unsketching,
which happens on the server, which is exactly where the error buffer lives.
(This is why the uplink message must be a sketch.  A per-client Top-K or Rand-K
would destroy information *at the client*, where no residual is stored, and a
server-side buffer could not compensate for it.)

The only change relative to FetchSGD is the update rule: instead of applying the
recovered update directly, we feed it through Muon's linear minimization oracle
over the spectral-norm ball, i.e. Newton-Schulz orthogonalization, per weight
matrix.

    server, each round k:
        Sk    = (1/n) sum_j S(g_j)                  # exact sketch of the mean
        m     = rho * m + Sk                        # momentum,  sketch space
        e     = e + t * m                           # error accumulation
        D     = unsketch(e), keep k coordinates     # --compressor
        e, m  = mask out the support of D           # momentum factor masking
        W_i  -= t * sqrt(max(1, rows/cols)) * NS5(D_i)   for each matrix i

Note this is a heuristic: it borrows FetchSGD's mechanism and Muon's step, so the
convergence guarantees of the EF21-Muon paper do not carry over to it.

Layer selection
---------------
This operates on the *dense* weight matrices of whichever modules the LoRA config
would have adapted (``target_modules`` in ``arch/lora.add_adapters_dataset``).
Run it with ``--apply_lora --lora_rank -1``: that path calls ``add_ft``, which
only flips ``requires_grad`` on those modules.  No LoRA factors, no PEFT
adapters, no merging are involved.
"""

import copy
import math
import time

import numpy as np
import torch

from optimizer.load_optimizer import load_optimizer

# Quintic Newton-Schulz coefficients from Jordan et al.'s Muon.
NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5

# Count Sketch geometry, matching optimizer/fetchsgd.py.
SKETCH_ROWS = 5
SKETCH_BLOCKS = 20


@torch.no_grad()
def newton_schulz5(G, steps=NS_STEPS, eps=1e-7):
    """Approximate the spectral-norm LMO: return an (approximately) orthogonal
    matrix sharing G's singular vectors, i.e. U V^T for G = U S V^T.

    This is the standard Muon quintic iteration: it converges to the orthogonal
    polar factor when run on G / ||G||_F.  Computed in bfloat16 like the
    reference implementation; the iteration is scale invariant, so the magnitude
    of G is irrelevant to the result.
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
def lmo_step(chunk):
    """Unit-radius LMO direction for one parameter.

    Matrices use the spectral-norm ball (Muon / Scion hidden layers); anything
    that is not 2-D uses the l_inf ball, whose LMO is sign(.), as the paper does
    for embedding and head layers.
    """
    if chunk.ndim == 2:
        rows, cols = chunk.shape
        scale = math.sqrt(max(1.0, rows / cols))
        return newton_schulz5(chunk.float()).float() * scale
    return torch.sign(chunk.float())


def resolve_sketch_size(sketch_size, d):
    """Interpret --sketch_size.

    A value in (0, 1) is a fraction of the d trainable coordinates (so 0.15
    means 15%, matching the Top15% naming in the EF21-Muon paper); a value >= 1
    is an absolute count, which is how every pre-existing caller passes it;
    anything <= 0 (e.g. the -1 default) means no compression.
    """
    if sketch_size is None or sketch_size <= 0:
        return -1
    if sketch_size < 1:
        return max(1, int(round(float(sketch_size) * d)))
    return min(int(sketch_size), d)


class ServerErrorFeedback:
    """Server-side momentum + error accumulation in sketch space.

    Holds no per-client state.  ``accumulate_client`` is called once per client
    per round with that client's gradient; ``extract`` closes the round and
    returns the sparse update to be fed to the LMO.
    """

    def __init__(self, d, k, compressor, momentum, device, dtype=torch.float32):
        from csvec import CSVec

        self.d = d
        self.k = k
        self.compressor = compressor
        self.momentum = momentum
        self.device = device
        self.dtype = dtype

        self.sketch = CSVec(d=d, c=k, r=SKETCH_ROWS,
                            device=device, numBlocks=SKETCH_BLOCKS)
        self.round_table = torch.zeros(SKETCH_ROWS, k, device=device, dtype=dtype)
        self.moment = torch.zeros(SKETCH_ROWS, k, device=device, dtype=dtype)
        self.error = torch.zeros(SKETCH_ROWS, k, device=device, dtype=dtype)
        self.bits_per_round = SKETCH_ROWS * k * 32

    def state_bytes(self):
        return 4 * (self.round_table.numel() + self.moment.numel() + self.error.numel())

    def start_round(self):
        self.round_table.zero_()

    def accumulate_client(self, g):
        """Sketch one client's gradient and add it in.  Linear, so the result
        after all clients equals the sketch of the summed gradient."""
        self.sketch.zero()
        self.sketch.accumulateVec(g.to(self.dtype))
        self.round_table += self.sketch.table

    def extract(self, client_num, lr):
        """Momentum, error accumulation, and heavy-hitter recovery."""
        self.moment.mul_(self.momentum).add_(self.round_table / client_num)
        self.error.add_(self.moment, alpha=lr)

        self.sketch.zero()
        self.sketch.accumulateTable(self.error)
        if self.compressor == "topk":
            update = self.sketch.unSketch(k=self.k)
        elif self.compressor == "randk":
            vals = self.sketch._findAllValues()
            idx = torch.randperm(self.d, device=vals.device)[:self.k]
            update = torch.zeros_like(vals)
            update[idx] = vals[idx]
        else:
            raise NotImplementedError("unknown compressor: {}".format(self.compressor))

        # FetchSGD's momentum factor masking: drop from both buffers whatever
        # was just handed to the optimizer, so it is not applied twice.
        self.sketch.zero()
        self.sketch.accumulateVec(update)
        nz = self.sketch.table.nonzero()
        self.error[nz[:, 0], nz[:, 1]] = 0
        self.moment[nz[:, 0], nz[:, 1]] = 0
        return update


class DenseServerState:
    """Uncompressed baseline (``--compressor none``): plain distributed Muon.

    No sketching and no error feedback -- the server just averages the client
    gradients and applies momentum, so this is the reference point the
    compressed runs are measured against.
    """

    def __init__(self, d, momentum, device, dtype=torch.float32):
        self.d = d
        self.momentum = momentum
        self.round_sum = torch.zeros(d, device=device, dtype=dtype)
        self.moment = torch.zeros(d, device=device, dtype=dtype)
        self.bits_per_round = d * 32

    def state_bytes(self):
        return 4 * (self.round_sum.numel() + self.moment.numel())

    def start_round(self):
        self.round_sum.zero_()

    def accumulate_client(self, g):
        self.round_sum += g.to(self.round_sum.dtype)

    def extract(self, client_num, lr):
        self.moment.mul_(self.momentum).add_(self.round_sum / client_num)
        return self.moment


def _trainable(model):
    """Ordered list of (name, param) over trainable tensors.  With
    --lora_rank -1 these are the dense weights of the LoRA target modules."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _flatten(model_grad, names):
    return torch.cat([model_grad[n].reshape(-1).float() for n in names])


def federated_ef14muon(model, loss_name, criterion, train_graphs, device, train_loaders,
                       server_optimizer, server_lr_scheduler, client_lr, opt_params,
                       model_params, server_epoch):
    from main import train
    from utilities import get_gpu_memory

    # Clients differentiate the shared server model and never take a local step,
    # so model_grad is a clean stochastic gradient at the current iterate.
    opt_params["local_update_ON"] = False

    client_num = opt_params["client_num"]
    client_opt_name = opt_params["client_opt_name"]
    client_epoch = opt_params["client_epoch"]

    if opt_params["client_partial"] < 1:
        client_num = int(opt_params["client_partial"] * client_num)
        client_selected = np.random.choice(opt_params["client_num"], client_num, replace=False)
    else:
        client_selected = np.arange(client_num)

    named = _trainable(model)
    names = [n for n, _ in named]
    d = sum(p.numel() for _, p in named)

    compressor = opt_params.get("compressor", "topk")
    k = resolve_sketch_size(opt_params["sketch_size"], d)
    if compressor == "none" or k <= 0:
        compressor = "none"

    # ---- lazily build the server state (no per-client state exists) ---------
    if "ef14_state" not in opt_params:
        rho = opt_params["server_momentum"]
        if compressor == "none":
            state = DenseServerState(d, rho, device)
            print("[ef14muon] no compression: dense distributed Muon baseline")
        else:
            state = ServerErrorFeedback(d, k, compressor, rho, device)
            print("[ef14muon] sketch: d={} k={} ({:.2f}% of d) rows={} blocks={}".format(
                d, k, 100.0 * k / d, SKETCH_ROWS, SKETCH_BLOCKS))
        print("[ef14muon] {} trainable tensors, {} coordinates, "
              "server state {:.2f} GB, per-client state 0 GB".format(
                  len(named), d, state.state_bytes() / 1024 ** 3))
        opt_params["ef14_state"] = state
        opt_params["ef14_bits"] = 0
    state = opt_params["ef14_state"]

    server_lr = 0
    for group in server_optimizer.param_groups:
        server_lr = group["lr"]

    # ---- client loop: every client differentiates the same model -----------
    # deepcopy without the server state: the sketch tables are large and the
    # CSVec hash tables are expensive to copy.
    opt_params.pop("ef14_state")
    client_opt_params = copy.deepcopy(opt_params)
    opt_params["ef14_state"] = state
    client_opt_params["train_stats"] = False

    state.start_round()
    training_time_accumulated = 0
    for client_id in client_selected:
        client_model = model  # alias, no deepcopy
        client_model.train()
        optimizer, lr_scheduler, _ = load_optimizer(
            client_opt_name, client_model, client_lr, opt_params["client_momentum"],
            opt_params["client_weight_decay"], opt_params["lr_decay"],
            opt_params["epochs_lr_decay"], False, model_params, client_opt_params)

        for epoch in range(client_epoch):
            try:
                train_graphs.loader_iter += 1
                # train_loaders[0] is an iterator in the LLM pipelines (hence the
                # StopIteration handler below) and a plain DataLoader elsewhere;
                # both are accepted here.
                start_time = time.time()
                _, model_grad = train(client_model, loss_name, criterion, device,
                                      train_loaders[0], optimizer, lr_scheduler,
                                      server_epoch, client_opt_params)
                end_time = time.time()
                training_time_accumulated += end_time - start_time
                print(f"Time taken for client {client_id}: {end_time - start_time:.3f}s")
            except StopIteration:
                print("\nData Iterator is reloaded")
                train_graphs.loader_iter += 1
                train_loaders[0] = iter(train_loaders[1])
                _, model_grad = train(client_model, loss_name, criterion, device,
                                      train_loaders[0], optimizer, lr_scheduler,
                                      server_epoch, client_opt_params)

        # uplink message: a sketch of this client's gradient
        state.accumulate_client(_flatten(model_grad, names))
        del model_grad

    print(f"Total training time: {training_time_accumulated:.3f}s")
    get_gpu_memory()

    opt_params["ef14_bits"] += state.bits_per_round
    print("[ef14muon] w2s bits/client this round: {} ({:.4f} x dense)".format(
        state.bits_per_round, state.bits_per_round / (d * 32)))

    # ---- server: momentum, error accumulation, recovery --------------------
    update = state.extract(client_num, server_lr)

    # ---- LMO step, per weight matrix ---------------------------------------
    offset = 0
    update_norm = 0.0
    for name, param in named:
        n = param.numel()
        chunk = update[offset:offset + n].view_as(param)
        offset += n
        direction = lmo_step(chunk)
        param.data -= (server_lr * direction).to(param.dtype)
        update_norm += (server_lr * direction).float().norm().item() ** 2
    assert offset == d, "flatten/unflatten mismatch: {} vs {}".format(offset, d)
    print("[ef14muon] applied update norm: {:.6f}".format(update_norm ** 0.5))

    if opt_params.get("train_stats", False):
        train_graphs.grad_norm.append(update_norm ** 0.5)

    if server_lr_scheduler is not None:
        server_lr_scheduler.step()
    for group in server_optimizer.param_groups:
        print("server lr", group["lr"])
