"""Riemannion: Muon on the fixed-rank matrix manifold, distributed.

Implements Algorithm 5 of "LoRA meets Riemannion: Muon optimizer for
parametrization-independent low-rank adapters" (Bogachev et al.), minus the LOI
initialization (Algorithm 3), in a federated setting.

Per adapted layer the state is a point on M_r = {X : rank(X) = r}, held as

    dW = A_L @ B.T,      A_L in R^{m x r} with ORTHONORMAL columns,
                         B   in R^{n x r},

together with a heavy-ball tangent vector stored in factored form (A_HB, B_HB)
of width 2r.  The base weight stores W' = W_pretrained - dW^(0) (line 3), so the
effective weight is always W' + dW.

Why this distributes exactly
----------------------------
Each iteration needs only two products with the Euclidean gradient,

    Adot = grad_W L @ B_R          B_R = qr(B).Q
    Bdot = grad_W L.T @ A_L

and both are *linear* in the gradient, so

    (1/n) sum_j (grad f_j @ B_R) = ((1/n) sum_j grad f_j) @ B_R.

Averaging the two thin factors is therefore identical to averaging dense
gradients and projecting afterwards -- the distributed method is not an
approximation of the single-node one, it is the same algorithm.  Each client
transmits (m+n)*r floats per layer, i.e. one LoRA adapter's worth.

How the two products are obtained without touching autograd
-----------------------------------------------------------
Eq. (18) of the paper (the single backward-pass trick) is literally a LoRA
forward: Z1 @ N.T + M @ Z2.T = [Z1 M] @ [N Z2].T.  So the server adapter is
allocated at width 2r and laid out as

    lora_B = [  0  | A_L ]   (m x 2r)
    lora_A = [ B_R | B   ].T (2r x n)

whose forward contribution is 0 @ B_R.T + A_L @ B.T = dW, while an ordinary
backward yields

    grad_lora_B[:, :r] = grad_W L @ B_R   = Adot
    grad_lora_A[r:, :] = A_L.T @ grad_W L = Bdot.T

No custom autograd, no dense gradient, and -- importantly -- no r x r inverse.
(A plain rank-r adapter would force Adot = grad_lora_B @ inv(R) where B = B_R R,
whose conditioning is that of dW's singular values; the 2r layout avoids it.)

This requires the adapter scaling to be exactly 1 (so lora_alpha = 2r) and the
LoRA dropout to be 0; both are enforced at setup time.
"""

import torch

from optimizer.federated_train_single_step import collect_client_grads


# The manifold state and every matrix product run in bf16, matching the model
# dtype, so nothing is silently promoted.  Exactly three things are not bf16,
# and each is forced rather than chosen:
#
# * QR and SVD.  There is no bf16 kernel for either, on CPU or CUDA
#   ("geqrf_cpu"/"linalg_svd_cpu" not implemented for 'BFloat16'), the same
#   constraint that makes `svd_orth` upcast in optimizer/ef14muon.py.  The
#   helpers below take the fp32 detour and hand back bf16.
# * The sum over clients.  Adding n gradients into one accumulator is the
#   textbook bf16 failure -- with an 8-bit mantissa, terms below 2^-9 of the
#   running total vanish -- and is exactly what federated_train guards against
#   by accumulating in fp32.  The two thin factors cost (m+n)r per layer, so
#   fp32 there is ~18 MB for the whole model; see `_accumulate`.
# * Printed diagnostics.  Scalar reductions over millions of entries stay in
#   fp32, and the step norm differences two nearly equal matrices, which
#   cancels catastrophically at bf16 precision.
BUF_DTYPE = torch.bfloat16


def _qr_q(X):
    """Q of a thin QR.  fp32 internally (no bf16 geqrf), result in X's dtype."""
    return torch.linalg.qr(X.float(), mode='reduced')[0].to(X.dtype)


# ---------------------------------------------------------------------------
# Manifold primitives (Algorithms 1 and 2 of the paper, plus the retraction).
#
# Tangent vectors at X = A_L @ B_R.T are parametrized as in Eq. (5):
#     xi = Adot @ B_R.T + A_L @ Bdot.T,     A_L.T @ Adot = 0,
# i.e. by the pair (Adot, Bdot) with left blocks [Adot, A_L] paired against
# right blocks [B_R, Bdot].  The pairing order matters and is kept consistent
# throughout.
# ---------------------------------------------------------------------------

@torch.no_grad()
def ortho_lr(A_L, B_R, Adot, Bdot):
    """Algorithm 1 (OrthoLR): Ortho_r(xi) for a tangent vector xi, returned in
    factored form (left, right) with left @ right.T having all 2r singular
    values equal to 1.

    Two typos in the printed algorithm are corrected here: the QR is taken of
    the n x 2r matrix (not its transpose), and the left/right blocks are paired
    as (Adot <-> B_R, A_L <-> Bdot) to match Eq. (5).

    All of QR, SVD and the products between them run in fp32 -- neither
    factorization has a bf16 kernel -- and the result comes back in the input
    dtype.  The concatenations are bf16 and cost nothing to promote.
    """
    dtype = Adot.dtype
    left = torch.cat([Adot, A_L], dim=1).float()    # (m, 2r)
    right = torch.cat([B_R, Bdot], dim=1).float()   # (n, 2r)

    Q_L, T_L = torch.linalg.qr(left, mode='reduced')    # (m,2r), (2r,2r)
    Q_R, T_R = torch.linalg.qr(right, mode='reduced')   # (n,2r), (2r,2r)

    core = T_L @ T_R.T                                  # (2r, 2r)
    U_c, _, Vh_c = torch.linalg.svd(core)               # discard the singular values
    return (Q_L @ U_c).to(dtype), (Q_R @ Vh_c.T).to(dtype)   # (m,2r), (n,2r)


@torch.no_grad()
def project_lr(P, Q, A_L, B_R):
    """Algorithm 2 (ProjectLR): tangent-space projection of the rank-k matrix
    Z = P @ Q.T onto T_X M_r, returned as (Adot, Bdot).

    From Eq. (6),  P_T(Z) = A_L A_L.T Z + (I - A_L A_L.T) Z B_R B_R.T, hence
        Adot = (I - A_L A_L.T) Z B_R = (P - A_L (A_L.T P)) (Q.T B_R)
        Bdot = Z.T A_L               = Q (P.T A_L)
    (the printed algorithm writes A_R where it means A_L).
    """
    Adot = (P - A_L @ (A_L.T @ P)) @ (Q.T @ B_R)   # (m, r)
    Bdot = Q @ (P.T @ A_L)                          # (n, r)
    return Adot, Bdot


@torch.no_grad()
def retraction_lr(left, right, rank):
    """Rank-`rank` truncated SVD of left @ right.T, computed through the factors
    in O((m+n)r^2 + r^3).  Returns (U, S, V) with U orthonormal (m x r),
    S (r,) and V (n x r), so that trunc_SVD(left @ right.T) = U diag(S) V.T.

    QR and SVD run in fp32 (neither has a bf16 kernel); the result is returned
    in the input dtype, so S is already rounded when the caller folds it into V.
    """
    Q_L, T_L = torch.linalg.qr(left.float(), mode='reduced')
    Q_R, T_R = torch.linalg.qr(right.float(), mode='reduced')
    U_c, S, Vh_c = torch.linalg.svd(T_L @ T_R.T)
    U = Q_L @ U_c[:, :rank]
    V = Q_R @ Vh_c.T[:, :rank]
    return U.to(left.dtype), S[:rank].to(left.dtype), V.to(left.dtype)


# ---------------------------------------------------------------------------
# Adapter <-> manifold bookkeeping
# ---------------------------------------------------------------------------

def riemann_layers(model, server_name):
    """Map base-layer name -> (lora_A param name, lora_B param name) for the
    server adapter."""
    layers = {}
    for name, param in model.named_parameters():
        if "lora_A.{}".format(server_name) in name:
            base = name.replace("lora_A.{}.weight".format(server_name), "base_layer")
            layers.setdefault(base, {})["A"] = name
        elif "lora_B.{}".format(server_name) in name:
            base = name.replace("lora_B.{}.weight".format(server_name), "base_layer")
            layers.setdefault(base, {})["B"] = name
    out = {}
    for base, d in layers.items():
        assert "A" in d and "B" in d, "incomplete adapter for {}".format(base)
        out[base] = (d["A"], d["B"])
    return out


@torch.no_grad()
def write_point(params, name_A, name_B, A_L, B, B_R, rank):
    """Store the manifold point in the 2r adapter layout:
        lora_B = [ 0 | A_L ],   lora_A = [ B_R | B ].T
    so that the forward contributes exactly dW = A_L @ B.T (scaling is 1)."""
    pB, pA = params[name_B], params[name_A]
    pB.data[:, :rank].zero_()
    pB.data[:, rank:] = A_L.to(pB.dtype)
    pA.data[:rank, :] = B_R.T.to(pA.dtype)
    pA.data[rank:, :] = B.T.to(pA.dtype)


@torch.no_grad()
def init_point(m, n, rank, alpha, device, dtype=BUF_DTYPE, generator=None):
    """Manifold init without LOI: A_L a random orthonormal frame, B a small
    random frame scaled by alpha.  The paper reports that a small ||dW^(0)||
    helps; its LOI default is alpha = 0.01/sqrt(r).

    B is deliberately not left at zero: dW = 0 is not on M_r and qr(0) would
    give an arbitrary tangent frame.

    Drawn and orthonormalized in fp32 (no bf16 geqrf), then rounded once: alpha
    is applied before the cast so that a small radius does not lose bits twice.
    """
    A_L = torch.linalg.qr(torch.randn(m, rank, device=device, dtype=torch.float32,
                                      generator=generator), mode='reduced')[0]
    B = torch.linalg.qr(torch.randn(n, rank, device=device, dtype=torch.float32,
                                    generator=generator), mode='reduced')[0] * alpha
    return A_L.to(dtype), B.to(dtype)


# ---------------------------------------------------------------------------
# Locally Optimal Initialization (Section 5 + Algorithms 3 and 5 lines 1-3)
# ---------------------------------------------------------------------------
#
# LOI picks the starting point X^(0) in M_r whose tangent space is best aligned
# with the Euclidean gradient at the pretrained weights (problem (16)).  By
# Theorem 5.1 the solution used in practice is
#
#     dW^(0) = alpha * U_{1,r} V_{r,2r}^T,
#
# where grad_W L(W) = [U_{1,r} U_{r,2r} U_perp] Sigma [V_{1,r} V_{r,2r} V_perp]^T.
# Note the deliberate mismatch -- the *first* r left singular vectors paired with
# the *second* block of r right ones.  That is what makes the whole rank-2r
# truncation land inside the tangent space: writing
#
#     truncSVD(grad, 2r) = U_{1,r} Sigma_{1,r} V_{1,r}^T + U_{r,2r} Sigma_{r,2r} V_{r,2r}^T,
#
# the first term is A_L Bdot^T and the second is Adot B_R^T with Adot^T A_L = 0,
# i.e. exactly the tangent parametrization (5).  So the very first Riemannian
# gradient equals the best rank-2r approximation of the full gradient.
#
# Getting the 2r-truncated SVD of grad_W L(W) without ever forming it is
# Algorithm 3 (BackPropRSVD): a randomized SVD whose only interaction with the
# loss is through the products grad @ N and grad^T @ M, each of which the
# single-backward trick (18) delivers for free.  Two notes on the printed
# algorithm, in the spirit of the corrections already made to Algorithm 1:
#
#   * line 4 prints qr([grad_B L]^T).Q, but grad_B L is already (n, k); the
#     transpose there would make the QR (k, n) and break the next line.  The
#     transpose belongs to line 6 only, where it forms the small Q^T grad.
#   * `Y` names two different things: the (m, k) range basis from lines 2/5 and
#     the (k, n) small matrix from line 6.  Line 8's "Y U" means the former.
#
# The probes need N, M with k = 2r + p columns, so the probe adapter is 2k wide,
# wider than the 2r the optimizer trains with.  It is therefore built here and
# deleted before the first optimizer step; see arch/lora.add_riemann_probe_adapter.


@torch.no_grad()
def _probe_write(params, probe_layers, k, N=None, M=None):
    """Load the probe adapter as lora_B = [0 | M], lora_A = [N^T ; 0].

    The forward contribution is 0 @ N^T + M @ 0 = 0, so the loss is evaluated at
    the pretrained W exactly, while one ordinary backward yields

        grad_lora_B[:, :k] = grad_W L @ N        (m, k)
        grad_lora_A[k:, :] = M^T grad_W L        (k, n)

    which is (18) with Z1 = Z2 = 0.
    """
    for base, (name_A, name_B) in probe_layers.items():
        pB, pA = params[name_B], params[name_A]
        pB.data.zero_()
        pA.data.zero_()
        if M is not None:
            pB.data[:, k:] = M[base].to(pB.dtype)
        if N is not None:
            pA.data[:k, :] = N[base].T.to(pA.dtype)


def _probe(model, loss_name, criterion, train_graphs, device, train_loaders, client_lr,
           opt_params, model_params, server_epoch, params, probe_layers, k,
           N=None, M=None):
    """One federated round of the probe: returns (grad @ N, grad^T @ M) averaged
    over the participating clients, as dicts keyed by base layer.

    Averaging in fp32 for the same reason as `_accumulate`: this is a sum over
    client_num bf16 gradients.
    """
    _probe_write(params, probe_layers, k, N=N, M=M)
    outN = {base: 0 for base in probe_layers} if N is not None else None
    outM = {base: 0 for base in probe_layers} if M is not None else None

    def _accumulate(client_id, model_grad):
        for base, (name_A, name_B) in probe_layers.items():
            if outN is not None:
                outN[base] = outN[base] + model_grad[name_B][:, :k].float()
            if outM is not None:
                outM[base] = outM[base] + model_grad[name_A][k:, :].T.float()

    client_num = collect_client_grads(
        model, loss_name, criterion, train_graphs, device, train_loaders,
        client_lr, opt_params, model_params, server_epoch, _accumulate)
    if outN is not None:
        outN = {b: v / client_num for b, v in outN.items()}
    if outM is not None:
        outM = {b: v / client_num for b, v in outM.items()}
    return outN, outM


@torch.no_grad()
def tangent_residual(U, S, V, A_L, B_R):
    """Relative ||Z - P_T(Z)||_F for Z = U diag(S) V^T, the quantity Theorem 5.1
    drives to zero.  Computed through the factors, never forming Z:

        Z - P_T(Z) = (I - A_L A_L^T) Z (I - B_R B_R^T) = L R^T,

    and ||L R^T||_F = ||T_L T_R^T||_F via the two thin QRs.
    """
    L = (U * S.unsqueeze(0)).float()
    L = L - A_L.float() @ (A_L.float().T @ L)
    R = V.float() - B_R.float() @ (B_R.float().T @ V.float())
    T_L = torch.linalg.qr(L, mode='reduced')[1]
    T_R = torch.linalg.qr(R, mode='reduced')[1]
    denom = (U * S.unsqueeze(0)).float().norm().item()
    return (T_L @ T_R.T).norm().item() / max(denom, 1e-30)


def loi_init(model, loss_name, criterion, train_graphs, device, train_loaders, client_lr,
             opt_params, model_params, server_epoch, rank, alpha,
             oversample=16, power_steps=1, probe_name="loi_probe"):
    """Algorithm 3 + Algorithm 5 lines 1-2.  Returns {base: (A_L, B)}.

    Costs 2(power_steps + 1) federated rounds -- 4 with the paper's q = 1 --
    and is run exactly once, before the first optimizer step.
    """
    from arch.lora import add_riemann_probe_adapter, drop_riemann_probe_adapter

    server_name = opt_params["server_name"]
    k = 2 * rank + oversample          # Algorithm 3 line 1, called with 2r
    print("[riemannion] LOI: sketch width k = 2r + p = {}, probe adapter width "
          "{}, {} power step(s) -> {} probe rounds".format(
              k, 2 * k, power_steps, 2 * (power_steps + 1)))

    grad_state = {n: p.requires_grad for n, p in model.named_parameters()}
    add_riemann_probe_adapter(model, 2 * k, probe_name, server_name)
    model.set_adapter(probe_name)      # only the probe contributes and trains
    for n, p in model.named_parameters():
        p.requires_grad = (".{}.".format(probe_name) in n)

    params = dict(model.named_parameters())
    probe_layers = riemann_layers(model, probe_name)
    run = lambda N, M: _probe(model, loss_name, criterion, train_graphs, device,
                              train_loaders, client_lr, opt_params, model_params,
                              server_epoch, params, probe_layers, k, N=N, M=M)
    try:
        # line 1: Omega ~ N(0, 1), one per layer, shaped (n, k)
        omega = {}
        for base, (name_A, name_B) in probe_layers.items():
            pA = params[name_A]
            omega[base] = torch.randn(pA.shape[1], k, device=pA.device,
                                      dtype=torch.float32)
        # line 2: Q = qr(grad @ Omega).Q, an orthonormal basis for range(grad)
        gN, _ = run(omega, None)
        Q = {b: _qr_q(v) for b, v in gN.items()}
        # lines 3-5: q power steps, alternating grad^T and grad
        for _ in range(power_steps):
            _, gM = run(None, Q)
            Qt = {b: _qr_q(v) for b, v in gM.items()}
            gN, _ = run(Qt, None)
            Q = {b: _qr_q(v) for b, v in gN.items()}
        # line 6: the small (k, n) matrix Q^T grad
        _, gM = run(None, Q)
    finally:
        drop_riemann_probe_adapter(model, probe_name, server_name)
        for n, p in model.named_parameters():
            if n in grad_state:
                p.requires_grad = grad_state[n]

    # lines 7-8 and Algorithm 5 line 2
    point = {}
    residuals, gaps = [], []
    for base in probe_layers:
        small = gM[base].T                                  # (k, n) = Q^T grad
        U_s, S, Vh = torch.linalg.svd(small, full_matrices=False)
        U = Q[base] @ U_s[:, :2 * rank]                     # (m, 2r)
        V = Vh[:2 * rank].T                                 # (n, 2r)
        A_L = U[:, :rank]                                   # U_{1,r}
        B = alpha * V[:, rank:2 * rank]                     # alpha * V_{r,2r}
        # B_R = qr(B).Q = V_{r,2r} already: alpha only scales an orthonormal frame
        residuals.append(tangent_residual(U, S[:2 * rank], V, A_L, V[:, rank:2 * rank]))
        if S.numel() > 2 * rank:
            # Theorem 5.1 needs sigma_2r != sigma_{2r+1}; a tight gap makes the
            # U_{1,r} / U_{r,2r} split of a *randomized* SVD arbitrary
            gaps.append((S[2 * rank - 1] / max(S[2 * rank].item(), 1e-30)).item())
        point[base] = (A_L.to(BUF_DTYPE), B.to(BUF_DTYPE))

    print("[riemannion] LOI: alpha={:.6g}, tangent residual "
          "||Z-P_T(Z)||/||Z|| max {:.2e} mean {:.2e} (0 = Theorem 5.1 exact)".format(
              alpha, max(residuals), sum(residuals) / len(residuals)))
    if gaps:
        print("[riemannion] LOI: sigma_2r / sigma_2r+1 min {:.3f} mean {:.3f}"
              " (needs > 1; near 1 makes the r / 2r split arbitrary)".format(
                  min(gaps), sum(gaps) / len(gaps)))
    return point


# ---------------------------------------------------------------------------
# Federated Riemannion
# ---------------------------------------------------------------------------

def federated_riemannion(model, loss_name, criterion, train_graphs, device, train_loaders,
                         server_optimizer, server_lr_scheduler, client_lr, opt_params,
                         model_params, server_epoch):
    from utilities import get_gpu_memory

    rank = opt_params["riemann_rank"]
    server_name = opt_params["server_name"]
    beta = opt_params["server_momentum"]          # heavy-ball coefficient (Eq. 10)
    gamma = opt_params["riemann_gamma"]           # weight decay in line 12
    retract_mode = opt_params.get("riemann_retract", "literal")

    # clients differentiate the shared point and never take a local step
    opt_params["local_update_ON"] = False

    params = dict(model.named_parameters())
    layers = riemann_layers(model, server_name)

    # ---- one-time setup: put every layer on the manifold and set W' --------
    if "riemann_state" not in opt_params:
        init_mode = opt_params.get("riemann_init", "random")
        loi_point = None
        if init_mode == "loi":
            # Algorithm 5 lines 1-2. Must run before write_point: the probes
            # need the adapter to contribute nothing, which is true only while
            # the server lora_B is still PEFT's zero initialization.
            loi_point = loi_init(
                model, loss_name, criterion, train_graphs, device, train_loaders,
                client_lr, opt_params, model_params, server_epoch, rank,
                opt_params["riemann_alpha"],
                oversample=opt_params.get("riemann_loi_oversample", 16),
                power_steps=opt_params.get("riemann_loi_power", 1))
            params = dict(model.named_parameters())   # probe adapter is gone
        state = {}
        total = 0
        for base, (name_A, name_B) in layers.items():
            pB, pA = params[name_B], params[name_A]
            m, two_r = pB.shape
            _, n = pA.shape
            assert two_r == 2 * rank, (
                "adapter width {} != 2*riemann_rank {} for {}".format(two_r, 2 * rank, base))
            if loi_point is not None:
                A_L, B = loi_point[base]
            else:
                A_L, B = init_point(m, n, rank, opt_params["riemann_alpha"],
                                    device=pB.device, dtype=BUF_DTYPE)
            B_R = _qr_q(B)
            state[base] = {"A_L": A_L, "B": B,
                           "A_HB": torch.zeros(m, 2 * rank, device=pB.device,
                                               dtype=BUF_DTYPE),
                           "B_HB": torch.zeros(n, 2 * rank, device=pB.device,
                                               dtype=BUF_DTYPE)}
            write_point(params, name_A, name_B, A_L, B, B_R, rank)
            # line 3: W' = W - dW^(0), so the initial forward is unchanged --
            # exactly in real arithmetic, and up to bf16 rounding here: dW^(0)
            # is ~1e-4 of ||W||, so most of this subtraction is below the
            # mantissa and the effective start is W + O(alpha*sqrt(r)).
            base_w = params.get(base + ".weight")
            if base_w is None:
                raise KeyError("no base weight for {}".format(base))
            base_w.data -= (A_L @ B.T).to(base_w.dtype)
            # A_L (m,r) + B (n,r) + A_HB (m,2r) + B_HB (n,2r) = 3r(m+n)
            total += 3 * rank * (m + n)
        opt_params["riemann_state"] = state
        elem = torch.empty(0, dtype=BUF_DTYPE).element_size()
        print("[riemannion] {} layers, manifold rank {}, adapter width {}, "
              "server state {:.3f} GB ({}), per-client state 0 GB".format(
                  len(layers), rank, 2 * rank, total * elem / 1024 ** 3,
                  str(BUF_DTYPE).replace("torch.", "")))
        print("[riemannion] init={} beta={} gamma={} retraction={}".format(
            init_mode, beta, gamma, retract_mode))
    state = opt_params["riemann_state"]

    server_lr = 0
    for group in server_optimizer.param_groups:
        server_lr = group["lr"]
    eta = server_lr

    # ---- client loop: every client differentiates the same point -----------
    # averaged tangent-space gradient factors, per layer
    grad_A = {base: 0 for base in layers}   # -> grad_W L @ B_R      (m, r)
    grad_B = {base: 0 for base in layers}   # -> grad_W L.T @ A_L    (n, r)

    def _accumulate(client_id, model_grad):
        # uplink: only the two halves that carry information, (m+n)*r floats.
        # grad_lora_B[:, r:] and grad_lora_A[:r, :] are never used (the latter
        # is identically zero because the Z1 block of lora_B is zero).
        #
        # fp32 here and only here: this is a sum of client_num bf16 gradients,
        # the one reduction long enough for an 8-bit mantissa to matter. The
        # accumulator is two thin factors, (m+n)r per layer, so the whole model
        # costs ~18 MB more than the bf16 version. It is rounded back to bf16
        # after the division by client_num below.
        for base, (name_A, name_B) in layers.items():
            grad_A[base] = grad_A[base] + model_grad[name_B][:, :rank]
            grad_B[base] = grad_B[base] + model_grad[name_A][rank:, :].T

    client_num = collect_client_grads(
        model, loss_name, criterion, train_graphs, device, train_loaders,
        client_lr, opt_params, model_params, server_epoch, _accumulate,
        exclude_from_copy=("riemann_state",))
    get_gpu_memory()

    # ---- server: one Riemannion step per layer (Algorithm 5, lines 6-14) ---
    point_norm = 0.0
    step_norm = 0.0
    for base, (name_A, name_B) in layers.items():
        st = state[base]
        A_L, B = st["A_L"], st["B"]

        B_R = _qr_q(B)                                                  # line 6

        # back to bf16 once the fp32 client sum has been averaged
        Adot = (grad_A[base] / client_num).to(BUF_DTYPE)                # line 7
        Bdot = (grad_B[base] / client_num).to(BUF_DTYPE)                # line 8

        # line 9: vector transport of the stored momentum to T_X M_r
        Adot_prev, Bdot_prev = project_lr(st["A_HB"], st["B_HB"], A_L, B_R)

        # line 10: heavy ball in the tangent space
        Adot = beta * Adot_prev + (Adot - A_L @ (A_L.T @ Adot))
        Bdot = beta * Bdot_prev + Bdot

        # line 11: Ortho_r then project back (Eq. 12)
        oleft, oright = ortho_lr(A_L, B_R, Adot, Bdot)
        Adot, Bdot = project_lr(oleft, oright, A_L, B_R)

        # line 12: retraction.
        # As printed, the right factor is [B_R, -eta*(Bdot + gamma*B_R)], which
        # does not carry the current point B forward -- so the new point is the
        # retraction of the step alone and ||dW|| stays ~eta every round.  That
        # is what "literal" reproduces.  "accumulate" is the reading consistent
        # with Algorithm 6 line 14 (dW - eta*(Mtilde + gamma*dW)).
        left = torch.cat([-eta * Adot, A_L], dim=1)
        if retract_mode == "literal":
            right = torch.cat([B_R, -eta * (Bdot + gamma * B_R)], dim=1)
        elif retract_mode == "accumulate":
            right = torch.cat([B_R, B - eta * (Bdot + gamma * B)], dim=1)
        elif retract_mode == "gemini":
            right = torch.cat([B_R, -eta * (Bdot + gamma * B)], dim=1)
        else:
            raise NotImplementedError("riemann_retract: {}".format(retract_mode))
        U, S, V = retraction_lr(left, right, rank)

        # line 13: keep Mtilde as the momentum for the next round
        st["A_HB"] = torch.cat([Adot, A_L], dim=1)
        st["B_HB"] = torch.cat([B_R, Bdot], dim=1)

        # line 14: the new point
        A_L_new = U.to(BUF_DTYPE)
        B_new = (V * S.unsqueeze(0)).to(BUF_DTYPE)
        # fp32 for the diagnostic: the two products are nearly equal (the step
        # is small next to the point), so the difference cancels away most of a
        # bf16 mantissa. This is also the only O(mn) work in the file.
        step_norm += ((A_L_new.float() @ B_new.float().T)
                      - (A_L.float() @ B.float().T)).norm().item() ** 2
        st["A_L"], st["B"] = A_L_new, B_new

        B_R_new = _qr_q(B_new)
        write_point(params, name_A, name_B, A_L_new, B_new, B_R_new, rank)
        # = ||dW||_F since A_L is orthonormal; fp32 like every printed reduction
        point_norm += B_new.float().norm().item() ** 2

    print("[riemannion] eta={:.3e} ||dW||_F={:.6f} ||step||_F={:.6f}".format(
        eta, point_norm ** 0.5, step_norm ** 0.5))

    if opt_params.get("train_stats", False):
        train_graphs.grad_norm.append(step_norm ** 0.5)

    if server_lr_scheduler is not None:
        server_lr_scheduler.step()
    for group in server_optimizer.param_groups:
        print("server lr", group["lr"])
