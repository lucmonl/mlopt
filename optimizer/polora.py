"""PoLoRA: preconditioned orthogonalized LoRA, federated.

Implements Algorithm 1 of "PoLoRA: A Preconditioned Orthogonalized LoRA
Optimizer" (Ghosh, Parshakova and Gower) as a ``--fedlora_avg`` method, with
the supporting subroutines of Appendix E (guarded power iteration, Algorithm 2,
and the Gram Newton-Schulz iteration, Algorithm 3).

The step, per adapted layer and its LoRA pair (A in R^{r x din}, B in R^{dout x r}):

    M_A  <- b1 M_A + (1-b1) G_A               buffer                  (line 1)
    Mh_A <- b1 M_A + (1-b1) G_A               look-ahead
    P <- diag(p/||p||_inf), Q <- diag(q/||q||_inf)                     (line 2)
    C_B <- B^T P B,   C_A <- A Q A^T
    D_A <- C_B^-1/2 msign(C_B^-1/2 Mh_A Q^-1/2) Q^-1/2                 (line 3)
    D_B <- P^-1/2 msign(P^-1/2 Mh_B C_A^-1/2) C_A^-1/2                 (line 4)
    rho <- eta / (||A||_2 + ||B||_2)                                   (line 5)
    A <- A - rho D_A/max(||D_A||_2, eps),  B likewise                  (line 6)
    q <- b2 q + (1-b2) diag(G_A^T C_B^-1 G_A)/r                        (line 7)
    p <- b2 p + (1-b2) diag(G_B C_A^-1 G_B^T)/r                        (line 8)

Why this distributes exactly
----------------------------
Everything the step consumes from the data is the pair of factor gradients
(G_A, G_B) = (B^T G, G A^T), and both are linear in the weight gradient G, so

    (1/n) sum_j (B^T grad f_j) = B^T ((1/n) sum_j grad f_j).

Every client differentiates the *same* server adapter (``local_update_ON`` is
forced off) and uploads the two factor gradients, r(din + dout) floats per
layer -- one adapter's worth, the same uplink as FedAvg on LoRA.  The server
then runs Algorithm 1 once on the average.  This is not an approximation of the
single-node method; it is the same method, with the batch spread over clients.
Consequently ``--client_epoch`` must stay 1 and the client optimizer is used
only to produce a gradient, never to take a step.

LoRA dropout (``--lora_dropout``, 0 by default here to match the paper's Table 3)
does not break this.  PEFT drops the adapter's *input*, so with xh = drop(x) the
two factor gradients of one client are still B^T Gh and Gh A^T for the same
Gh = dL/dout xh^T -- the same mask enters both -- and B and A are the server's,
shared.  The average over clients is therefore B^T mean_j(Gh_j), exactly as
above, with the per-sample dropout masks playing the role they already play in a
single-node run.  What dropout does change is the variance of G_A and G_B, which
lines 7 and 8 read raw, so the curvature estimate is noisier; that is an
argument about regularization, not about correctness of the distribution.

Adapter scaling
---------------
The paper sets alpha = r so the merged weight is W_0 + BA.  Here the adapter
carries PEFT's scaling s = lora_alpha/r, so the merged weight is W_0 + s BA and
the merged update is s(B dA + dB A).  Only line 5 cares: rho is divided by s as
well, which keeps the bound ||dW||_2 <= eta on what the *layer* sees.  Nothing
else changes, because the rest of the step is invariant to the scale of the
factor gradients -- msign is scale invariant, P and Q are normalized to unit
largest entry, and D_A, D_B are divided by their own spectral norms.

The server optimizer supplies only the learning rate eta and its schedule; the
update is applied here, so ``--server_opt_name`` and ``--momentum`` do not
enter the step (the momentum is beta1, ``--polora_beta1``).
"""

import torch

from optimizer.federated_train_single_step import collect_client_grads
# generic adapter bookkeeping: base layer name -> (lora_A name, lora_B name)
from optimizer.riemannion import riemann_layers as adapter_layers


# Dtype policy: everything with a d_in or d_out side -- the factors, the
# momentum buffers, every product against them -- stays in the adapter's own
# dtype (bf16 in the LLM runs).  No d-sized tensor is ever materialized in fp32.
# What is fp32 is only ever r x r, a vector, or a scalar, and each case is
# forced rather than chosen:
#
# * The r x r Newton-Schulz work: both Gram iterations and the inverse square
#   roots.  Appendix E.2 asks for exactly this ("we run the Gram Newton-Schulz
#   iteration in fp32, which is inexpensive because every matrix is only
#   r x r").  The Grams that feed it are accumulated in fp32 as well, but a
#   chunk of rows at a time (`weighted_gram`), so the promotion is bounded by
#   GRAM_CHUNK x r and no d-sided matrix is ever promoted whole.  Leaving that
#   accumulation in bf16 does not merely blur the result, it diverges -- the
#   docstring there has the measurement.  `svd_msign` is the one exception to
#   the no-d-sided-fp32 rule, and only because torch.linalg.svd has no bf16
#   kernel on CPU or CUDA.
# * The sum over clients, the curvature vectors p and q, and scalar reductions
#   remain in bf16 too.  This keeps the optimizer state and all d-sided working
#   buffers in bf16; only the small Gram / Newton--Schulz numerical kernels
#   below are fp32.

# PolarExpress coefficients (Amsel et al., arXiv:2505.16932): the per-iteration
# quintic p_t(s) = a s + b s^3 + c s^5 of Eq. (33), fit so that the composition
# drives every singular value in [1e-3, 1] to 1.  The tail is the classical
# quintic Newton-Schulz map (15s - 10s^3 + 3s^5)/8, which is what the sequence
# converges to; iterations past the table reuse it.
POLAR_EXPRESS = (
    (8.28721201814563, -23.595886519098837, 13.470325693189728),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524836),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
)


# Rows of the reduced dimension promoted at a time in `weighted_gram`.  At
# r = 256 one chunk is a 4 MB fp32 buffer; at the r = 16 this repo usually runs,
# 256 KB.
GRAM_CHUNK = 4096


# ---------------------------------------------------------------------------
# Numerical subroutines (Appendix E)
# ---------------------------------------------------------------------------

@torch.no_grad()
def weighted_gram(M, w=None, chunk=GRAM_CHUNK):
    """M^T diag(w) M, accumulated in fp32 over chunks of M's d rows.

    This is the one place a d-sided quantity has to be promoted, and the reason
    is definiteness rather than precision: the r x r result feeds a
    Newton-Schulz iteration that assumes eigenvalues in (0, 1].  Forming the
    Gram with a bf16 matmul rounds the result by ~4e-3 relative, enough to push
    a small eigenvalue negative, and the quintic then amplifies that eigenvalue
    by about a^K (8.3^8 ~ 1e7) instead of driving it to 1 -- the iterate
    overflows rather than merely blurring.  Measured on the synthetic problem:
    an eigenvalue of -1.7e-4 relative in the line-4 matrix sign takes B to
    non-finite on the second step.  The two curvature Grams are damped by delta
    and survive bf16, but with little margin; the matrix-sign Gram has no
    damping at all.

    Only `chunk` rows are promoted at a time, so the fp32 footprint is
    chunk x r instead of d x r, and M itself is never copied whole.  The result
    is symmetrized, which the callers' iteration assumes.
    """
    d, r = M.shape
    out = torch.zeros(r, r, device=M.device, dtype=torch.float32)
    for i in range(0, d, chunk):
        blk = M[i:i + chunk].float()
        out += blk.T @ blk if w is None else (blk * w[i:i + chunk].unsqueeze(1)).T @ blk
    return (out + out.T) / 2


@torch.no_grad()
def gram_newton_schulz(S, gamma, steps):
    """Algorithm 3: Z ~= (S/gamma)^-1/2 for positive definite S in R^{r x r} and
    gamma >= lambda_max(S), using only r x r matrix multiplications.

    R <- S/gamma has all eigenvalues in (0, 1]; each iteration applies the
    quintic M = a I + b R + c R^2 to both the accumulator (Z <- M Z) and the
    normalized matrix (R <- M R M).  Since every M is a polynomial in R, Z stays
    symmetric and commutes with R, so R_K ~= I forces Z ~= R_0^-1/2.

    Runs in fp32 (Appendix E.2, "Stability and precision").
    """
    print("using gram_newton_schulz msign")
    R = S.float() / gamma
    eye = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
    Z = eye.clone()
    for t in range(steps):
        a, b, c = POLAR_EXPRESS[min(t, len(POLAR_EXPRESS) - 1)]
        M = a * eye + b * R + c * (R @ R)
        Z = M @ Z
        R = M @ R @ M
    return Z


@torch.no_grad()
def svd_msign(X):
    """Exact msign(X) = U V^T from the thin SVD, the reference for the Gram path.

    Same target as the Gram iteration, without the polynomial approximation --
    use it to check how far that iteration actually is from the true polar
    factor.  It costs a full SVD of an r x d matrix per call, against r x r
    matrix multiplications, so it is a diagnostic rather than a drop-in.

    torch.linalg.svd has no bfloat16 kernel on CPU or CUDA, so the
    factorization runs in fp32 and the result is cast back to X's dtype.

    Lemma 2's minimizer is the one of *least Frobenius norm*, i.e. the reduced
    SVD over the nonzero singular values only, so directions below the standard
    numerical-rank cutoff are dropped.  This is not cosmetic here: unlike the
    spectral LMO of Muon, the step does not multiply those directions by zero
    downstream -- D_A is rescaled by its own spectral norm -- so keeping them
    would inject arbitrary orthonormal vectors into the update.  The Gram
    iteration lifts them only partially and is conservative in the same way.
    """
    print("using svd msign")
    X_f = X.float()
    try:
        U, S, Vh = torch.linalg.svd(X_f, full_matrices=False)
    except Exception:
        # cuSOLVER gesvd occasionally fails to converge; the CPU path does not
        U, S, Vh = torch.linalg.svd(X_f.cpu(), full_matrices=False)
        U, S, Vh = U.to(X_f.device), S.to(X_f.device), Vh.to(X_f.device)
    keep = S > S[0] * max(X.shape) * torch.finfo(torch.float32).eps
    return (U[:, keep] @ Vh[keep]).to(X.dtype)


@torch.no_grad()
def msign(X, steps, method="svd"):
    """msign(X) = U V^T for the reduced SVD X = U S V^T.

    `method` picks how the polar factor is evaluated: "gram" is the Gram
    Newton-Schulz iteration used for training, "svd" the exact factorization
    (`svd_msign`).  Note that the curvature inverse square roots of lines 3 and
    4 go through `damped_inv_sqrt` and stay on the Gram iteration either way.

    The Gram path uses msign(X) = (X X^T)^-1/2 X on the short side, so the
    Newton-Schulz work is r x r rather than r x d (Appendix F: a factor d/(2r)
    per iteration).  X is never copied: the products against it run in its own
    dtype and only the r x r Gram and iterate are fp32 (`weighted_gram` explains
    why that one cannot be bf16).  The svd path is fp32 because
    torch.linalg.svd has no bf16 kernel, and unlike the Gram path it does have
    to promote a d-sided matrix.

    X = 0 has no polar factor -- the paper's convention is X/||X|| := 0 -- and
    it does occur, at the first round where B = 0 makes G_A = M_A = 0.
    """
    transposed = X.shape[0] > X.shape[1]
    Y = X.T if transposed else X                       # (short, long)
    # ||Y||_F^2 >= lambda_max(Y Y^T), the bound Algorithm 3 needs.  Accumulated
    # in fp32 without materializing an fp32 copy of Y, which the svd path -- the
    # only one that does not need one anyway -- would otherwise pay for.
    gamma = Y.pow(2).sum(dtype=torch.float32).item()
    if not gamma > 0:
        return torch.zeros_like(X)
    if method == "svd":
        out = svd_msign(Y)
    elif method == "gram":
        # Y Y^T, accumulated in fp32 a chunk at a time; Y itself is never
        # copied.  This Gram carries no damping, so it is the one most exposed
        # to a rounded-negative eigenvalue -- see `weighted_gram`.
        Z = gram_newton_schulz(weighted_gram(Y.T), gamma, steps)
        out = (Z.to(Y.dtype) @ Y) / gamma ** 0.5
    else:
        raise NotImplementedError("polora msign method: {}".format(method))
    return out.T if transposed else out


@torch.no_grad()
def damped_inv_sqrt(C, delta, eps, steps, cache, key, iters):
    """Chat = C + max(delta*lambda_max(C), eps) I, then Chat^-1/2 (Appendix E).

    The damping caps the condition number of Chat at about 1/delta, so a factor
    with tiny singular values -- B at initialization is exactly zero -- cannot
    blow the update up.  gamma = Tr(Chat) >= lambda_max(Chat) is the bound
    Algorithm 3 needs.

    C arrives as the fp32 r x r Gram from `weighted_gram`, so the damping floor
    delta = 1e-4 sits far above its rounding and Chat is positive definite by
    construction -- which is what the Newton-Schulz iteration assumes.
    """
    lam = spectral_norm(C, cache, key, iters)          # C is PSD: ||C||_2 = lambda_max
    Chat = C.float() + max(delta * lam, eps) * torch.eye(
        C.shape[0], device=C.device, dtype=torch.float32)
    gamma = Chat.diagonal().sum().item()
    return gram_newton_schulz(Chat, gamma, steps) / gamma ** 0.5


@torch.no_grad()
def spectral_norm(M, cache, key, iters, eps=1e-12):
    """Algorithm 2: warm-started power iteration for ||M||_2, guarded from below.

    The leading vector is cached across optimizer steps, so it starts near the
    top singular vector.  A cold or stale start underestimates ||M||_2, and the
    optimizer *divides* by this estimate, so the maximum row norm -- a lower
    bound that does not depend on the cached vector -- is returned whenever it
    is larger.  Iterates the smaller of the two Grams.
    """
    X = M if M.shape[0] <= M.shape[1] else M.T
    # Keep power-iteration reductions in bf16 too; no d-sided fp32 copy is
    # materialized.
    lower = X.pow(2).sum(dim=1, dtype=torch.bfloat16).max().sqrt().item()
    v = cache.get(key)
    if (v is None or v.shape[0] != X.shape[0] or v.dtype != X.dtype
            or not bool(torch.isfinite(v).all())
            or v.pow(2).sum(dtype=torch.bfloat16).item() == 0):
        v = X @ torch.ones(X.shape[1], device=X.device, dtype=X.dtype)
    for _ in range(iters):
        Xw = X @ (X.T @ v)
        norm = Xw.pow(2).sum(dtype=torch.bfloat16).sqrt().item()
        if norm <= eps:
            cache.pop(key, None)
            return lower
        v = Xw / norm
    cache[key] = v
    return max((X.T @ v).pow(2).sum(dtype=torch.bfloat16).sqrt().item(), lower)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def adapter_scalings(model, server_name):
    """Per base layer, the PEFT scaling s = lora_alpha/r of the server adapter."""
    out = {}
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        # LlamaAttention also carries a float `scaling`; only LoraLayer.scaling
        # is a {adapter_name: float} dict.
        if isinstance(scaling, dict) and server_name in scaling:
            out[name + ".base_layer"] = float(scaling[server_name])
    return out


@torch.no_grad()
def init_state(params, layers, eps):
    """Algorithm 1 "Initialize": M_A, M_B <- 0 and p, q <- eps 1.

    The normalization in line 2 maps p = q = eps 1 to the identity metric at the
    first step, so the first update is a plain Product Muon step.
    """
    state, total = {}, 0
    for base, (name_A, name_B) in layers.items():
        pA, pB = params[name_A], params[name_B]
        r, din = pA.shape
        dout, r_B = pB.shape
        assert r == r_B, "rank mismatch for {}: A is {}, B is {}".format(
            base, tuple(pA.shape), tuple(pB.shape))
        state[base] = {
            "M_A": torch.zeros(r, din, device=pA.device, dtype=pA.dtype),
            "M_B": torch.zeros(dout, r, device=pB.device, dtype=pB.dtype),
            "q": torch.full((din,), eps, device=pA.device, dtype=torch.bfloat16),
            "p": torch.full((dout,), eps, device=pB.device, dtype=torch.bfloat16),
            "norm_cache": {},
        }
        total += r * (din + dout) * pA.element_size()   # M_A + M_B
        total += (din + dout) * 2                       # p + q, bf16
    return state, total


@torch.no_grad()
def polora_step(A, B, G_A, G_B, st, eta, scaling, beta1, beta2, eps, delta,
                ns_steps, pow_iters, msign_method="gram"):
    """One PoLoRA step (Algorithm 1) for a single LoRA pair, in place on A and B.

    `st` is that pair's state -- momentum buffers M_A, M_B, curvature vectors
    p, q and the power-iteration cache -- and is updated in place too.  Returns
    (rho, ||B||_2/||A||_2, ||dA||_F^2 + ||dB||_F^2) for the round's diagnostics.
    """
    cache = st["norm_cache"]
    r = A.shape[0]

    # line 1: buffer, then the look-ahead read off the updated buffer
    # st["M_A"].mul_(beta1).add_(G_A, alpha=1 - beta1)
    # st["M_B"].mul_(beta1).add_(G_B, alpha=1 - beta1)
    # Mh_A = beta1 * st["M_A"] + (1 - beta1) * G_A
    # Mh_B = beta1 * st["M_B"] + (1 - beta1) * G_B
    st["M_A"].mul_(beta1).add_(G_A)
    st["M_B"].mul_(beta1).add_(G_B)
    Mh_A = st["M_A"] 
    Mh_B = st["M_B"]

    # line 2.  P and Q are diagonal (Adafactor-style), so they are held as
    # vectors and the products with them are elementwise scalings.  The damping
    # of Appendix E is simply +delta once the largest entry is 1.
    q, p = st["q"], st["p"]
    Q_diag = q / q.max().clamp_min(eps) + delta          # (din,) bf16
    P_diag = p / p.max().clamp_min(eps) + delta          # (dout,) bf16
    # Both Grams accumulate in fp32 a chunk of rows at a time, so neither factor
    # is copied whole (A.T is a view).  Their eigenvalues feed an inverse square
    # root, which is why the accumulation cannot be left in bf16.
    C_B = weighted_gram(B, P_diag)                       # B^T P B     (r, r)
    C_A = weighted_gram(A.T, Q_diag)                     # A Q A^T     (r, r)

    # the inverse square roots are the r x r fp32 work of Appendix E.2; they are
    # cast down once here and used only in bf16 products from this point on
    CB_isqrt = damped_inv_sqrt(C_B, delta, eps, ns_steps, cache, "C_B", pow_iters).to(A.dtype)
    CA_isqrt = damped_inv_sqrt(C_A, delta, eps, ns_steps, cache, "C_A", pow_iters).to(B.dtype)
    Q_isqrt = Q_diag.rsqrt().to(A.dtype)                 # (din,)
    P_isqrt = P_diag.rsqrt().to(B.dtype).unsqueeze(1)    # (dout, 1)

    # lines 3 and 4: the preconditioned polar step, Eq. (19)
    D_A = (CB_isqrt @ msign((CB_isqrt @ Mh_A) * Q_isqrt,
                            ns_steps, msign_method)) * Q_isqrt
    D_B = P_isqrt * (msign((P_isqrt * Mh_B) @ CA_isqrt,
                           ns_steps, msign_method) @ CA_isqrt)

    # line 5: one update size for both factors, bounding the *merged* update by
    # eta.  What the layer sees is s(B dA + dB A), hence the extra /s.
    norm_A = spectral_norm(A, cache, "A", pow_iters)
    norm_B = spectral_norm(B, cache, "B", pow_iters)
    rho = eta / (scaling * max(norm_A + norm_B, eps))

    # line 6
    step_A = rho / max(spectral_norm(D_A, cache, "D_A", pow_iters), eps)
    step_B = rho / max(spectral_norm(D_B, cache, "D_B", pow_iters), eps)
    A.add_(D_A, alpha=-step_A)
    B.add_(D_B, alpha=-step_B)

    # lines 7 and 8: the coupled curvature estimator (32), from the *raw* factor
    # gradients and reusing the curvature matrices formed for line 2.
    #   diag(G_A^T C_B^-1 G_A) = column-wise squared norms of C_B^-1/2 G_A
    # Z and the curvature EMA both stay in bf16.
    Z_A = CB_isqrt @ G_A                                 # (r, din)
    Z_B = G_B @ CA_isqrt                                 # (dout, r)
    q.mul_(beta2).add_(Z_A.pow(2).sum(dim=0, dtype=torch.bfloat16) / r, alpha=1 - beta2)
    p.mul_(beta2).add_(Z_B.pow(2).sum(dim=1, dtype=torch.bfloat16) / r, alpha=1 - beta2)

    # ||dA||_F^2 + ||dB||_F^2, bf16 like the rest of the d-sided state.
    delta_sq = (step_A ** 2 * D_A.pow(2).sum(dtype=torch.bfloat16).item()
                + step_B ** 2 * D_B.pow(2).sum(dtype=torch.bfloat16).item())
    return rho, norm_B / max(norm_A, eps), delta_sq


# ---------------------------------------------------------------------------
# Federated PoLoRA
# ---------------------------------------------------------------------------

def federated_polora(model, loss_name, criterion, train_graphs, device, train_loaders,
                     server_optimizer, server_lr_scheduler, client_lr, opt_params,
                     model_params, server_epoch):
    from utilities import get_gpu_memory

    server_name = opt_params["server_name"]
    beta1 = opt_params["polora_beta1"]
    beta2 = opt_params["polora_beta2"]
    eps = opt_params["polora_eps"]
    delta = opt_params["polora_delta"]
    ns_steps = opt_params["polora_ns_steps"]
    pow_iters = opt_params["polora_power_iters"]
    msign_method = opt_params["polora_msign"]

    # clients differentiate the shared adapter and never take a local step
    opt_params["local_update_ON"] = False

    params = dict(model.named_parameters())
    layers = adapter_layers(model, server_name)
    scalings = adapter_scalings(model, server_name)

    if "polora_state" not in opt_params:
        covered = {n for names in layers.values() for n in names}
        missing = [n for n, p in model.named_parameters()
                   if p.requires_grad and n not in covered]
        assert not missing, (
            "polora only steps LoRA pairs, but these trainable parameters are "
            "not part of one: {}".format(missing[:5]))
        unscaled = [base for base in layers if base not in scalings]
        assert not unscaled, (
            "no PEFT scaling found for {} -- line 5 needs it to bound the merged "
            "update".format(unscaled[:5]))
        state, bytes_used = init_state(params, layers, eps)
        opt_params["polora_state"] = state
        dtype = params[next(iter(layers.values()))[0]].dtype
        s_min, s_max = min(scalings.values()), max(scalings.values())
        print("[polora] {} layers, server state {:.3f} GB ({}), per-client state 0 GB"
              .format(len(layers), bytes_used / 1024 ** 3,
                      str(dtype).replace("torch.", "")))
        print("[polora] beta1={} beta2={} eps={} delta={} msign={} NS steps={} "
              "power iters={} adapter scaling in [{:.4g}, {:.4g}]".format(
                  beta1, beta2, eps, delta, msign_method, ns_steps, pow_iters,
                  s_min, s_max))
    state = opt_params["polora_state"]

    eta = 0
    for group in server_optimizer.param_groups:
        eta = group["lr"]

    # ---- client loop: every client differentiates the same adapter ---------
    # Keep the federated sum in bf16 as well.  The buffers are allocated once
    # and added into in place.
    grad_A = {base: torch.zeros_like(params[names[0]], dtype=torch.bfloat16)
              for base, names in layers.items()}          # -> B^T G   (r, din)
    grad_B = {base: torch.zeros_like(params[names[1]], dtype=torch.bfloat16)
              for base, names in layers.items()}          # -> G A^T   (dout, r)

    def _accumulate(client_id, model_grad):
        for base, (name_A, name_B) in layers.items():
            grad_A[base].add_(model_grad[name_A])
            grad_B[base].add_(model_grad[name_B])

    client_num = collect_client_grads(
        model, loss_name, criterion, train_graphs, device, train_loaders,
        client_lr, opt_params, model_params, server_epoch, _accumulate,
        exclude_from_copy=("polora_state",))
    get_gpu_memory()

    # ---- server: one PoLoRA step per LoRA pair (Algorithm 1) ---------------
    step_norm = 0.0
    rho_sum = 0.0
    balance_sum = 0.0
    for base, (name_A, name_B) in layers.items():
        A, B = params[name_A].data, params[name_B].data     # (r, din), (dout, r)
        rho, balance, delta_sq = polora_step(
            A, B,
            (grad_A[base] / client_num).to(A.dtype),
            (grad_B[base] / client_num).to(B.dtype),
            state[base], eta, scalings.get(base, 1.0),
            beta1, beta2, eps, delta, ns_steps, pow_iters, msign_method)
        rho_sum += rho
        balance_sum += balance
        step_norm += delta_sq

    n_layers = len(layers)
    print("[polora] eta={:.3e} mean rho={:.3e} ||step||_F={:.6f} "
          "mean ||B||_2/||A||_2={:.3f}".format(
              eta, rho_sum / n_layers, step_norm ** 0.5, balance_sum / n_layers))

    if opt_params.get("train_stats", False):
        train_graphs.grad_norm.append(step_norm ** 0.5)

    if server_lr_scheduler is not None:
        server_lr_scheduler.step()
    for group in server_optimizer.param_groups:
        print("server lr", group["lr"])
