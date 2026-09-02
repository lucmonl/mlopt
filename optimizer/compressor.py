"""Layer-wise sparsification operators.

These implement Definition 1 of "Error Feedback for Muon and Friends"
(Gruntkowska et al., ICLR 2026): a (possibly randomized) map C is contractive
with parameter alpha in (0, 1] if

    E[ ||C(X) - X||^2 ] <= (1 - alpha) ||X||^2   for all X.

Compression is applied **per tensor**: each weight matrix gets its own budget
k_i = ratio * numel_i, so every layer is represented in every message rather
than competing for one global budget.

Note this is *not* the same requirement as unbiasedness.  ``tensor_randk`` in
``utilities.py`` rescales the retained entries by n/k so that E[C(X)] = X, which
makes it unbiased but *not* contractive (the residual is inflated by the same
factor).  Error feedback needs contractivity, so ``rand_k`` below keeps the
selected entries unscaled: it satisfies Definition 1 with alpha = k/n.
``top_k`` likewise selects over the flattened tensor rather than along the last
axis, which is what ``torch.topk`` on a 2-D tensor would do.
"""

import math

import torch

COMPRESSORS = ["topk", "randk", "none"]


def resolve_k(numel, ratio):
    """Budget for one tensor.

    ratio in (0, 1) is a fraction of that tensor's entries (0.02 -> 2%);
    ratio >= 1 is an absolute count; ratio <= 0 disables compression.
    """
    if ratio is None or ratio <= 0:
        return numel
    if ratio < 1:
        return max(1, int(round(float(ratio) * numel)))
    return min(int(ratio), numel)


def top_k(x, k):
    """Keep the k largest-magnitude entries of x, zero the rest."""
    flat = x.reshape(-1)
    out = torch.zeros_like(flat)
    _, idx = torch.topk(flat.abs(), k, sorted=False)
    out[idx] = flat[idx]
    return out.view_as(x)


def rand_k(x, k):
    """Keep k uniformly random entries of x, zero the rest, *without* the n/k
    unbiasedness rescaling.  Contractive with alpha = k/n."""
    flat = x.reshape(-1)
    out = torch.zeros_like(flat)
    idx = torch.randperm(flat.numel(), device=flat.device)[:k]
    out[idx] = flat[idx]
    return out.view_as(x)


def compress(x, compressor, ratio):
    """Sparsify a single tensor.  Returns (compressed tensor, entries kept)."""
    if compressor is None or compressor == "none":
        return x, x.numel()
    k = resolve_k(x.numel(), ratio)
    if k >= x.numel():
        return x, x.numel()
    if compressor == "topk":
        return top_k(x, k), k
    if compressor == "randk":
        return rand_k(x, k), k
    raise NotImplementedError("unknown compressor: {}".format(compressor))


def compressed_bits(nnz, numel, value_bits=32):
    """Bits on the wire for a sparse message: values plus their indices."""
    if nnz >= numel:
        return numel * value_bits
    index_bits = max(1, math.ceil(math.log2(max(numel, 2))))
    return nnz * (value_bits + index_bits)
