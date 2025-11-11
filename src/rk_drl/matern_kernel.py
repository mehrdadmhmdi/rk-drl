import math, time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv
import torch
from contextlib import nullcontext

# --- internal cap: max temporary elements per block (tune if needed) ---
_MAX_BLOCK_ELEMS = 2_000_000  # ~2M elems ≈ 8 MB in float32, 16 MB in float64

def matern_kernel_np(X1, X2, nu, length_scale, sigma=1.0):
    """
    Matérn kernel (NumPy/SciPy) with row-chunking to limit peak memory.
    Returns (X1.shape[0], X2.shape[0]) ndarray.
    """
    X1 = np.asarray(X1, dtype=float, order="C")
    X2 = np.asarray(X2, dtype=float, order="C")
    m, n = X1.shape[0], X2.shape[0]
    out = np.empty((m, n), dtype=float)

    ell = float(length_scale)
    if ell <= 0 or sigma <= 0:
        raise ValueError("length_scale and sigma must be > 0.")
    if nu <= 0:
        raise ValueError("nu must be > 0.")

    # rows per block so that (bs * n) <= cap
    bs = max(1, min(m, _MAX_BLOCK_ELEMS // max(1, n)))
    root2nu_over_ell = math.sqrt(2.0 * nu) / ell
    coeff = (sigma ** 2) * (2.0 ** (1.0 - nu)) / gamma(nu)

    for i in range(0, m, bs):
        Xi = X1[i:i+bs]
        # Euclidean distances for the block
        D = cdist(Xi, X2, metric='euclidean')  # (bs, n)
        z = root2nu_over_ell * D
        # avoid 0^nu in Bessel term; Bessel handles >0
        z_safe = np.maximum(z, np.finfo(float).eps)
        Kblk = coeff * np.power(z_safe, nu) * kv(nu, z_safe)
        # exact variance on zero distances (diagonal when X1==X2 and block aligns)
        Kblk[D == 0.0] = (sigma ** 2)
        out[i:i+bs] = Kblk
        # free big temporaries
        del Xi, D, z, z_safe, Kblk

    return out


def matern_kernel(X1: torch.Tensor, X2: torch.Tensor, nu: float, length_scale: float, sigma: float = 1.0) -> torch.Tensor:
    """
    Matérn kernel for half-integer ν = p + 0.5 using exp×poly closed form,
    computed in row-chunks to bound memory. Returns (N, M) tensor on X1.device.
    """
    if X1.ndim != 2 or X2.ndim != 2 or X1.size(1) != X2.size(1):
        raise ValueError("X1, X2 must be 2D with same feature dim.")
    if length_scale <= 0 or sigma <= 0:
        raise ValueError("length_scale and sigma must be > 0.")
    # enforce half-integer ν (p ∈ ℕ₀)
    p = int(nu - 0.5)
    if abs(nu - (p + 0.5)) > 1e-8:
        raise ValueError(f"nu={nu} must be half-integer (p + 0.5).")

    # align device/dtype; keep contiguity
    X1 = X1.contiguous()
    X2 = X2.to(device=X1.device, dtype=X1.dtype).contiguous()

    m, n = X1.shape[0], X2.shape[0]
    out = X1.new_empty((m, n))

    ell = float(length_scale)
    s2  = float(sigma) ** 2
    root2nu = math.sqrt(2.0 * nu)

    # choose rows-per-block so (bs * n) <= cap
    bs = max(1, min(m, _MAX_BLOCK_ELEMS // max(1, n)))

    # optional AMP on CUDA for lower peak memory
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (X1.is_cuda) else nullcontext()

    # precompute integer polynomial coefficients on CPU (Horner later on GPU)
    # a_m = (2p - m)! / ( (p - m)! * m! ), m=0..p
    coeffs = [math.factorial(2 * p - m) // (math.factorial(p - m) * math.factorial(m)) for m in range(p + 1)]
    prefac = s2 * (math.factorial(p) / math.factorial(2 * p))

    with torch.no_grad(), amp_ctx:
        for i in range(0, m, bs):
            Xi = X1[i:i+bs]                              # (bs, d)
            D  = torch.cdist(Xi, X2, p=2.0)              # (bs, n)  :contentReference[oaicite:1]{index=1}
            # z = sqrt(2ν) * d / ℓ
            z  = D.mul_(root2nu / max(ell, 1e-12))
            E  = torch.exp(-z)                            # exp(-z) — computed per block
            # Horner in t=2z for the polynomial part
            t  = z.mul(2.0)
            poly = t.new_full(t.shape, float(coeffs[-1]))
            for m_ in range(p - 1, -1, -1):
                poly.mul_(t).add_(float(coeffs[m_]))
            Kblk = prefac * E * poly
            out[i:i+bs] = Kblk

            # release block temps early
            del Xi, D, z, E, t, poly, Kblk

    return out
