import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv
import torch
import time
import math
# import torch_bessel



# def matern_kernel(x:np.ndarray,
#                   y:np.ndarray,
#                   nu=1.5, length_scale=1.0):
#
#     kernel = Matern(length_scale=length_scale, nu=nu)
#     return kernel(x, y)

def matern_kernel_np(X1, X2, nu, length_scale, sigma=1.0):
    """
    Compute the Matérn kernel matrix between two sets of vectors.
    Parameters:
        - x (np.ndarray): Array of shape (n_samples_x, n_features) representing input points.
        - y (np.ndarray): Array of shape (n_samples_y, n_features) representing input points.
        - nu (float, optional): Smoothness parameter of the Matern kernel.
        - length_scale (float, optional): Length scale parameter of the Matern kernel.
        - sigma (float, optional):  parameter of the Matern kernel. default is 1.0.

    Returns:
        - kernel_matrix (np.ndarray): The computed Matern kernel matrix of shape (n_samples_x, n_samples_y).
    """
    dist = cdist(X1, X2, metric='euclidean')
    scaled_dist = np.sqrt(2 * nu) * dist / length_scale
    scaled_safe = np.maximum(scaled_dist, np.finfo(float).eps)

    coeff  = sigma ** 2 * (2 ** (1 - nu)) / gamma(nu)
    kernel = coeff * (scaled_safe ** nu) * kv(nu, scaled_safe)
    kernel[dist == 0] = sigma ** 2  # variance on the diagonal

    return kernel


@torch.no_grad()
def matern_kernel(X1: torch.Tensor,
                  X2: torch.Tensor,
                  nu: float,
                  length_scale: float,
                  sigma: float = 1.0) -> torch.Tensor:
    """
    Matérn kernel for ν = p + 0.5 (p ∈ ℕ₀), memory-safe and drop-in compatible.

    Args:
        X1: (N, D)
        X2: (M, D)
        nu: p + 0.5
        length_scale: ℓ > 0
        sigma: σ > 0
    Returns:
        (N, M) kernel on the same device/dtype as X1
    """
    # Validate
    if X1.ndim != 2 or X2.ndim != 2 or X1.size(1) != X2.size(1):
        raise ValueError("X1, X2 must be 2D with same feature dim.")
    if length_scale <= 0 or sigma <= 0:
        raise ValueError("length_scale and sigma must be > 0.")
    p = int(nu - 0.5)
    if abs(nu - (p + 0.5)) > 1e-8:
        raise ValueError(f"nu={nu} must be half-integer (p + 0.5)")

    # Align device/dtype exactly like the original
    X1 = X1.contiguous()
    X2 = X2.to(device=X1.device, dtype=X1.dtype).contiguous()
    N, M = X1.size(0), X2.size(0)
    if N == 0 or M == 0:
        return torch.empty(N, M, device=X1.device, dtype=X1.dtype)

    # Precompute constants and coefficients
    # We evaluate the closed-form with t = 2*sqrt(2ν)*||x-x'||/ℓ
    t_scale = 2.0 * math.sqrt(2.0 * nu)
    prefac = (sigma ** 2) * (math.factorial(p) / math.factorial(2 * p))
    coeffs = [math.factorial(2 * p - m) // (math.factorial(p - m) * math.factorial(m)) for m in range(p + 1)]

    # Output on the same device/dtype as X1
    K = torch.empty(N, M, device=X1.device, dtype=X1.dtype)

    # Choose a block size that fits GPU memory (2 big work buffers per block: t and poly)
    # On CPU we just do it in one go.
    if X1.is_cuda:
        try:
            free_bytes, _ = torch.cuda.mem_get_info(X1.device)
        except Exception:
            free_bytes = 0
        esize = torch.tensor([], dtype=X1.dtype, device=X1.device).element_size()
        # Reserve ~60% of reported free memory for safety and assume 2 buffers
        safety = 0.60
        buffers = 2
        # elements per column-block = N * B -> B = available_bytes / (buffers * N * esize)
        B_est = int(max(1, (free_bytes * safety) // max(1, buffers * N * esize)))
        block = int(min(M, max(1, B_est)))
    else:
        block = M  # CPU: no special blocking needed by default

    # Blocked compute to avoid extra full N×M temporaries
    j0 = 0
    while j0 < M:
        j1 = min(M, j0 + block)
        # t := pairwise distances / ℓ
        t = torch.cdist(X1, X2[j0:j1], p=2.0)
        t.div_(float(length_scale))
        # t := 2*sqrt(2ν)*dist/ℓ
        t.mul_(t_scale)

        # Horner evaluation of polynomial in t (no powers, minimal temps)
        poly = t.new_full(t.shape, float(coeffs[-1]))
        for m in range(p - 1, -1, -1):
            poly.mul_(t).add_(float(coeffs[m]))

        # Turn t into exp(-t/2) in-place and finish: K_block = prefac * poly * exp(-t/2)
        t.mul_(-0.5).exp_()
        poly.mul_(t).mul_(prefac)

        # Write block
        K[:, j0:j1] = poly

        # Release temps before next block
        del t, poly
        j0 = j1

        # (Optional) adapt block size if CUDA memory fluctuates
        if X1.is_cuda:
            try:
                free_bytes, _ = torch.cuda.mem_get_info(X1.device)
                esize = torch.tensor([], dtype=X1.dtype, device=X1.device).element_size()
                B_est = int(max(1, (free_bytes * 0.60) // max(1, 2 * N * esize)))
                if B_est > 0:
                    block = int(min(M - j0, max(1, B_est)))
            except Exception:
                pass

    return K
