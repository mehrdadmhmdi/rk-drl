from __future__ import annotations

import math
import os
import time
import numpy as np
import torch

# Prefer explicit imports to avoid namespace leakage
from .matern_kernel import matern_kernel
from .Gamma_sa import Gamma_sa
from .G_sa import compute_transformed_grid_pytorch, compute_G_pytorch_batched
from .ZGrid import ZGrid
from .H_sa import H_sa
from .Phi_sa import Phi_sa
from .IS_ULSIF import ULSIFEstimator
from .optimize import RKDRL_Optimizer


def RK_DRL(
    *,
    # --- core data ---
    s0: torch.Tensor,               # (N, Ds) current states
    s1: torch.Tensor,               # (N, Ds) next states
    a1: torch.Tensor,               # (N, Da) next actions
    a0: torch.Tensor,               # (N, Da) current actions
    s_star: torch.Tensor,           # (N* or 1, Ds) eval state(s)
    a_star: torch.Tensor,           # (N* or 1, Da) eval action(s)
    r: torch.Tensor,                # (n_rewards, Dr) reward samples (for Z grid, G/H)
    # --- policy + kernel/alg params (required) ---
    target_p_choice: str,
    target_p_params: dict,
    nu: float,
    length_scale: float,
    sigma: float = 1.0,
    gamma_val: float = 0.9,
    lambda_reg: float = 1e-6,
    num_grid_points: int = 200,
    # --- sensible defaults for everything else ---
    hull_expand_factor: float = 1.8,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    num_steps: int = 5000,
    FP_penalty_lambda: float = 1e2,
    use_low_rank: bool = False,
    rank_for_low_rank: int | None = None,
    B_positive: bool = False,
    fixed_point_constraint: bool = True,
    exact_projection: bool = False,
    ortho_lambda: float = 1e1,
    B_conv: bool = False,
    Sum_one_W: bool = False,
    NonNeg_W: bool = False,
    mass_anchor_lambda: float = 10.0,     # soft scale to prevent collapse when no W constraints
    target_mass: float = 1.0,             # desired 1^T w for the anchor
    B_ridge_penalty: bool = False,
    H_batch_size: int = 10,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,                 # new: control printing without changing defaults
):
    """
    End-to-end RKHS-based DRL pipeline (Torch).

    Steps
    -----
    1) Form (s,a) design matrices.
    2) Estimate uLSIF alpha (importance weights) for policy shift.
    3) Build Z support via KMeans + convex-hull expansion.
    4) Compute kernel Gram matrices for (s,a), (s',a'), and Z.
    5) Solve Γ (Gamma) and Φ (Phi) operators in RKHS.
    6) Build G and H operators over Z.
    7) Optimize B with optional constraints (low-rank, nonnegativity, convexity, mass anchor, etc.).

    Returns
    -------
    B_hat : torch.Tensor
        Learned coefficient matrix on Z-support (shape depends on optimizer’s parameterization).
    history_obj : list[float]
        Objective values per iteration.
    history_be : list[float]
        Bellman error (or auxiliary diagnostics) per iteration.
    pre : dict[str, torch.Tensor]
        Cache of intermediate matrices:
        { "Z", "k_sa", "K_sa", "K_sa_prime", "K_Z", "H_emp", "G_emp", "Phi_emp", "Gamma_sa_emp" }.
    """
    t0 = time.time()

    if verbose:
        print("=" * 40)
        print("Estimating the mean embedding via RK-DRL")

    # ------------------------------------------------------------------------------
    # 0) Setup / device / dtype
    # ------------------------------------------------------------------------------
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # move & cast once; keep shapes and memory coherent
    def TD(x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=dev)

    s0, s1, a0, a1, s_star, a_star, r = map(TD, (s0, s1, a0, a1, s_star, a_star, r))

    # ------------------------------------------------------------------------------
    # 1) Preconditions / Contraction checks
    #    - Check shapes, basic validity, and contraction proxy needed by theory.
    # ------------------------------------------------------------------------------
    # Shape checks (strict to avoid silent broadcasting bugs)
    if s0.ndim != 2 or a0.ndim != 2:
        raise ValueError("s0 and a0 must be 2D: (N, Ds), (N, Da).")
    if s1.ndim != 2 or a1.ndim != 2:
        raise ValueError("s1 and a1 must be 2D: (N, Ds), (N, Da).")
    if s0.shape[0] != a0.shape[0] or s1.shape[0] != a1.shape[0] or s0.shape[0] != s1.shape[0]:
        raise ValueError("Row counts must match: N(s0)=N(a0)=N(s1)=N(a1).")
    if s_star.ndim not in (1, 2) or a_star.ndim not in (1, 2):
        raise ValueError("s_star and a_star must be rank-1 or rank-2 (use 1D for single query).")

    # Ensure evaluation pair is 2D (N*, ·). Avoid re-casting (keeps device/dtype).
    if s_star.ndim == 1:
        s_star = s_star.unsqueeze(0)
    if a_star.ndim == 1:
        a_star = a_star.unsqueeze(0)

    # Basic parameter checks
    if not (0.0 < gamma_val < 1.0):
        raise ValueError("gamma_val (discount) must be in (0, 1).")
    if nu <= 1.0:
        raise ValueError("nu must exceed 1.0 for the Lipschitz bound used in the contraction proxy.")
    if r.ndim != 2:
        raise ValueError("r must be 2D: (n_rewards, Dr).")
    if num_grid_points < 2:
        raise ValueError("num_grid_points must be >= 2.")

    # Contraction proxy L_k (Matérn Lipschitz bound) and sufficient condition γ·L_k < 1
    Dr = r.shape[1]
    L_k = (sigma * math.sqrt(nu * Dr / (nu - 1))) / max(1e-12, length_scale)  # safe denom
    if not (gamma_val * L_k < 1.0):
        needed_ell = gamma_val * sigma * math.sqrt((Dr * nu) / (nu - 1.0))
        raise ValueError(
            "\n========================================\n"
            "CONTRACTION CONDITION NOT MET\n"
            "========================================\n"
            f"Current: length_scale ℓ={length_scale:.6f}, nu={nu:.3g}, sigma={sigma:.3g}, Dr={Dr}, γ={gamma_val:.3f}\n"
            f"Require: ℓ > {needed_ell:.6f} to ensure γ·L_k<1.\n"
        )

    if verbose:
        Ds, Da = s0.shape[1], a0.shape[1]
        print(f"Data dims: N={s0.shape[0]}, Ds={Ds}, Da={Da}, Dr={Dr}")
        print(f"(s*, a*) dims: ({s_star.shape[0]}, {s_star.shape[1]}) × ({a_star.shape[0]}, {a_star.shape[1]})")

    # ------------------------------------------------------------------------------
    # 2) Build (s,a) blocks
    # ------------------------------------------------------------------------------
    s_a       = torch.cat([s0, a0], dim=1)         # (N, Ds+Da)
    s_a_prime = torch.cat([s1, a1], dim=1)         # (N, Ds+Da)
    s_a_star  = torch.cat([s_star, a_star], dim=1) # (N*, Ds+Da)

    # ------------------------------------------------------------------------------
    # 3) uLSIF importance weights α (policy shift)
    #    - Returns α ~ dπ_target / dπ_behavior
    # ------------------------------------------------------------------------------
    ulsif = ULSIFEstimator(
        kernel_func=matern_kernel,
        lambda_reg=lambda_reg,
        nu=nu,
        length_scale=length_scale,
        sigma=sigma,
    )
    # Fit on (s1, a1) to estimate weights towards target policy at next step
    alpha = ulsif.fit(s1, a1, target_p_choice, target_p_params, plot=False)
    alpha_t = torch.as_tensor(alpha, dtype=dtype, device=dev).view(-1, 1)

    if verbose:
        print("\nα estimated via uLSIF.")
        try:
            ess_train = ulsif.compute_ess(s1, a1)
            print(f"ESS (training batch): {ess_train:.1f} / {s1.shape[0]}")
        except Exception:
            print("ESS unavailable (compute_ess not provided).")
        print("+" * 15)

    # ------------------------------------------------------------------------------
    # 4) Z-grid via k-means + hull expansion
    # ------------------------------------------------------------------------------
    Z = ZGrid.Z_kmeans(r, n_clusters=int(num_grid_points), constant_factor=float(hull_expand_factor))
    if verbose:
        print("Z_grid shape:", tuple(Z.shape))

    # ------------------------------------------------------------------------------
    # 5) Kernel Gram matrices
    # ------------------------------------------------------------------------------
    # k_sa: cross Gram between behavior (s,a) and query (s*,a*); used for evaluation weight vector
    k_sa       = matern_kernel(s_a,       s_a_star,  nu=nu, length_scale=length_scale, sigma=sigma)  # (N, N*)
    K_sa       = matern_kernel(s_a,       s_a,       nu=nu, length_scale=length_scale, sigma=sigma)  # (N, N)
    K_sa_prime = matern_kernel(s_a_prime, s_a_prime, nu=nu, length_scale=length_scale, sigma=sigma)  # (N, N)
    K_Z        = matern_kernel(Z,         Z,         nu=nu, length_scale=length_scale, sigma=sigma)  # (m, m)

    # ------------------------------------------------------------------------------
    # 6) Γ(s,a): solve (K_sa + λI)^(-1) k_sa   (per query column; here we pass all at once)
    # ------------------------------------------------------------------------------
    Gamma_sa_emp = Gamma_sa(K_sa, k_sa, lambda_reg)   # (N, N*) effective representer coefficients

    # ------------------------------------------------------------------------------
    # 7) Φ: propagate Γ with α to form feature map (policy-weighted)
    # ------------------------------------------------------------------------------
    Phi_emp = Phi_sa(K_sa_prime, Gamma_sa_emp, alpha_t)  # (N, N*) or (N, 1) depending on s_a_star

    # ------------------------------------------------------------------------------
    # 8) G operator on Z (batched)
    #    - G encodes the transition of Z under Bellman transform (γZ + r), projected via kernels
    # ------------------------------------------------------------------------------
    T_pt = compute_transformed_grid_pytorch(Z, r, gamma_val)     # shape depends on implementation
    G_emp = compute_G_pytorch_batched(
        T_pt, Gamma_sa_emp, nu=nu, length_scale=length_scale, sigma=sigma
    )  # (m, m)

    # ------------------------------------------------------------------------------
    # 9) H operator on Z (row-batched to limit memory)
    # ------------------------------------------------------------------------------
    H_emp = H_sa(
        Gamma_sa_emp, gamma_val, r, Z,
        nu=nu, length_scale=length_scale, sigma=sigma,
        batch_size=int(H_batch_size)
    )  # (m, m)

    # ------------------------------------------------------------------------------
    # 10) Optimize B
    #      - If low-rank enabled, pick rank (default ~ m/2).
    #      - Closed-form warm start (B0) helps stability.
    # ------------------------------------------------------------------------------
    N = s_a.shape[0]
    m = Z.shape[0]
    optimizer = RKDRL_Optimizer(device=dev, dtype=dtype)

    initial_B_guess = optimizer.closed_form_B0(
        k_sa=k_sa,
        Phi=Phi_emp,
        K_Zpi=K_Z,
        H_mat=H_emp,
        G_mat=G_emp,
    )

    if use_low_rank:
        rk = rank_for_low_rank if rank_for_low_rank is not None else max(1, m // 2)
        if verbose:
            print("Low-rank optimization enabled with rank =", rk)
    else:
        rk = None

    if verbose:
        print("~" * 40)

    B_hat, history_obj, history_be = optimizer.optimize(
        k_sa=k_sa,
        K_Zpi=K_Z,
        H_mat=H_emp,
        Phi=Phi_emp,
        G_mat=G_emp,
        initial_B=initial_B_guess,
        lr=lr,
        weight_decay=weight_decay,
        num_steps=int(num_steps),
        FP_penalty_lambda=FP_penalty_lambda,
        use_low_rank=use_low_rank,
        rank=rk,
        ortho_lambda=ortho_lambda,
        B_positive=B_positive,
        exact_projection=exact_projection,
        fixed_point_constraint=fixed_point_constraint,
        B_conv=B_conv,
        Sum_one_W=Sum_one_W,
        NonNeg_W=NonNeg_W,
        mass_anchor_lambda=mass_anchor_lambda,
        target_mass=target_mass,
        B_ridge_penalty=B_ridge_penalty,
        verbose=verbose,
    )

    # Make sure output is a torch.Tensor on the working device/dtype
    B_hat_torch = torch.as_tensor(B_hat, dtype=dtype, device=dev)

    # ------------------------------------------------------------------------------
    # Pack cache for downstream analysis / debugging
    # ------------------------------------------------------------------------------
    pre_computed_matrices = {
        "Z_grid": Z,
        "k_sa": k_sa,
        "K_sa": K_sa,
        "K_sa_prime": K_sa_prime,
        "K_Z": K_Z,
        "H": H_emp,
        "G": G_emp,
        "Phi": Phi_emp,
        "Gamma_sa": Gamma_sa_emp,
    }

    if verbose:
        dt = time.time() - t0
        print(f"Done in {dt:.2f}s.")

    # Avoid holding onto large tensors beyond return
    del s0, s1, a0, a1, s_star, a_star, r, s_a, s_a_prime, s_a_star
    del ulsif, alpha, alpha_t, T_pt, optimizer, initial_B_guess, N, m, rk

    return B_hat_torch, history_obj, history_be, pre_computed_matrices
