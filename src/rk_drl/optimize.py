import os, math
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import cho_factor, cho_solve, eigh

class RKDRL_Optimizer:
    """
    Wraps:
      1) closed‐form solver for the unconstrained quadratic min Q(B)
      2) AdamW‐based gradient descent under soft constraints + structural penalties.
    """
    def __init__(self, device=None, dtype=torch.float64):
        self.dev   = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

    # ─── Hessian operator Q ──────────────────────────────────────────────
    @staticmethod
    def _make_Q_operator(K_Z_np, H_np, G_np, k_np, Phi_np):
        n = k_np.shape[0]
        m = K_Z_np.shape[0]
        kkT     = k_np @ k_np.T
        kPhiT   = k_np @ Phi_np.T
        PhiPhiT = Phi_np @ Phi_np.T
        PhikT   = Phi_np @ k_np.T

        def matvec(x):
            B = x.reshape(n, m)
            M = (kkT       @ B @ K_Z_np
                 - kPhiT   @ B @ H_np
                 - PhikT   @ B @ H_np.T
                 + PhiPhiT @ B @ G_np)
            return M.ravel()

        return LinearOperator((n*m, n*m), matvec=matvec, dtype=np.float64)

    # ─── Closed‐form unconstrained minimizer ─────────────────────────────
    def closed_form_B0(self, k_sa, Phi, K_Zpi, H_mat, G_mat):
        # to NumPy
        k_np   = k_sa.detach().cpu().view(-1,1).numpy()
        Phi_np = Phi.detach().cpu().view(-1,1).numpy()
        K_np   = K_Zpi.detach().cpu().numpy()
        H_np   = H_mat.detach().cpu().numpy()
        G_np   = G_mat.detach().cpu().numpy()

        n, m   = k_np.shape[0], K_np.shape[0]
        Qop    = self._make_Q_operator(K_np, H_np, G_np, k_np, Phi_np)

        # Lanczos for smallest‐magnitude eigenvector
        lam, vecs = eigsh(Qop, k=1, which='SA', tol=1e-6)
        b0 = vecs[:,0]
        if lam[0] > 0:
            b0 = -b0

        B0 = torch.from_numpy(b0.reshape(n, m)).to(self.dev, self.dtype)
        return B0

    # ─── Schur‐complement checks (unchanged) ─────────────────────────────
    @staticmethod
    def check_Q_psd_conditions(K_Z, G, H, k, Phi):
        """
        Strict Schur‐complement dominance test.
        Returns:
          cond1, cond2: booleans for each PSD condition
          mineig1, mineig2: the minimum eigenvalues of the two difference matrices
        """

        # helper to get a stable inverse via Cholesky + jitter
        def chol_inverse(A, jitter=1e-8):
            n = A.shape[0]
            # add tiny jitter to diagonal to ensure PD
            L, lower = cho_factor(A + jitter * np.eye(n), lower=True, check_finite=False)
            # solve A X = I for X
            return cho_solve((L, lower), np.eye(n), check_finite=False)

        # compute inverses of K_Z and G safely
        KZ_inv = chol_inverse(K_Z)
        G_inv  = chol_inverse(G)

        # squared norms of k and Phi vectors
        k2   = float((k.ravel() @ k.ravel()))    # ‖k‖²
        Phi2 = float((Phi.ravel() @ Phi.ravel()))  # ‖Φ‖²

        # form the two Schur‐complement matrices (should be PD if conditions hold)
        M1 = k2 * G - H.T @ KZ_inv @ H
        M2 = Phi2 * K_Z - H   @ G_inv  @ H.T

        # get the smallest eigenvalue of each (must be > 0 for PD)
        mineig1 = np.linalg.eigvalsh(M1).min()
        mineig2 = np.linalg.eigvalsh(M2).min()

        # check strict positivity
        cond1 = mineig1 > 0
        cond2 = mineig2 > 0

        # report results
        print(f"Cond 1 (‖k‖² G − Hᵀ K_Z⁻¹ H ≻ 0): {cond1},  λ₁₋min = {mineig1:.3e}")
        print(f"Cond 2 (‖Φ‖² K_Z − H G⁻¹ Hᵀ ≻ 0): {cond2},  λ₂₋min = {mineig2:.3e}")

        return cond1, cond2, mineig1, mineig2

    # ─── The main optimize method ────────────────────────────────────────
    def optimize(
            self,
            k_sa: torch.Tensor,
            K_Zpi: torch.Tensor,
            H_mat: torch.Tensor,
            Phi: torch.Tensor,
            G_mat: torch.Tensor,
            *,
            initial_B: torch.Tensor | None = None,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            num_steps: int = 2000,
            FP_penalty_lambda: float = 1e2,
            use_low_rank: bool = False,
            rank: int = 300,
            ortho_lambda: float = 0.0,
            B_positive: bool = False,
            fixed_point_constraint: bool = True,
            exact_projection: bool = False,
            B_conv: bool = False,
            Sum_one_W: bool = False,
            NonNeg_W: bool = False,
            mass_anchor_lambda: float = 10.0,  # soft scale to prevent collapse when no W constraints
            target_mass: float = 1.0,  # desired 1^T w for the anchor
            B_ridge_penalty: bool = False,
            verbose: bool = True,
    ) -> tuple[torch.Tensor, list[float], list[float]]:

        # ─── Convert to numpy for PSD checks ───────────────────────────────────
        K_Zpi_np = K_Zpi.cpu().numpy()
        G_mat_np = G_mat.cpu().numpy()
        H_mat_np = H_mat.cpu().numpy()
        k_sa_np = k_sa.cpu().numpy()
        Phi_np = Phi.cpu().numpy()

        # ── Schur PSD checks  ─────────────────────────────────────────────────
        cond1, cond2, mineig1, mineig2 = self.check_Q_psd_conditions(
            K_Zpi_np, G_mat_np, H_mat_np, k_sa_np.reshape(-1, 1), Phi_np.reshape(-1, 1)
        )
        print("+" * 30)
        print(f"Schur PSD tests of Q: cond1={cond1}, cond2={cond2}\n")

        # Determine if constraints or penalties are active (same logic as before)
        has_constraints = (
                fixed_point_constraint or B_ridge_penalty or B_positive or exact_projection
                or B_conv or Sum_one_W or NonNeg_W or ortho_lambda > 0.0
        )

        # If no constraints, use closed-form solution
        if not has_constraints:
            if verbose: print("No constraints ⇒ using closed‐form quadratic minimizer.")
            # ─── closed_form_B0 solve the unconstraint problem ────────
            B0 = self.closed_form_B0(k_sa, Phi, K_Zpi, H_mat, G_mat)
            # compute objective value at B0
            k_vec   = k_sa.view(-1, 1).to(self.dev, self.dtype)
            phi_vec = Phi.view(-1, 1).to(self.dev, self.dtype)
            H_torch = H_mat.to(self.dev, self.dtype)
            G_torch = G_mat.to(self.dev, self.dtype)
            K_Z     = K_Zpi.to(self.dev, self.dtype)

            term1   = (k_vec.t() @ B0 @ K_Z @ B0.t() @ k_vec).squeeze()
            term2   = -2.0 * (k_vec.t() @ B0 @ H_torch @ B0.t() @ phi_vec).squeeze()
            term3   = (phi_vec.t() @ B0 @ G_torch @ B0.t() @ phi_vec).squeeze()
            obj     = term1 + term2 + term3

            bellman_res = (B0.t() @ k_vec) - (B0.t() @ phi_vec)

            print("bellman_res =", torch.norm(bellman_res, p='fro').item())
            print("objective   =", obj.item())
            return B0.cpu(), [], []
        else:
            if verbose: print("The optimization has constraints ⇒ using ADAM minimizer.")

        # Otherwise, proceed with iterative optimization
        # If initial_B is None, use closed-form B0 as initial value
        if initial_B is None:
            if verbose: print(f"Using unconstrained solution as initial B for constrained optimization.\n")
            initial_B = self.closed_form_B0(k_sa, Phi, K_Zpi, H_mat, G_mat)

        # ─── Print settings ───────────────────────────────────────────────────
        print("=" * 50)
        print("Optimization Constraint Enforcements:")
        print("Used low rank factorization for K/G =", use_low_rank, "-- with rank=", rank)
        print("Fixed point constraint penalty:", fixed_point_constraint, "-- with penalty=", FP_penalty_lambda)
        print("Doing exact projection onto the fixed point:", exact_projection)
        print("Orthonormal-rows penalty=", ortho_lambda)
        print("B>0 Penalty:", B_positive)
        print("Convex-ball Penalty:", B_conv)
        print("Row-sum-to-one penalty:", Sum_one_W)
        print("Non-negative W penalty:", NonNeg_W)
        print("Used ridge Penalty for B:", B_ridge_penalty)
        print("verbose to print progress:", verbose)
        print("Mass anchor active (when no W constraints):", (not (Sum_one_W or NonNeg_W)),
              "-- with lambda=", mass_anchor_lambda, ", target_mass=", target_mass)
        print("")

        # ── Move data to device/dtype ─────────────────────────────────────────
        dev, dtype = self.dev, self.dtype
        k_vec = k_sa.view(-1, 1).to(dev, dtype)
        phi_vec = Phi.view(-1, 1).to(dev, dtype)
        H_torch = H_mat.to(dev, dtype)
        G_torch = G_mat.to(dev, dtype)
        K_Z = K_Zpi.to(dev, dtype)

        # ── Low‐rank factorization (optional) ─────────────────────────────────
        if use_low_rank:
            m_dim = G_mat.shape[0]
            eff_rank = max(1, min(rank, m_dim))

            def _sqrt_factor(M: np.ndarray, r: int) -> torch.Tensor:
                vals, vecs = eigh(M)
                idx = np.argsort(vals)[::-1][:r]
                L = vecs[:, idx] @ np.diag(np.sqrt(np.clip(vals[idx], 0, None)))
                return torch.from_numpy(L).to(dev, dtype)

            L_K = _sqrt_factor(K_Zpi_np, eff_rank)
            L_G = _sqrt_factor(G_mat_np, eff_rank)

        # ── Optionally precompute ridge_eps ───────────────────────────────────
        ridge_eps = 0.0
        if B_ridge_penalty:
            Qop_np = self._make_Q_operator(
                K_Zpi_np, H_mat_np, G_mat_np, k_sa_np.reshape(-1, 1), Phi_np.reshape(-1, 1)
            )
            lam_min = eigsh(Qop_np, k=1, which='SA', return_eigenvectors=False)[0]
            epsilon = 1e-2
            ridge_eps = max(0.0, -lam_min) + epsilon
            print(f"Adding ridge_eps = {ridge_eps:.3e} to make Q + ridge·I ≽ 0\n")

        # ── Initialize B ──────────────────────────────────────────────────────
        n_dim, m_dim = k_vec.shape[0], G_torch.shape[0]
        B = torch.nn.Parameter(initial_B.to(dev, dtype))

        # ── Optimizer & LR scheduler ─────────────────────────────────────────
        opt = optim.AdamW([B], lr=lr, weight_decay=weight_decay)
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10, threshold=1e-3, min_lr=1e-8
        )

        history_obj, history_be = [], []

        for step in range(1, num_steps + 1):
            opt.zero_grad()

            # ---- core quadratic terms ----
            if use_low_rank:
                term1 = (k_vec.t() @ (B @ L_K)).pow(2).sum()
                term3 = (phi_vec.t() @ (B @ L_G)).pow(2).sum()
            else:
                term1 = (k_vec.t() @ B @ K_Z @ B.t() @ k_vec).squeeze()
                term3 = (phi_vec.t() @ B @ G_torch @ B.t() @ phi_vec).squeeze()
            term2 = -2.0 * (k_vec.t() @ B @ H_torch @ B.t() @ phi_vec).squeeze()
            obj = term1 + term2 + term3

            # ----  ridge ℓ₂ on B ----
            if B_ridge_penalty:
                obj = obj + ridge_eps * torch.norm(B, p='fro') ** 2

            # ---- structural penalties ----
            if ortho_lambda > 0:
                I_n = torch.eye(n_dim, device=dev, dtype=dtype)
                obj = obj + ortho_lambda * torch.norm(B @ B.t() - I_n, p='fro') ** 2

            if B_conv:
                obj = obj + torch.relu(torch.norm(B, p='fro') ** 2 - 1.0) ** 2

            # ---- fixed_point_constraint Bellman residual penalty ----
            bellman_res = (B.t() @ k_vec) - (B.t() @ phi_vec)

            # --- Nscale anchor to avoid collapse  ---
            if not (Sum_one_W or NonNeg_W):
                mass = (B.t() @ k_vec).sum()  # 1^T w
                obj = obj + mass_anchor_lambda * (mass - target_mass).pow(2)

            if fixed_point_constraint:
                if FP_penalty_lambda <= 0:
                    raise ValueError(
                        f"FP_penalty_lambda must be in positive with fixed_point_constraint=True. It is {FP_penalty_lambda}. \n"
                    )
                loss = obj + FP_penalty_lambda * bellman_res.pow(2).sum()
            else:
                loss = obj

            loss.backward()
            clip_grad_norm_([B], max_norm=1e2)
            opt.step()

            # ---- projection on weights (only if requested) ----
            if Sum_one_W or NonNeg_W:
                with torch.no_grad():
                    w = B.t() @ k_vec  # [m,1]
                    if Sum_one_W:  # {w>=0, 1^T w = 1}
                        u = w.view(-1).sort(descending=True).values
                        css = torch.cumsum(u, 0)
                        j = torch.arange(1, u.numel() + 1, device=w.device, dtype=w.dtype)
                        rho = torch.nonzero(u > (css - 1) / j)[-1]
                        theta = (css[rho] - 1) / (rho + 1).to(w.dtype)
                        w_hat = torch.clamp(w - theta, min=0.0)
                    else:  # NonNeg_W
                        w_hat = torch.clamp(w, min=0.0)

                    denom = (k_vec * k_vec).sum().clamp_min(1e-12)
                    B.add_(k_vec @ (w_hat - w).t() / denom)

            # ---- clamp positivity if requested ----
            if B_positive:
                with torch.no_grad():
                    B.data[:] = B.clamp(min=0.0)

            if exact_projection:
                with torch.no_grad():
                    # current residual: r = Bᵀk − BᵀΦ  (shape [m,1])
                    r = (B.t() @ k_vec) - (B.t() @ phi_vec)  # shape [m,1]
                    d = (k_vec - phi_vec)  # d = k − φ  (shape [n,1])
                    denom = (d * d).sum()  # scalar = ‖k−φ‖²
                    if denom > 1e-12:  # If denom is tiny, skip projection to avoid NaNs
                        DeltaB = - d @ r.t()
                        DeltaB.div_(denom)  # ΔB = - d @ rᴛ / denom
                        B.data.add_(DeltaB)

            # ---- Compute Bellman-error norm ----
            with torch.no_grad():
                be_norm = torch.norm(bellman_res, p='fro').item()

            # ---- Record histories ----
            curr_loss = loss.item()
            history_obj.append(curr_loss)
            history_be.append(be_norm)
            grad_norm = B.grad.norm().item()

            # ---- Verbose logging ----
            if verbose and step % (num_steps // 10) == 0:
                print(
                    f"Iter {step}/{num_steps}|  ‖∇B‖={grad_norm:.2e} | "
                    f"loss={curr_loss:.3e}| log-loss={math.log(curr_loss):.3e} | "
                    f"Bellman Error={be_norm:.3e} | log-Bellman Error={math.log(be_norm):.3e} "
                )

            # ---- Early stopping on small gradient ----
            if grad_norm < 1e-7:
                if verbose:
                    print(f"Converged at step {step}: ‖∇B‖={grad_norm:.2e}")
                break

            # ---- record & schedule ----
            sched.step(curr_loss)

        history_be = [math.log(x) for x in history_be]
        history_obj = [math.log(x) for x in history_obj]

        return B.detach().cpu(), history_obj, history_be

