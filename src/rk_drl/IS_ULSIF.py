import torch
import matplotlib.pyplot as plt
import seaborn as sns  # only for plotting
from .Probability_Densities import Probability_Densities
from .matern_kernel import matern_kernel
from pathlib import Path

class ULSIFEstimator:
    def __init__(self, kernel_func=matern_kernel, lambda_reg: float = 1e-3,
                 nu: float = 1.5, length_scale: float = 1.0, sigma: float = 1.0):
        self.kernel_func = kernel_func
        self.lambda_reg = lambda_reg
        self.kernel_kwargs = {'nu': nu, 'length_scale': length_scale, 'sigma': sigma}
        self.alpha = None               # (n_beta, 1) torch tensor
        self._X_beta = None             # (n_beta, d_s + d_a) torch tensor

    def _sample_target(self, action_dim, s: torch.Tensor,
                       target_p_choice: str, target_p_params: dict) -> torch.Tensor:
        """
        Torch version of target action sampling.
        Keeps original logic: sample per (i, dim) from Probability_Densities_*.
        """
        if not target_p_params:
            raise ValueError("target_p_params must be provided for Torch sampling.")
        prob_density = Probability_Densities(**target_p_params)

        n = s.shape[0]
        a_pi = torch.empty((n, action_dim), device=s.device, dtype=s.dtype)
        for dim in range(action_dim):
            for i in range(n):
                sample = prob_density.sample_pdf(target_p_choice, s[i, :])
                if sample is None:
                    raise RuntimeError("sample_pdf returned None; check target_p_choice/params.")
                # sample may be shape (1,) — take scalar
                a_pi[i, dim] = sample.reshape(-1)[0].to(device=s.device, dtype=s.dtype)
        return a_pi

    def fit(self,
            S: torch.Tensor,
            A: torch.Tensor,
            target_p_choice: str,
            target_p_params: dict,
            plot: bool = True) -> torch.Tensor:
        """
        Torch version of uLSIF fit. Mirrors original math:
          X_beta = [S, A]
          X_pi   = [S, a_pi]
          K_bb, K_bp via kernel
          H = (K_bb @ K_bb^T)/n_beta
          h = (K_bp @ 1)/n_pi
          alpha = solve(H + λI, h) then refined via Cholesky
        """
        device = S.device
        dtype  = S.dtype

        # ensure 2D
        if S.ndim != 2 or A.ndim != 2:
            raise ValueError("S and A must be 2D tensors.")
        n, d_a = A.shape

        X_beta = torch.cat([S, A], dim=1).to(device=device, dtype=dtype)  # (n_beta, d)
        self._X_beta = X_beta

        a_pi = self._sample_target(d_a, S.to(device, dtype), target_p_choice, target_p_params)  # (n, d_a)
        X_pi = torch.cat([S.to(device, dtype), a_pi], dim=1)                                     # (n_pi, d)

        # Kernels (Torch)
        K_bb = self.kernel_func(X_beta, X_beta, **self.kernel_kwargs)  # (n_beta, n_beta)
        K_bp = self.kernel_func(X_beta, X_pi,   **self.kernel_kwargs)  # (n_beta, n_pi)

        n_beta = K_bb.shape[0]
        n_pi   = K_bp.shape[1]

        H = (K_bb @ K_bb.T) / float(n_beta)                            # (n_beta, n_beta)
        h = (K_bp @ torch.ones(n_pi, device=device, dtype=dtype)) / float(n_pi)  # (n_beta,)

        # Solve alpha as in original: first solve, then Cholesky refinement
        I = torch.eye(n_beta, device=device, dtype=dtype)
        reg = self.lambda_reg * I

        # direct solve
        alpha_direct = torch.linalg.solve(H + reg, h)                  # (n_beta,)

        # Cholesky solve (refinement, same as original)
        L = torch.linalg.cholesky(H + reg)
        y = torch.cholesky_solve(h.unsqueeze(1), L).squeeze(1) * 0.0   # placeholder to keep flow identical
        # original code: y = solve(L, h); alpha = solve(L^T, y)
        y   = torch.linalg.solve(L, h)                                  # (n_beta,)
        alpha_chol = torch.linalg.solve(L.T, y)                         # (n_beta,)

        # match original: store the Cholesky version
        self.alpha = alpha_chol.reshape(-1, 1)                          # (n_beta, 1)

        if plot:
            Path('plots').mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                eta_hat = (K_bb @ self.alpha).squeeze(1)                # (n_beta,)
            plt.figure()
            sns.histplot(eta_hat.detach().cpu().numpy(), kde=True, bins=30)
            plt.title(f"Distribution of uLSIF eta Coefficients ,Target = {target_p_choice}")
            plt.savefig(f'./plots/eta_uLSIF_{S.shape[1]}_{A.shape[1]}.png')
            plt.show()

            plt.figure()
            sns.histplot(self.alpha.squeeze(1).detach().cpu().numpy(), kde=True, bins=30)
            plt.title(f"Distribution of uLSIF alpha Coefficients ,Target = {target_p_choice}")
            plt.savefig(f'./plots/alpha_uLSIF_{S.shape[1]}_{A.shape[1]}.png')
            plt.show()

        return self.alpha  # (n_beta, 1)

    def predict(self,
                S_new: torch.Tensor,
                A_new: torch.Tensor) -> torch.Tensor:
        if self.alpha is None:
            raise RuntimeError("Call fit() first.")
        if S_new.ndim != 2 or A_new.ndim != 2:
            raise ValueError("S_new and A_new must be 2D tensors.")
        X_new = torch.cat([S_new, A_new], dim=1).to(device=self._X_beta.device, dtype=self._X_beta.dtype)
        K_new = self.kernel_func(self._X_beta, X_new, **self.kernel_kwargs)          # (n_beta, n_new)
        return (K_new.T @ self.alpha).reshape(-1, 1)                                  # (n_new, 1)
##===================================
    def compute_ess(self,
                    S: torch.Tensor,
                    A: torch.Tensor) -> float:
        """
        Compute the effective sample size (ESS) of the importance weights
        estimated by ULSIF fit, for the given (S,A) batch.
        Returns a Python float.
        """
        # 1) get unnormalized weights w_i = eta_hat(s_i,a_i)
        eta = self.predict(S, A).reshape(-1)   # (n,)

        # 2) stabilize & normalize
        #    Subtract max for numeric stability when exponentiating,
        #    but here eta is already positive direct ratio estimate,
        #    so we skip exp and just normalize directly.
        w = eta
        w_sum = torch.sum(w)
        if w_sum <= 0:
            raise RuntimeError("Sum of estimated weights is non-positive!")
        w_norm = w / w_sum                     # sum_i w_norm = 1

        # 3) ESS = 1 / sum_i w_norm[i]^2
        ess = 1.0 / torch.sum(w_norm * w_norm)

        return ess.item()