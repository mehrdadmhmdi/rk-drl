import os, glob
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from functools import partial
from .matern_kernel import matern_kernel
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import matplotlib.font_manager as fm
from scipy.interpolate import interp1d
import re


class RecoverAndPlot:
    """
    PLOTS toolkit:
      - recover_joint_beta      : solve ONCE for β_full on full d-D Z-grid
      - eval_joint_on_grid_2d   : evaluate 2-D projection using same β_full
      - marginals_from_beta     : 1-D projections using same β_full
      - recover_marginals       : convenience wrapper (computes β_full once, then 1-D)
      - recover_joint_2d        : convenience wrapper (computes β_full once, then 2-D)
      - compute_L2_distance     : L2(pdf_est || Parzen_Matern(Z_true)) per dim (no KDE)
      - plot_*                  : visualization helpers

    Assumes tensors you pass are already on the desired device (e.g., cuda:0).
    """

    def __init__(self, config):
        self.cfg = config
        self.true_kde_grids = {}

    # -------------------- helpers --------------------
    def _footer(self, extra: str = "") -> str:
        c = self.cfg

        base = (
            f"Behavioral Policy: {c['behavioral_policy']}  |  Target Policy: {c['target_policy']}\n"
            rf"Data id={c['data_ID']} | Matérn($\nu,\ell,\sigma$)=({c['nu']},{c['length_scale']},{np.array(c['sigma_k']).round(2)}) | "
            f"n={c['n_ids']}, T={c['n_timepoints']} | dims(S,R,A)=({c['state_dim']},{c['reward_dim']},{c['action_dim']})"
            f"\nlr={c['lr']:.1e} | Training Reg-$\\lambda$={c['lambda_reg']:.1e} | "
            f"FP_constraint={c['fixed_point_constraint']} | $FP_\\lambda$={c['FP_penalty_lambda']} | "
            f"Recovery Reg-$\\lambda$={c['lambda_rec']:.1e}"
            f"\nDensity G-Bandwidth={c.get('bandwidth', 'N/A')} | Z grid expand factor = {c.get('hull_expand_factor', 'N/A')}"
            f"\n(s*,a*)=({np.array(c['s_star']).round(3)}, {np.array(c['a_star']).round(3)})"
        )

        return base + (f" | {extra}" if extra else "")

    def _fname(self, plot_type: str) -> str:
        c = self.cfg
        return (f"{plot_type}_{c['job_id']}.png")

    @staticmethod
    def _trapz1(u: torch.Tensor, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Trapezoidal rule integrating u along axis `dim` using a 1D grid x.
        Works for any ndim(u) with x.shape == (u.size(dim),).
        """
        if x.ndim != 1:
            raise ValueError("x must be 1D for _trapz1.")
        if x.numel() != u.size(dim):
            raise ValueError(f"x length ({x.numel()}) must equal u.size(dim) ({u.size(dim)}).")

        dx = x[1:] - x[:-1]  # [L-1]
        sl0 = [slice(None)] * u.ndim; sl0[dim] = slice(0, -1)
        sl1 = [slice(None)] * u.ndim; sl1[dim] = slice(1, None)
        u0 = u[tuple(sl0)]
        u1 = u[tuple(sl1)]
        shape = [1] * u.ndim; shape[dim] = dx.numel()
        dxv = dx.view(shape)
        return (0.5 * (u0 + u1) * dxv).sum(dim=dim)

    # -------------------- single joint solve --------------------
    @staticmethod
    @torch.no_grad()
    def _beta_full(B, k_sa, phi, Z_grid, K_sa, *, method, nu, length_scale, sigma_k, lambda_reg):
        """
        Solve ONCE on the full d-D grid Z_grid to get β_full ∈ R^m.
        """
        if method == "song":
            return (B.T @ k_sa.view(-1)).contiguous()

        if method == "bellman":
            return (B.T @ phi.view(-1)).contiguous()

        if method == "Schuster":
            ## in our case L_Z = L_Y = L_ZY=Kgg as we have grid = y
            L = torch.linalg.cholesky(A)
            L_inv = torch.linalg.inv(L)
            A_inv_cholesky = torch.matmul(L_inv.T, L_inv)

            m        = Kgg.size(0)
            n        = K_sa.size(0)

            Kgg      = matern_kernel(Z_grid, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma_k)  # [m,m]
            Aux_Kgg  = torch.linalg.inv(torch.linalg.cholesky(Kgg +  lambda_reg * torch.eye(m, device=Z_grid.device, dtype=Z_grid.dtype)))
            Kgg_inv  = torch.matmul(Aux_Kgg.T, Aux_Kgg)

            Aux_K_sa = torch.linalg.inv(torch.linalg.cholesky(K_sa + n*lambda_reg * torch.eye(n, device=Z_grid.device, dtype=Z_grid.dtype)))
            K_sa_inv = torch.matmul(Aux_K_sa.T, Aux_K_sa)

            num      = (Kgg_inv**2)* Kgg * K_sa_inv * k_sa
            OP       = num/(m**2)
            return OP*Kgg

        raise ValueError(f"unknown method: {method}")

    @torch.no_grad()
    def recover_joint_beta(self, B, k_sa, Z_grid, phi,K_sa, *, nu, length_scale, sigma_k, method, lambda_reg):
        """
        Public one-liner: compute β_full once on the FULL d-D grid.
        Returns β_full weights (simplex-normalized) and Z_grid.
        """
        beta = self._beta_full(B, k_sa, phi, Z_grid,K_sa,
                               method=method, nu=nu, length_scale=length_scale,
                               sigma_k=sigma_k, lambda_reg=lambda_reg)
        self.save_csv(beta.detach().cpu().numpy(), f"weights_{self.cfg['data_ID']}.csv", base_path="./mu")
        return beta, Z_grid

    # -------------------- evaluation (no re-solves) --------------------
    @torch.no_grad()
    def marginals_from_beta(self, beta_full, Z_grid_full, reward_dim, *,
                            nu, length_scale, sigma_k, lambda_rec, bandwidth=0.5, n_grid=400, margin_factor=0.35):
        """
        1-D projections from the SAME β_full.
        Exact if kernel factorizes (product kernel); otherwise a diagnostic projection.
        Returns F[n_grid, reward_dim], grids{j}.
        """
        import math
        device, dtype = Z_grid_full.device, Z_grid_full.dtype
        n = Z_grid_full.shape[0]
        F_cols, grids = [], {}
        for j in range(reward_dim):
            Zj = Z_grid_full[:, [j]]
            zmin, zmax = Zj.min().item(), Zj.max().item()
            r = (zmax - zmin) * margin_factor
            u = torch.linspace(zmin - r, zmax + r, n_grid, device=device, dtype=dtype).unsqueeze(1)

            KZ = matern_kernel(Zj, Zj, nu=nu, length_scale=length_scale, sigma=sigma_k)  # [n,n]
            KZ = 0.5 * (KZ + KZ.T)  # symmetrize for numerical stability
            M = torch.linalg.cholesky(KZ + (n * lambda_rec) * torch.eye(n, device=device, dtype=dtype))

            # Gaussian-smoothing kernel for pdf
            z = Zj.view(-1)  # [n]
            h = bandwidth
            T = (z - u) / h  # [n_grid,n]
            g = torch.exp(-0.5 * T ** 2) / (math.sqrt(2.0 * math.pi) * h)  # [n_grid,n]

            # c(U) = (KZ + nλI)^(-1) g(U)
            C = torch.cholesky_solve(g.T, M)  # [n, n_grid]

            # f(U) = ω^T KZ c(U)
            omega = beta_full.view(-1)  # [n]
            Kw = KZ @ omega  # [n]
            f = (Kw.unsqueeze(0) @ C).view(-1)  # [n_grid]

            # 1-D normalization
            f = f.clamp_min(0)
            mass = self._trapz1(f, u.view(-1), dim=0)
            f = f / (mass + torch.finfo(dtype).eps)

            F_cols.append(f)
            grids[j] = u.view(-1).detach().cpu().numpy()
        F = torch.stack(F_cols, dim=1)  # [n_grid, reward_dim]
        return F, grids

    # -------------------- metrics (no KDE) --------------------
    @staticmethod
    @torch.no_grad()
    def compute_L2_distance(pdf, Z_true, grid, *, nu, length_scale, sigma_k):
        """
        L2 distance between estimated pdf on 'grid' and a Parzen estimate
        built from Z_true using the same Matérn kernel (no KDE).
        Works for 1D (scalar) or per-dimension marginals.
        """
        device, dtype = Z_true.device, Z_true.dtype

        p_est = torch.as_tensor(pdf, dtype=dtype, device=device)
        if p_est.ndim == 1: p_est = p_est[:, None]

        # grid handling
        if isinstance(grid, dict):  # {j: 1D np/tensor}
            Gcols = [torch.as_tensor(grid[j], dtype=dtype, device=device)[:, None] for j in range(len(grid))]
            G = torch.cat(Gcols, dim=1)
        else:
            G = torch.as_tensor(grid, dtype=dtype, device=device)
            if G.ndim == 1: G = G[:, None]

        m, dG = G.shape
        dZ = Z_true.shape[1]
        d_use = min(p_est.shape[1], dG, dZ)

        # normalize estimated columns
        p_est = p_est[:, :d_use]
        p_est = p_est / (p_est.sum(dim=0, keepdim=True) + 1e-12)

        # choose chunk over Z to avoid forming full m×N kernel
        N = Z_true.shape[0]
        elem = torch.tensor([], dtype=dtype, device=device).element_size()
        try:
            free = torch.cuda.mem_get_info(device.index)[0] if device.type == 'cuda' else 64 * (1024 ** 3)
        except Exception:
            free = 2 * (1024 ** 3)
        budget = int(0.30 * free)
        b = max(1, min(N, budget // max(m * elem * 6, 1)))  # conservative
        if b == 0: b = 1

        def parzen_1d(Gj, Zj):
            acc = torch.zeros(m, dtype=dtype, device=device)
            for s in range(0, N, b):
                e = min(s + b, N)
                Kblk = matern_kernel(Gj, Zj[s:e], nu=nu, length_scale=length_scale, sigma=sigma_k)  # [m, e-s]
                acc += Kblk.sum(dim=1)
                del Kblk
            p = acc / float(N)
            p = p / (p.sum() + 1e-12)
            return p

        if d_use == 1:
            p_true = parzen_1d(G[:, :1], Z_true[:, :1])
            return float(torch.linalg.vector_norm(p_est[:, 0] - p_true))

        L2s = []
        for j in range(d_use):
            pj_true = parzen_1d(G[:, j:j + 1], Z_true[:, j:j + 1])
            L2s.append(float(torch.linalg.vector_norm(p_est[:, j] - pj_true)))
        return L2s

    # -------------------- plots (NumPy/Matplotlib/Seaborn) --------------------
    def plot_densities(self, fz, Z_true, r, Z, grid_dict, outdir="./plots/ind_plots"):
        d = self.cfg['reward_dim']
        fig, axes = plt.subplots(1, d, figsize=(6 * d, 6))
        axes = np.atleast_1d(axes)

        fz_np = fz.detach().cpu().numpy() if hasattr(fz, "detach") else np.asarray(fz)
        Z_np  = Z.detach().cpu().numpy() if hasattr(Z, "detach") else np.asarray(Z)
        Zt_np = Z_true.detach().cpu().numpy() if hasattr(Z_true, "detach") else np.asarray(Z_true)
        r_np  = r.detach().cpu().numpy() if hasattr(r, "detach") else np.asarray(r)

        for j, ax in enumerate(axes):
            ax.plot(np.asarray(grid_dict[j]), fz_np[:, j], color="#FF5F05", lw=2, label="Estimated Marginal Density")
            sns.kdeplot(Zt_np[:, j], ax=ax, fill=True, color="#FCB316", alpha=0.8, label="MonteCarlo True Z")
            sns.histplot(r_np[:, j], ax=ax, bins=40, stat="density", color="#007E8E", alpha=0.6,
                         label="Observed Rewards")
            for z in Z_np[:, j]:
                ax.axvline(z, ymin=0, ymax=0.03, color="#FF5F05", lw=0.6)
            ax.set_title(f"Z-dim {j + 1}")
            ax.set_xlabel("Z");
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(0.5, 0.04, self._footer(), ha="center", fontsize=11, wrap=True)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/{self._fname('marginal_fZ')}", dpi=600)
        plt.close()

    def plot_bellman_error(self, hist_be, outdir="./plots/ind_plots"):
        plt.figure(figsize=(7, 5))
        plt.plot(np.asarray(hist_be), color="#FF5F05")
        plt.xlabel("Iteration"); plt.ylabel("Log ‖Bellman Error‖")
        plt.title("Log Bellman Error (ADAM)")
        plt.grid(alpha=0.3, linestyle="--")
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(0.5, 0.04, self._footer(f"Final log BE: {hist_be[-1]:.6e} | Final BE: {math.exp(hist_be[-1]):.6e} "), ha="center", fontsize=legendsize, wrap=True)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/{self._fname('BellmanError_ADAM')}", dpi=600)
        plt.close()

    def plot_total_loss(self, hist_obj, outdir="./plots/ind_plots"):
        plt.figure(figsize=(7, 5))
        plt.plot(np.asarray(hist_obj), color="#FF5F05")
        plt.xlabel("Iteration"); plt.ylabel("Log Loss (objective + penalty)")
        plt.title("Log Total Loss (ADAM)",fontsize=12)
        plt.grid(alpha=0.3, linestyle="--")
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(0.5, 0.04, self._footer(f"Final log loss: {hist_obj[-1]:.6e} | Final loss: {math.exp(hist_obj[-1]):.6e} "), ha="center", fontsize=10, wrap=True)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/{self._fname('Loss_ADAM')}", dpi=600)
        plt.close()

    # -------------------- logging / csv --------------------
    @staticmethod
    def save_csv(data, filename, base_path="./LOSSvalues"):
        os.makedirs(base_path, exist_ok=True)
        arr  = np.asarray(data)
        path = f"{base_path}/{filename}"
        np.savetxt(path, arr, delimiter=",", fmt="%.8e")
        print(f"Saved {path} shape={arr.shape}")

    ##========================================
    ##  Mean Embedding ##
    ##========================================
    # ========== compute once on full grid, save CSV, return cache ==========
    @torch.no_grad()
    def mean_embeddings_full(self, beta, Z_grid, Z_true, *, nu, length_scale, sigma_k,
                             max_samples: int | None = None,
                             block_bytes: int | None = None):
        """
        Computes μ̂ and μ_true on the given full d-D grid (no plotting).
        Saves CSV once and returns a cache dict for reuse by plotting functions.
        """
        # optional uniform subset of Z_true
        if max_samples is not None and Z_true.shape[0] > max_samples:
            idx = torch.randperm(Z_true.shape[0], device=Z_true.device)[:max_samples]
            Z_use = Z_true[idx]
        else:
            Z_use = Z_true

        # μ̂(z) = K(Zg,Zg) β
        Kgg = matern_kernel(Z_grid, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma_k)
        mu_hat = (Kgg @ beta.view(-1)).contiguous()

        # μ_true(z) = E[k(z, Z)] streamed
        m, N = Z_grid.shape[0], Z_use.shape[0]
        dev, dt = Z_grid.device, Z_grid.dtype
        elem = torch.tensor([], dtype=dt, device=dev).element_size()
        try:
            free = torch.cuda.mem_get_info(dev.index)[0] if (dev.type == "cuda") else 4 * (1024 ** 3)
        except Exception:
            free = 2 * (1024 ** 3)
        budget = int(0.25 * free) if block_bytes is None else int(block_bytes)
        b = max(1, min(N, budget // max(m * elem, 1)))

        acc = torch.zeros(m, dtype=dt, device=dev)
        for s in range(0, N, b):
            e = min(N, s + b)
            Kblk = matern_kernel(Z_grid, Z_use[s:e], nu=nu, length_scale=length_scale, sigma=sigma_k)
            acc += Kblk.sum(dim=1)
            del Kblk
        mu_true = (acc / float(N)).contiguous()

        # CSV (full grid)
        self.save_csv(mu_hat.detach().cpu().numpy(), f"mu_hat_{self.cfg['data_ID']}.csv", base_path="./mu")
        self.save_csv(mu_true.detach().cpu().numpy(), f"mu_true_{self.cfg['data_ID']}.csv", base_path="./mu")

        return {
            "Z_grid": Z_grid, "beta": beta, "mu_hat": mu_hat, "mu_true": mu_true,
            "nu": nu, "length_scale": length_scale, "sigma_k": sigma_k
        }

    # ========== 1-D per-dim plot FROM CACHE (no recompute) =======================
    def plot_mu_per_dim(self, cache, outdir="./plots/ind_plots"):
        """
        Side-by-side smoothed μ̂ vs μ per dimension using the full-grid cache.
        """
        import numpy as np, os
        os.makedirs(outdir, exist_ok=True)

        Zg = cache["Z_grid"].detach().cpu().numpy()  # (m,d)
        mh = cache["mu_hat"].detach().cpu().numpy()  # (m,)
        mt = cache["mu_true"].detach().cpu().numpy()  # (m,)
        m, d = Zg.shape

        k = max(5, (m // 50) | 1)  # ~2% window, odd length
        ker = np.hanning(k);
        ker /= ker.sum()
        smooth = lambda y: np.convolve(y, ker, mode="same")

        fig, axes = plt.subplots(1, d, figsize=(5 * d, 4), sharey=True)
        axes = np.atleast_1d(axes)
        for j in range(d):
            x = Zg[:, j];
            idx = np.argsort(x)
            ax = axes[j]
            ax.plot(x[idx], smooth(mh[idx]), label=fr"$\hat \mu$-(grid,grid), dim {j+1}", lw=2)
            ax.plot(x[idx], smooth(mt[idx]), "--", label=fr"$\mu$-(grid,true), dim {j+1}", lw=1)
            ax.set_xlabel(f"Z value (dim {j+1})");
            ax.grid(alpha=0.3);
            ax.legend(fontsize=8)
            if j == 0: ax.set_ylabel("Mean embedding value")

        fig.suptitle("Estimated vs True Mean Embedding")
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(0.5, 0.02, self._footer(), ha="center", fontsize=10, wrap=True)
        fname = self._fname(f"mu_{self.cfg['data_ID']}")
        plt.savefig(f"{outdir}/{fname}", dpi=600);
        plt.close()

    # ========== 2-D joint slice (contour) + CSV (only slice work computed) =======
    @torch.no_grad()
    def mean_embedding_joint_2d_slice(self, cache, *, dims=(0, 1), ref="median",
                                      n1=120, n2=120, margin_factor=0.35,
                                      block_bytes=None):
        """
        2-D slice of mean embeddings μ̂ and μ_true on (dims=j,k) using the cache.
        Saves CSV for the slice as well. Returns (μ̂_2d, μ_true_2d, Xg, Yg).
        """
        beta, Z_grid, Z_true = cache["beta"], cache["Z_grid"], self._Z_true_ref
        nu, length_scale, sigma_k = cache["nu"], cache["length_scale"], cache["sigma_k"]

        device, dtype = Z_grid.device, Z_grid.dtype
        j, k = dims
        m, d = Z_grid.shape
        # stash Z_true in object once to avoid threading through args everywhere
        # (set via the orchestrator below)
        if Z_true is None:
            raise RuntimeError("Internal Z_true reference not set; call orchestrator first.")

        ref_vec = Z_grid.median(0).values if ref == "median" else Z_grid.mean(0)

        # 2D grid
        x_min, x_max = Z_grid[:, j].min().item(), Z_grid[:, j].max().item()
        y_min, y_max = Z_grid[:, k].min().item(), Z_grid[:, k].max().item()
        rx = (x_max - x_min) * margin_factor;
        ry = (y_max - y_min) * margin_factor
        x = torch.linspace(x_min - rx, x_max + rx, n1, device=device, dtype=dtype)
        y = torch.linspace(y_min - ry, y_max + ry, n2, device=device, dtype=dtype)
        Xg, Yg = torch.meshgrid(x, y, indexing="xy")

        # build query points for the slice
        Q = ref_vec.repeat(n1 * n2, 1)
        Q[:, j] = Xg.reshape(-1);
        Q[:, k] = Yg.reshape(-1)

        # μ̂ on slice
        Kqg = matern_kernel(Q, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma_k)
        mu_hat2 = (Kqg @ beta.view(-1)).view(n2, n1).contiguous()

        # μ_true on slice (stream over Z_true)
        N = Z_true.shape[0]
        elem = torch.tensor([], dtype=dtype, device=device).element_size()
        try:
            free = torch.cuda.mem_get_info(device.index)[0] if (device.type == "cuda") else 4 * (1024 ** 3)
        except Exception:
            free = 2 * (1024 ** 3)
        budget = int(0.25 * free) if block_bytes is None else int(block_bytes)
        b = max(1, min(N, budget // max((n1 * n2) * elem, 1)))

        acc = torch.zeros(n1 * n2, dtype=dtype, device=device)
        for s in range(0, N, b):
            e = min(N, s + b)
            acc += matern_kernel(Q, Z_true[s:e], nu=nu, length_scale=length_scale, sigma=sigma_k).sum(dim=1)
        mu_true2 = (acc / float(N)).view(n2, n1).contiguous()

        # CSV (slice)
        did = self.cfg['data_ID']
        self.save_csv(mu_hat2.detach().cpu().numpy(), f"mu2D_hat_{did}_dims{j}{k}.csv", base_path="./mu")
        self.save_csv(mu_true2.detach().cpu().numpy(), f"mu2D_true_{did}_dims{j}{k}.csv", base_path="./mu")

        return mu_hat2, mu_true2, Xg, Yg
    #======================================================
    def plot_mean_embedding_3d_slice(self, cache, *, dims=(0, 1), ref="median",
                                     n1=120, n2=120, margin_factor=0.35,
                                     block_bytes=None, outdir="./plots"):
        """
        Side-by-side 3D surfaces for μ̂ and μ_true on a 2-D slice (dims=j,k).
        Uses mean_embedding_joint_2d_slice to compute the slice, then plots.
        """
        import os, numpy as np, matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        os.makedirs(outdir, exist_ok=True)
        # ----- font controls (single source of truth) -----
        font_family = "Bitstream Charter"
        titlesize = 13
        labelsize = 12
        x_ticksize = 10
        y_ticksize = 10

        plt.rcParams.update({
            "font.family": font_family,
            "axes.titlesize": titlesize,
            "axes.labelsize": labelsize,
            "xtick.labelsize": x_ticksize,
            "ytick.labelsize": y_ticksize,
            "legend.fontsize": labelsize,
        })

        # compute the 2D slice (torch tensors)
        mu_hat2, mu_true2, Xg_t, Yg_t = self.mean_embedding_joint_2d_slice(
            cache, dims=dims, ref=ref, n1=n1, n2=n2, margin_factor=margin_factor, block_bytes=block_bytes
        )

        # to numpy
        Xg = Xg_t.detach().cpu().numpy()
        Yg = Yg_t.detach().cpu().numpy()
        Z1 = mu_hat2.detach().cpu().numpy()
        Z2 = mu_true2.detach().cpu().numpy()

        # common color scale for fair comparison
        vmin = np.nanmin([Z1.min(), Z2.min()])
        vmax = np.nanmax([Z1.max(), Z2.max()])

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        # estimated μ̂
        surf1 = ax1.plot_surface(Xg, Yg, Z1, rcount=100, ccount=100,
                                 cmap="viridis", linewidth=0, antialiased=True,
                                 vmin=vmin, vmax=vmax)
        ax1.set_title(r"Estimated mean embedding $\hat{\mu}$")
        ax1.set_xlabel(f"Z[{dims[0]}]")
        ax1.set_ylabel(f"Z[{dims[1]}]")
        ax1.zaxis.set_rotate_label(False)  # disable auto rotation
        ax1.set_zlabel(r"$\hat{\mu}$", rotation=90, labelpad=8)
        # enforce font family on 3D tick labels too
        for lab in ax1.get_xticklabels() + ax1.get_yticklabels() + ax1.get_zticklabels():
            lab.set_fontname(font_family)

        # true μ
        surf2 = ax2.plot_surface(Xg, Yg, Z2, rcount=100, ccount=100,
                                 cmap="viridis", linewidth=0, antialiased=True,
                                 vmin=vmin, vmax=vmax)
        ax2.set_title(r"MonteCarlo True mean embedding $\mu$")
        ax2.set_xlabel(f"Z[{dims[0]+1}]")
        ax2.set_ylabel(f"Z[{dims[1]+1}]")
        ax2.zaxis.set_rotate_label(False)
        ax2.set_zlabel(r"$\mu$", rotation=90, labelpad=8)
        for lab in ax2.get_xticklabels() + ax2.get_yticklabels() + ax2.get_zticklabels():
            lab.set_fontname(font_family)

        # shared colorbar
        divider = make_axes_locatable(ax2)
        cax     = divider.append_axes("right", size="3%", pad=0.15, axes_class=plt.Axes)  # ← force 2D
        cbar    = fig.colorbar(surf2, cax=cax)
        cbar.ax.tick_params(labelsize=y_ticksize)
        cbar.set_label("value", fontsize=labelsize)
        plt.subplots_adjust(bottom=0.25)
        plt.figtext(0.5, 0.02, self._footer(), ha="center", fontsize=10, wrap=True)
        fname = self._fname(f"mu_3D_{self.cfg['data_ID']}")
        plt.savefig(os.path.join(outdir, fname), dpi=400)
        plt.close()
        print(f"Saved {os.path.join(outdir, fname)}")

    #======================================================
    def plot_mu_joint_contour(self, mu_hat_2d, mu_true_2d, Xg, Yg, dims=(0, 1), outdir="./plots/ind_plots"):
        import numpy as np, os
        os.makedirs(outdir, exist_ok=True)
        Z1, Z2 = Xg.detach().cpu().numpy(), Yg.detach().cpu().numpy()
        EH = mu_hat_2d.detach().cpu().numpy()
        ET = mu_true_2d.detach().cpu().numpy()
        vmin, vmax = float(min(EH.min(), ET.min())), float(max(EH.max(), ET.max()))
        levels = np.linspace(vmin, vmax, 15)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        c1 = ax1.contourf(Z1, Z2, EH, levels=levels, cmap="Blues");
        fig.colorbar(c1, ax=ax1)
        ax1.set(title=fr"$\hat\mu$ contour", xlabel=f"Z[{dims[0]+1}]", ylabel=f"Z[{dims[1]+1}]")
        c2 = ax2.contourf(Z1, Z2, ET, levels=levels, cmap="Oranges");
        fig.colorbar(c2, ax=ax2)
        ax2.set(title=fr"$\mu$ true contour", xlabel=f"Z[{dims[0]+1}]", ylabel=f"Z[{dims[1]+1}]")
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(0.5, 0.02, self._footer(), ha="center", fontsize=10, wrap=True)
        fname = self._fname(f"mu2D_{self.cfg['data_ID']}")
        plt.savefig(f"{outdir}/{fname}", dpi=600);
        plt.close()

    # ========== Orchestrator (one call from your script) =========================
    def mean_embedding_all(self, beta, Z_grid, Z_true, *,  # single entry point
                           nu, length_scale, sigma_k,
                           max_samples=None, block_bytes=None,
                           do_joint_dims=(0, 1), n1=120, n2=120, margin_factor=0.35,
                           outdir="./plots/ind_plots"):
        """
        - Computes μ̂ & μ_true on full grid once (CSV) and plots per-dim.
        - Computes a 2-D slice (CSV) and plots contours.
        """
        # compute full-grid once
        cache = self.mean_embeddings_full(
            beta, Z_grid, Z_true,
            nu=nu, length_scale=length_scale, sigma_k=sigma_k,
            max_samples=max_samples, block_bytes=block_bytes
        )
        # make Z_true accessible to the slice method without recomputing
        self._Z_true_ref = Z_true

        # per-dim plot (uses cache; no recompute)
        self.plot_mu_per_dim(cache, outdir=outdir)

        # joint 2-D slice (only slice work computed)
        mu2d_hat, mu2d_true, Xg, Yg = self.mean_embedding_joint_2d_slice(
            cache, dims=do_joint_dims, n1=n1, n2=n2, margin_factor=margin_factor
        )
        self.plot_mu_joint_contour(mu2d_hat, mu2d_true, Xg, Yg, dims=do_joint_dims, outdir=outdir)
        self.plot_mean_embedding_3d_slice(cache,  dims=do_joint_dims, ref="median",
                                          n1=n1, n2=n2, margin_factor=margin_factor,block_bytes=None, outdir=outdir)
        return cache, (mu2d_hat, mu2d_true, Xg, Yg)
    ##==================================================


    # ─────────────────
    # All runs
    # ─────────────────
    def plot_all_mu(self, data_dir="./mu", outdir="./plots", n_bins=20):
        """
        One-page summary (2x3):
          [A] mean±SD ribbons of μ̂ and μ (over grid index)
          [B] Empirical Cumulative Distribution Function (ECDF) of |μ̂-μ| pooled
          [C] per-run boxplots: RMSE & sup-norm
          [D] Quantile calibration with 95% CI + Deming slope/intercept
          [E] Bland–Altman Difference Plot (pooled) with param & nonparam LOA + trend line
          [F] Text panel: aggregate numbers
        """
        import os, numpy as np, matplotlib.pyplot as plt
        os.makedirs(outdir, exist_ok=True)
        data_dir = os.fspath(data_dir)

        # ----- font controls (single source of truth) -----
        font_family = "Bitstream Charter"
        titlesize   = 16
        labelsize   = 14
        x_ticksize  = 14
        y_ticksize  = 12
        legendsize  = 10

        # set global defaults
        plt.rcParams.update({
            "font.family": font_family,
            "axes.titlesize": titlesize,
            "axes.labelsize": labelsize,
            "xtick.labelsize": x_ticksize,
            "ytick.labelsize": y_ticksize,
            "legend.fontsize": legendsize,
        })



        # ---- load all runs
        def extract_index(fname, pattern):
            match = re.search(pattern, os.path.basename(fname))
            return int(match.group(1)) if match else None

        # Collect all matching files
        hat_files  = glob.glob(os.path.join(data_dir, "mu_hat_*.csv"))
        true_files = glob.glob(os.path.join(data_dir, "mu_true_*.csv"))

        hat_by_idx  = {extract_index(f, r"mu_hat_(\d+)\.csv"): f for f in hat_files}
        true_by_idx = {extract_index(f, r"mu_true_(\d+)\.csv"): f for f in true_files}

        # Intersect indices
        matched_indices = sorted(set(hat_by_idx) & set(true_by_idx))

        if not matched_indices:
            print(f"No matching mu_hat_*.csv / mu_true_*.csv in {data_dir}")
            return

        # Load aligned files
        H, T = [], []
        for i in matched_indices:
            fh, ft = hat_by_idx[i], true_by_idx[i]
            H.append(np.loadtxt(fh, delimiter=",").reshape(-1))
            T.append(np.loadtxt(ft, delimiter=",").reshape(-1))

        H    = np.vstack(H)
        T    = np.vstack(T)
        R, m = H.shape
        xidx = np.arange(m)

        # ---- per-run metrics
        rmse, supn, nrmse, mae, medae, p90, bias = [], [], [], [], [], [], []
        for r in range(R):
            e = H[r] - T[r]
            rmse.append(np.sqrt(np.mean(e ** 2)))
            supn.append(np.max(np.abs(e)))
            nrmse.append(rmse[-1] / (np.std(T[r], ddof=1) + 1e-12))
            mae.append(np.mean(np.abs(e)))
            medae.append(np.median(np.abs(e)))
            p90.append(np.quantile(np.abs(e), 0.90))
            bias.append(e.mean())

        # ---- across-run ribbons
        Hm, Hsd = H.mean(axis=0), H.std(axis=0, ddof=1)
        Tm, Tsd = T.mean(axis=0), T.std(axis=0, ddof=1)

        # ---- pooled abs error ECDF
        abs_err = np.abs(H - T).ravel()
        ecdf_x  = np.sort(abs_err);
        ecdf_y  = np.arange(1, ecdf_x.size + 1) / ecdf_x.size

        # ---- quantile calibration (with 95% CI) & Deming regression
        x   = T.ravel();
        y   = H.ravel()
        q   = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        idx = np.digitize(x, q[1:-1], right=True)
        xb, yb, ylo, yhi, nb = [], [], [], [], []
        for b in range(n_bins):
            mask = (idx == b)
            if not np.any(mask):
                xb  += [np.nan];
                yb  += [np.nan];
                ylo += [np.nan];
                yhi += [np.nan];
                nb  += [0];
                continue
            xm, ym = x[mask].mean(), y[mask].mean()
            s      = y[mask].std(ddof=1);
            n      = mask.sum()
            se     = s / np.sqrt(max(n, 1))
            xb    += [xm];
            yb    += [ym];
            ylo   += [ym - 1.96 * se];
            yhi   += [ym + 1.96 * se];
            nb    += [int(n)]
        xbar, ybar = x.mean(), y.mean()
        sx2, sy2   = np.var(x, ddof=1), np.var(y, ddof=1)
        sxy        = np.cov(x, y, ddof=1)[0, 1]
        num        = (sy2 - sx2) + np.sqrt((sy2 - sx2) ** 2 + 4 * sxy ** 2)
        dem_slope  = num / (2 * sxy) if sxy != 0 else np.nan
        dem_inter  = ybar - dem_slope * xbar if np.isfinite(dem_slope) else np.nan

        # ---- Bland–Altman (pooled)
        avg            = 0.5 * (H + T);
        diff           = (H - T)
        avg, diff      = avg.ravel(), diff.ravel()
        mbias          = diff.mean();
        s              = diff.std(ddof=1)
        loa_lo, loa_hi = mbias - 1.96 * s, mbias + 1.96 * s
        q_lo, q_hi     = np.quantile(diff, [0.025, 0.975])
        X              = np.vstack([np.ones_like(avg), avg]).T
        b              = np.linalg.lstsq(X, diff, rcond=None)[0]  # intercept, slope
        xline          = np.array([avg.min(), avg.max()]);
        yline          = b[0] + b[1] * xline

        # ---- figure
        fig, axs = plt.subplots(3, 2, figsize=(12, 14))
        (axA, axB, axC, axD, axE, axF) = axs.ravel()

        # [A] ribbons
        axA.plot(xidx, Hm, lw=2, color="#FF5F05", label=r"$\bar{\hat\mu}$")
        axA.fill_between(xidx, Hm - Hsd, Hm + Hsd, color="#FF5F05", alpha=0.2)
        axA.plot(xidx, Tm, lw=2, color="k", label=r"$\bar{\mu}$")
        axA.fill_between(xidx, Tm - 1.96*Tsd, Tm + 1.96*Tsd, color="k", alpha=0.15)
        axA.set_title("(A) Mean ±1.96 SD Across Runs", fontname=font_family, fontsize=titlesize)
        axA.set_xlabel("Index on Z grid", fontname=font_family, fontsize=labelsize)
        axA.set_ylabel("Mean embedding", fontname=font_family, fontsize=labelsize)
        axA.grid(alpha=0.3)
        axA.legend(prop={"family": font_family, "size": labelsize})
        axA.tick_params(axis="both", labelsize=x_ticksize)
        for t in axA.get_xticklabels() + axA.get_yticklabels():
            t.set_fontfamily(font_family)
        # [B] Quantile calibration + CI
        axB.errorbar(xb, yb, yerr=[np.array(yb) - np.array(ylo), np.array(yhi) - np.array(yb)],
                     fmt='o-', lw=2, capsize=3, label="binned means ±95% CI", color='#FF5F05')
        lo, hi = np.nanmin(xb), np.nanmax(xb)
        axB.plot([lo, hi], [lo, hi], 'k--', label="ideal")
        axB.set_title("(B) Quantile Calibration (pooled)", fontname=font_family, fontsize=titlesize)
        axB.set_xlabel("True mean embedding (bin mean)", fontname=font_family, fontsize=labelsize)
        axB.set_ylabel("Estimated mean embedding (bin mean)", fontname=font_family, fontsize=labelsize)
        legB = axB.legend(title=f"Deming slope={dem_slope:.3f}, int={dem_inter:.3f}",
                          frameon=True, prop={"family": font_family, "size": labelsize})
        if legB and legB.get_title():
            legB.get_title().set_fontfamily(font_family)
            legB.get_title().set_fontsize(labelsize)
        axB.tick_params(axis="both", labelsize=y_ticksize)
        for t in axB.get_xticklabels() + axB.get_yticklabels():
            t.set_fontfamily(font_family)


        # [C] per-run boxplots
        axC.boxplot([rmse, supn], showmeans=True, labels=["RMSE", "sup-norm error"])
        axC.set_title("(C) Per-run Error Summaries", fontname=font_family, fontsize=titlesize)
        axC.set_xlabel("", fontname=font_family, fontsize=labelsize)
        axC.set_ylabel("", fontname=font_family, fontsize=labelsize)
        axC.grid(alpha=0.3)
        axC.tick_params(axis="both", labelsize=x_ticksize)
        for t in axC.get_xticklabels() + axC.get_yticklabels():
            t.set_fontfamily(font_family)

        # [D] ECDF
        axD.plot(ecdf_x, ecdf_y, lw=2)
        axD.set_title(r"(D) Empirical CDF of $|\hat\mu-\mu|$ (pooled)", fontname=font_family, fontsize=titlesize)
        axD.set_xlabel(r"$|\hat\mu-\mu|$", fontname=font_family, fontsize=labelsize)
        axD.set_ylabel("ECDF", fontname=font_family, fontsize=labelsize)
        axD.grid(alpha=0.3)
        axD.tick_params(axis="both", labelsize=x_ticksize)
        for t in axD.get_xticklabels() + axD.get_yticklabels():
            t.set_fontfamily(font_family)

        # [E] Bland–Altman-Difference Plot
        hb = axE.hexbin(avg, diff, gridsize=60, bins='log', mincnt=1, cmap='YlOrRd')
        cb = fig.colorbar(hb, ax=axE)
        cb.set_label("log count", fontname=font_family, fontsize=labelsize)
        cb.ax.tick_params(labelsize=x_ticksize)
        for t in cb.ax.get_yticklabels():
            t.set_fontfamily(font_family)
        axE.axhline(mbias, color='blue', lw=2, label=f"mean bias={mbias:.4f}")
        axE.axhline(loa_lo, color='green', ls='--', label="±1.96·SD (param)")
        axE.axhline(loa_hi, color='green', ls='--')
        axE.axhline(q_lo, color='purple', ls=':', label="2.5–97.5% (nonparam)")
        axE.axhline(q_hi, color='purple', ls=':')
        axE.plot(xline, yline, 'k--', lw=1.6, label=f"trend slope={b[1]:.3f}")
        axE.set_title("(E) Difference-vs-Average Plot (pooled)", fontname=font_family, fontsize=titlesize)
        axE.set_xlabel(r"Average: $(\hat\mu+\mu)/2$", fontname=font_family, fontsize=labelsize)
        axE.set_ylabel(r"Difference: $\hat\mu-\mu$", fontname=font_family, fontsize=labelsize)
        axE.legend(loc="lower right", frameon=True, prop={"family": font_family, "size": legendsize})
        axE.tick_params(axis="both", labelsize=x_ticksize)
        for t in axE.get_xticklabels() + axE.get_yticklabels():
            t.set_fontfamily(font_family)

        # [F] text panel
        axF.axis("off")
        txt = (
            f"Runs: {R}\n"
            f"RMSE: {np.mean(rmse):.4f} ± {np.std(rmse, ddof=1):.4f}\n"
            f"sup-norm error: {np.mean(supn):.4f} ± {np.std(supn, ddof=1):.4f}\n"
            f"MAE: {np.mean(mae):.4f}   MedianAE: {np.mean(medae):.4f}   P90AE: {np.mean(p90):.4f}\n\n"
            f"Mean Difference: mean bias={mbias:.4f}, SD={s:.4f},\n"
            f"95% difference interval =[{loa_lo:.4f},{loa_hi:.4f}],\n"
            f"95% difference interval (empirical)=[{q_lo:.4f},{q_hi:.4f}]"
        )
        axF.text(0.02, 0.98, txt, va="top", ha="left",
                 fontname=font_family, fontsize=titlesize, wrap=True)

        plt.subplots_adjust(bottom=0.18, wspace=0.25, hspace=0.35)
        plt.figtext(0.5, 0.05, self._footer(), ha="center",
                    fontname=font_family, fontsize=titlesize, wrap=True)
        fname = self._fname("mu_summary_allin1")
        plt.savefig(f"{outdir}/{fname}", dpi=600)
        plt.close()

    # ─────────────────
    # single run
    # ─────────────────
    # ========== Error vs distance from mode (single run) ================
    def plot_error_vs_distance_from_mode(self, cache, dims=None, n_bins=15, outdir="./plots/ind_plots"):
        os.makedirs(outdir, exist_ok=True)
        Zg  = cache["Z_grid"].detach().cpu().numpy()
        mh  = cache["mu_hat"].detach().cpu().numpy().reshape(-1)
        mt  = cache["mu_true"].detach().cpu().numpy().reshape(-1)
        err = np.abs(mh - mt)

        # mode by argmax of μ_true on the grid
        z_mode = Zg[np.argmax(mt)]
        if dims is not None:
            Z_use = Zg[:, list(dims)]
            z_mode = z_mode[list(dims)]
        else:
            Z_use = Zg

        dist = np.linalg.norm(Z_use - z_mode, axis=1)
        bins = np.linspace(dist.min(), dist.max(), n_bins+1)
        which = np.digitize(dist, bins[1:-1], right=True)
        d_mid = 0.5*(bins[:-1]+bins[1:])
        err_bin = np.array([err[which==b].mean() if np.any(which==b) else np.nan for b in range(n_bins)])

        plt.figure(figsize=(7,5))
        plt.plot(d_mid, err_bin, marker='o')
        plt.xlabel("Distance to mode (L2)")
        plt.ylabel(r"$\mathbb{E}[\,|\hat\mu-\mu|\,]$ in bin")
        plt.title("Error vs distance from mode (single run)")
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(0.5, 0.02, self._footer(), ha="center", fontsize=10, wrap=True)
        plt.savefig(f"{outdir}/{self._fname('ErrVsMode')}", dpi=600); plt.close()

    # ==========  Bland–Altman (single run) ===============================
    def plot_bland_altman(self, cache, outdir="./plots/ind_plots"):
        os.makedirs(outdir, exist_ok=True)
        mh = cache["mu_hat"].detach().cpu().numpy().reshape(-1)
        mt = cache["mu_true"].detach().cpu().numpy().reshape(-1)
        avg = 0.5 * (mh + mt)
        diff = mh - mt
        m, s = diff.mean(), diff.std(ddof=1)
        loA, hiA = m - 1.96 * s, m + 1.96 * s

        plt.figure(figsize=(7,5))
        plt.hexbin(avg, diff, gridsize=45, bins='log')
        plt.axhline(m,   color='#FF5F05', ls='-',
                    label=f"mean bias = {m:.3g}")
        plt.axhline(loA, color='orange', ls='--',
                    label=f"±1.96·SD bands")
        plt.axhline(hiA, color='orange', ls='--')
        plt.xlabel(r"Average: $(\hat\mu+\mu)/2$")
        plt.ylabel(r"Difference: $\hat\mu-\mu$")
        plt.title("Bland–Altman (single run)")
        plt.legend(frameon=True, fontsize=9,loc='lower right')
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(0.5, 0.02, self._footer(), ha="center", fontsize=10, wrap=True)
        plt.savefig(f"{outdir}/{self._fname('BA_single')}", dpi=600); plt.close()

    # ========== Quantile calibration (single run) =======================
    def plot_quantile_calibration(self, cache, n_bins=20, outdir="./plots/ind_plots"):
        os.makedirs(outdir, exist_ok=True)
        mh = cache["mu_hat"].detach().cpu().numpy().reshape(-1)
        mt = cache["mu_true"].detach().cpu().numpy().reshape(-1)

        q = np.quantile(mt, np.linspace(0,1,n_bins+1))
        idx = np.digitize(mt, q[1:-1], right=True)
        mt_bin = np.array([mt[idx==b].mean() if np.any(idx==b) else np.nan for b in range(n_bins)])
        mh_bin = np.array([mh[idx==b].mean() if np.any(idx==b) else np.nan for b in range(n_bins)])

        plt.figure(figsize=(6,5))
        plt.plot(mt_bin, mh_bin, marker='o',color='#FF5F05',  lw=2, label="binned means")
        lo, hi = np.nanmin(mt_bin), np.nanmax(mt_bin)
        plt.plot([lo,hi],[lo,hi],'k--',label="ideal")
        plt.xlabel(r"True mean embedding (bin mean)")
        plt.ylabel(r"Estimated mean embedding (bin mean)")
        plt.title("Quantile calibration (single run)")
        plt.legend(frameon=False)
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(0.5, 0.02, self._footer(), ha="center", fontsize=10, wrap=True)
        plt.savefig(f"{outdir}/{self._fname('Quantile_single')}", dpi=600); plt.close()


    # ========== Operator check on 2D slice (needs R & γ) ===============
    @torch.no_grad()
    def plot_operator_check_2d(self, cache, R, gamma, dims=(0,1),
                               n1=80, n2=80, margin_factor=0.35,
                               max_rewards=2000, outdir="./plots/ind_plots"):
        """
        Compares μ̂ on a 2D slice to Tμ̂(z)=E_r[ E_{Z'~μ̂} k(z, r+γ Z') ] on the same slice.
        R: tensor [N,d] of reward samples (will be subsampled).
        """
        os.makedirs(outdir, exist_ok=True)
        beta, Zg = cache["beta"], cache["Z_grid"]
        nu, ell, sig = cache["nu"], cache["length_scale"], cache["sigma_k"]
        dev, dt = Zg.device, Zg.dtype

        # Build slice queries Q (reuse your helper)
        # (μ̂_2d from slice)
        mu2d_hat, _, Xg, Yg = self.mean_embedding_joint_2d_slice(
            cache, dims=dims, n1=n1, n2=n2, margin_factor=margin_factor
        )
        Q = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1).to(dev, dt)   # [Q,2] within selected dims
        # Expand Q back to d-D by fixing other coords at median
        d = Zg.shape[1]
        ref = Zg.median(0).values.clone()
        Qd = ref.repeat(Q.shape[0], 1)
        Qd[:, dims[0]] = Q[:,0]; Qd[:, dims[1]] = Q[:,1]                       # [Q,d]

        # Subsample rewards for tractability
        if R.shape[0] > max_rewards:
            ridx = torch.randperm(R.shape[0], device=R.device)[:max_rewards]
            Ruse = R[ridx].to(dev, dt)
        else:
            Ruse = R.to(dev, dt)

        # Tμ̂ on slice: average over rewards of K(Qd, γ Zg + r) β
        Q_out = torch.zeros(Qd.shape[0], dtype=dt, device=dev)
        m = Zg.shape[0]
        for r in Ruse:
            shifted = (gamma * Zg) + r  # [m,d]
            K = matern_kernel(Qd, shifted, nu=nu, length_scale=ell, sigma=sig)  # [Q, m]
            Q_out += K @ beta.view(-1)
            del K, shifted
        Tmu2d = (Q_out / float(Ruse.shape[0])).view_as(mu2d_hat).contiguous()

        # Plot μ̂, Tμ̂, and residual
        Xn, Yn = Xg.detach().cpu().numpy(), Yg.detach().cpu().numpy()
        A = mu2d_hat.detach().cpu().numpy()
        B = Tmu2d.detach().cpu().numpy()
        RZ = (A - B)

        vmin = min(A.min(), B.min()); vmax = max(A.max(), B.max())
        fig, axs = plt.subplots(1,3, figsize=(14,4))
        im0 = axs[0].pcolormesh(Xn, Yn, A, cmap="Blues", shading="auto"); fig.colorbar(im0, ax=axs[0]); axs[0].set(title="μ̂ (slice)")
        im1 = axs[1].pcolormesh(Xn, Yn, B, cmap="Greens", shading="auto"); fig.colorbar(im1, ax=axs[1]); axs[1].set(title="T μ̂ (slice)")
        im2 = axs[2].pcolormesh(Xn, Yn, np.abs(RZ), cmap="magma", shading="auto"); fig.colorbar(im2, ax=axs[2]); axs[2].set(title="|μ̂ − T μ̂|")
        for ax in axs: ax.set(xlabel=f"Z[{dims[0]+1}]", ylabel=f"Z[{dims[1]+1}]")
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(0.5, 0.02, self._footer(), ha="center", fontsize=10, wrap=True)
        plt.savefig(f"{outdir}/{self._fname('OperatorCheck2D')}", dpi=600); plt.close()

    # ==========  Error heatmap on 2D slice (single run) ==================
    @torch.no_grad()
    def plot_error_heatmap(self, cache, dims=(0,1), n1=120, n2=120, margin_factor=0.35,
                           outdir="./plots/ind_plots"):
        os.makedirs(outdir, exist_ok=True)
        mu2d_hat, mu2d_true, Xg, Yg = self.mean_embedding_joint_2d_slice(
            cache, dims=dims, n1=n1, n2=n2, margin_factor=margin_factor
        )
        Delta = torch.abs(mu2d_hat - mu2d_true).detach().cpu().numpy()
        Xn, Yn = Xg.detach().cpu().numpy(), Yg.detach().cpu().numpy()

        plt.figure(figsize=(6.8,5))
        im = plt.pcolormesh(Xn, Yn, Delta, cmap="magma", shading="auto")
        plt.colorbar(im, label="absolute error")
        plt.xlabel(f"Z[{dims[0]+1}]"); plt.ylabel(f"Z[{dims[1]+1}]")
        plt.title("|μ̂ − μ| heatmap (2D slice)")
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(0.5, 0.02, self._footer(), ha="center", fontsize=10, wrap=True)
        plt.savefig(f"{outdir}/{self._fname('ErrHeatmap2D')}", dpi=600); plt.close()

    # ==========  table (all runs) ==================
    @torch.no_grad()
    def export_metrics_tables(self, data_dir="./mu", metrics_dir="./metrics",Zgrid_dir="./data", n_bins=20,
                              device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float64 ):
        """
        Writes CSVs for:
          - per_run_metrics.csv  (RMSE, NRMSE, sup-norm, MAE, MedianAE, P90AE, Bias)
          - aggregate_metrics.csv (mean±sd over runs)
          - bland_altman_stats.csv (mean bias, SD, param & nonparam LOA, trend)
          - calibration_bins.csv (bin_mean_true, bin_mean_est, 95% CI, n)
          - mean and variance, median
        """
        import os, csv, numpy as np
        os.makedirs(metrics_dir, exist_ok=True)
        data_dir = os.fspath(data_dir)

        H, T = [], []
        for i in range(100):
            fh = os.path.join(data_dir, f"mu_hat_{i}.csv")
            ft = os.path.join(data_dir, f"mu_true_{i}.csv")
            if os.path.exists(fh) and os.path.exists(ft):
                H.append(np.loadtxt(fh, delimiter=",").reshape(-1))
                T.append(np.loadtxt(ft, delimiter=",").reshape(-1))
        if not H:
            print("No mu_hat_*.csv / mu_true_*.csv found.");
            return
        H = np.vstack(H);
        T = np.vstack(T)
        R, m = H.shape

        # per-run metrics
        rows = []
        for r in range(R):
            e     = H[r] - T[r]
            rmse  = np.sqrt(np.mean(e ** 2))
            nrmse = rmse / (np.std(T[r], ddof=1) + 1e-12)
            supn  = np.max(np.abs(e))
            mae   = np.mean(np.abs(e))
            medae = np.median(np.abs(e))
            p90   = np.quantile(np.abs(e), 0.90)
            bias  = e.mean()
            rows.append([r, rmse, nrmse, supn, mae, medae, p90, bias])

        with open(os.path.join(metrics_dir, "per_run_metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_id", "RMSE", "NRMSE", "SupNorm", "MAE", "MedianAE", "P90AE", "MeanError(Bias)"])
            w.writerows(rows)

        # aggregate metrics
        A   = np.array(rows)[:, 1:]  # drop run_id
        agg = np.concatenate([A.mean(axis=0), A.std(axis=0, ddof=1)])
        with open(os.path.join(metrics_dir, "aggregate_metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["RMSE_mean", "NRMSE_mean", "SupNorm_mean", "MAE_mean", "MedianAE_mean", "P90AE_mean", "Bias_mean",
                 "RMSE_sd", "NRMSE_sd", "SupNorm_sd", "MAE_sd", "MedianAE_sd", "P90AE_sd", "Bias_sd"])
            w.writerow(list(agg))

        # pointwise bias & variability summaries (scalars for paper table)
        pbias = (H.mean(axis=0) - T.mean(axis=0))  # bias per grid index
        pstd  = H.std(axis=0, ddof=1)  # sd of μ̂ across runs per grid index
        with open(os.path.join(metrics_dir, "pointwise_summaries.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mean_abs_pointwise_bias", "mean_pointwise_sd"])
            w.writerow([float(np.mean(np.abs(pbias))), float(np.mean(pstd))])

        # Bland–Altman stats (pooled)
        avg      = 0.5 * (H + T);
        diff     = (H - T)
        avg      = avg.ravel();
        diff     = diff.ravel()
        m        = diff.mean();
        s        = diff.std(ddof=1)
        loA, hiA = m - 1.96 * s, m + 1.96 * s
        loQ, hiQ = np.quantile(diff, [0.025, 0.975])
        X        = np.vstack([np.ones_like(avg), avg]).T
        b        = np.linalg.lstsq(X, diff, rcond=None)[0]
        with open(os.path.join(metrics_dir, "bland_altman_stats.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mean_bias", "sd", "loa_low_param", "loa_high_param", "loa_low_np", "loa_high_np",
                        "trend_intercept", "trend_slope"])
            w.writerow([m, s, loA, hiA, loQ, hiQ, b[0], b[1]])

        # calibration bins (across all runs)
        x   = T.ravel();
        y   = H.ravel()
        q   = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        idx = np.digitize(x, q[1:-1], right=True)
        with open(os.path.join(metrics_dir, "calibration_bins.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bin", "true_mean", "est_mean", "est_ci_low", "est_ci_high", "n"])
            for b in range(n_bins):
                mask = (idx == b)
                if not np.any(mask):
                    w.writerow([b, np.nan, np.nan, np.nan, np.nan, 0]);
                    continue
                xm = x[mask].mean();
                ym = y[mask].mean()
                s  = y[mask].std(ddof=1);
                n  = mask.sum()
                se = s / np.sqrt(max(n, 1))
                w.writerow([b, xm, ym, ym - 1.96 * se, ym + 1.96 * se, int(n)])

        # Deming regression (pooled) — export slope & intercept
        xbar, ybar = x.mean(), y.mean()
        sx2,  sy2  = np.var(x, ddof=1), np.var(y, ddof=1)
        sxy        = np.cov(x, y, ddof=1)[0, 1]
        num        = (sy2 - sx2) + np.sqrt((sy2 - sx2)**2 + 4 * sxy**2)
        dem_slope  = num / (2 * sxy) if sxy != 0 else np.nan
        dem_inter  = ybar - dem_slope * xbar if np.isfinite(dem_slope) else np.nan

        with open(os.path.join(metrics_dir, "calibration_deming.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["deming_slope", "deming_intercept", "xbar", "ybar", "var_x", "var_y", "cov_xy"])
            w.writerow([dem_slope, dem_inter, xbar, ybar, sx2, sy2, sxy])



        # === Compute per-run mean and variance from weights and Z_grid ===
        # Extract suffix from filename using regex
        def extract_index(fname, pattern):
            match = re.search(pattern, os.path.basename(fname))
            return int(match.group(1)) if match else None

        # Gather available files
        weight_files = glob.glob(os.path.join(data_dir, "weights_*.csv"))
        zgrid_files  = glob.glob(os.path.join(Zgrid_dir, "Zgrid_*.pt"))

        # Map: index -> filepath
        weights_by_idx = {extract_index(f, r"weights_(\d+)\.csv"): f for f in weight_files}
        zgrids_by_idx  = {extract_index(f, r"Zgrid_(\d+)\.pt"): f for f in zgrid_files}

        # Intersect available indices
        matched_indices = sorted(set(weights_by_idx) & set(zgrids_by_idx))
        if not matched_indices:
            print("No matching weight/grid file pairs found.")
            return
        # Now loop over matched pairs only
        mu_list, var_list, med_list = [], [], []
        for idx in matched_indices:
            w_file = weights_by_idx[idx]
            z_file = zgrids_by_idx[idx]

            beta_np = np.ravel(np.loadtxt(w_file, delimiter=","))
            beta    = torch.as_tensor(beta_np, device=device, dtype=dtype).view(-1)
            Z_grid  = torch.load(z_file).to(device=device, dtype=dtype)
            if Z_grid.shape[0] == 0:
                continue

            d = Z_grid.shape[1]
            mu_run, var_run, med_run = [], [], []
            for j in range(d):
                idx_sort    = torch.argsort(Z_grid[:, j])
                z_sorted    = Z_grid[idx_sort].contiguous()
                beta_sorted = beta[idx_sort]
                cache       = {"Z_grid": z_sorted, "beta": beta_sorted}
                mu_j        = self.moment_k(cache, k=1, dims=[j]).item()
                center      = torch.zeros(d, device=z_sorted.device)
                center[j]   = mu_j
                var_j       = self.moment_k(cache, k=2, dims=[j], center=center).item()
                med_j = self.quantile_from_cdf_projection(cache, float(0.5), j, self.cfg['lambda_rec'], self.cfg['nu'],
                                                          self.cfg['length_scale'],self.cfg['sigma_k'])
                med_run.append(med_j)
                mu_run.append(mu_j)
                var_run.append(var_j)

            mu_list.append(mu_run)
            var_list.append(var_run)
            med_list.append(med_run)

        # === Export per-run mean and variance ===
        mu_arr  = np.vstack(mu_list)
        var_arr = np.vstack(var_list)
        med_arr = np.vstack(med_list)
        np.savetxt(os.path.join(metrics_dir, "per_run_mean.csv"), mu_arr, delimiter=",")
        np.savetxt(os.path.join(metrics_dir, "per_run_variance.csv"), var_arr, delimiter=",")
        np.savetxt(os.path.join(metrics_dir, "per_run_median.csv"), med_arr, delimiter=",")

        # Export aggregate stats
        mu_mean, mu_sd = mu_arr.mean(axis=0), mu_arr.std(axis=0, ddof=1)
        var_mean, var_sd = var_arr.mean(axis=0), var_arr.std(axis=0, ddof=1)

        with open(os.path.join(metrics_dir, "mean_variance_summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            header = []
            for j in range(mu_mean.shape[0]):
                header += [f"mean_dim{j + 1}", f"mean_sd_dim{j + 1}", f"var_dim{j + 1}", f"var_sd_dim{j + 1}"]
            row = []
            for j in range(mu_mean.shape[0]):
                row += [mu_mean[j], mu_sd[j], var_mean[j], var_sd[j]]
            w.writerow(header)
            w.writerow(row)


        print(f"Saved CSV tables to {metrics_dir}")

    # ==========  Statistics (projection-based CDF)  ==================
    # ---------- helpers (instance methods) ----------
    @torch.no_grad()
    def _symmetrize(self, K: torch.Tensor) -> torch.Tensor:
        return 0.5 * (K + K.transpose(-1, -2))

    @torch.no_grad()
    def isotonic_project01(self, t_grid: torch.Tensor, F_vals: torch.Tensor) -> torch.Tensor:
        """Project to the nearest nondecreasing sequence in [0,1] (PAVA)."""
        y = F_vals.clone();
        w = torch.ones_like(y);
        n = int(y.numel());
        i = 0
        while i < n - 1:
            if y[i] <= y[i + 1] + 1e-12: i += 1; continue
            j = i
            while j >= 0 and y[j] > y[j + 1] + 1e-12:
                num = w[j] * y[j] + w[j + 1] * y[j + 1];
                den = w[j] + w[j + 1]
                val = num / den;
                y[j] = y[j + 1] = val;
                w[j] = w[j + 1] = den;
                j -= 1
            i += 1
        return torch.clamp(y, 0.0, 1.0)

    @torch.no_grad()
    def _prepare_cdf_projection(self, cache: dict, dim: int,
                                lambda_rec: float, nu: float, length_scale: float, sigma: float):
        """
        Precompute: v = (K + mλI)^{-1} K β  and prefix sums over z[:,dim] for CDF queries.
        Then F̂(t) = sum_{j : z_j<=t} v_j   (isotonic clamp applied later).
        """
        Zg   = cache["Z_grid"]  # (m,d)
        beta = cache["beta"].view(-1)  # (m,)
        device, dtype = Zg.device, Zg.dtype
        m, d = Zg.shape
        assert 0 <= dim < d

        K        = matern_kernel(Zg, Zg, nu=nu, length_scale=length_scale, sigma=sigma)
        I        = torch.eye(m, device=device, dtype=dtype)
        A        = K + (m * lambda_rec) * I   #A = K + m *lambda_rec*I
        L        = torch.linalg.cholesky(A)
        v        = torch.cholesky_solve(K @ beta[:, None], L).view(-1)  # (m,)  v=A^{-1}(K*beta)
        v        = v / (v.sum() + 1e-12)  # — normalize the v‐weights so that sum_j v_j = 1
        z        = Zg[:, dim]
        idx      = torch.argsort(z)
        z_sorted = z[idx].contiguous()
        v_sorted = v[idx].contiguous()
        prefix_v = torch.cumsum(v_sorted, dim=0)
        cache["_cdf_proj_cache"] = {
            "dim": dim, "lambda_rec": float(lambda_rec),
            "nu": float(nu), "length_scale": float(length_scale), "sigma": float(sigma),
            "z_sorted": z_sorted, "prefix_v": prefix_v
        }
        return cache["_cdf_proj_cache"]

    @torch.no_grad()
    def cdf_from_embedding_projection(self, cache: dict, t, dim: int,
                                      lambda_rec: float, nu: float, length_scale: float, sigma: float):
        """Return a proper CDF in [0,1], with a single normalization based on RAW cumulative endpoints."""
        Zg            = cache["Z_grid"];
        device, dtype = Zg.device, Zg.dtype
        t             = torch.as_tensor(t, device=device, dtype=dtype).view(-1)
        prep          = cache.get("_cdf_proj_cache")

        def _same(p):
            return (p["dim"] == dim and abs(p["lambda_rec"] - lambda_rec) < 1e-14
                    and abs(p["nu"] - nu) < 1e-12 and abs(p["length_scale"] - length_scale) < 1e-12
                    and abs(p["sigma"] - sigma) < 1e-12)

        if prep is None or not _same(prep):
            prep = self._prepare_cdf_projection(cache, dim, lambda_rec, nu, length_scale, sigma)

        z_sorted, prefix_v = prep["z_sorted"], prep["prefix_v"]  # :contentReference[oaicite:0]{index=0}
        # --- RAW endpoints for normalization (no isotonic, no clamp)
        F_lo_raw = float(prefix_v[0].item())
        F_hi_raw = float(prefix_v[-1].item())

        # raw prefix-sum CDF at query t
        idx = torch.searchsorted(z_sorted, t, right=True)
        Fq_raw = torch.where(idx > 0, prefix_v[idx - 1], torch.full_like(t, F_lo_raw))
        Fq_raw = torch.where(idx >= z_sorted.numel(), torch.full_like(t, F_hi_raw), Fq_raw)
        return Fq_raw

    @torch.no_grad()
    def quantile_from_cdf_projection(self, cache: dict, tau: float, dim: int,
                                     lambda_rec: float, nu: float, length_scale: float, sigma: float,
                                     t_lo=None, t_hi=None, max_iter: int = 60, tol: float = 1e-6):
        Zg = cache["Z_grid"];
        z = Zg[:, dim]
        lo = float(z.min().item()) if t_lo is None else float(t_lo)
        hi = float(z.max().item()) if t_hi is None else float(t_hi)

        tau = float(np.clip(tau, 0.0, 1.0))
        lo_, hi_ = lo, hi
        for _ in range(max_iter):
            mid = 0.5 * (lo_ + hi_)
            Fm = float(self.cdf_from_embedding_projection(cache, mid, dim,lambda_rec, nu, length_scale, sigma)[0])
            if Fm >= tau:
                hi_ = mid
            else:
                lo_ = mid
            if abs(hi_ - lo_) <= tol * (1 + abs(mid)): break
        return 0.5 * (lo_ + hi_)

    @torch.no_grad()
    def moment_k(self, cache, k, dims=None, center=None):
        """E[(Z_j-center_j)^k] under μ̂ using weights β on Z_grid (unchanged)."""
        Zg   = cache["Z_grid"];
        beta = cache["beta"].view(-1);
        # beta = beta / beta.sum()
        d    = Zg.shape[1]
        if center is None: center = torch.zeros(d, device=Zg.device, dtype=Zg.dtype)
        if dims is None:
            if hasattr(k, "__iter__"): raise ValueError("scalar k when dims=None")
            return torch.stack([(beta * ((Zg[:, j] - center[j]) ** k)).sum() for j in range(d)])
        dims = list(dims);
        ks   = list(k) if hasattr(k, "__iter__") else [k] * len(dims)
        prod = torch.ones(Zg.size(0), device=Zg.device, dtype=Zg.dtype)
        for jj, kk in zip(dims, ks): prod = prod * ((Zg[:, jj] - center[jj]) ** kk)
        return (beta * prod).sum()

    # ---------- ALL-RUNS PAGE ----------
    @torch.no_grad()
    def plot_all_statistics(self, Z_true, lambda_rec: float, nu: float, length_scale: float, sigma: float,
                            taus=(0.1, 0.25, 0.5, 0.75, 0.9), dim: int = 0,
                            Zgrid_dir: str = "./data", data_dir: str = "./mu",
                            outdir: str = "./plots", dpi: int = 600):
        """
        Across runs using ONLY weights_{i}.csv:
          (A) CDF mean±SD via projection vs ECDF
          (B) Quantiles mean±1.96*SD vs MC
          (C) Means: μ̂ mean±1.96*SD vs MC
          (D) Variances: μ̂ mean±SD vs MC
        """
        os.makedirs(outdir, exist_ok=True)
        font_family = "Bitstream Charter"
        titlesize = labelsize = 14
        ticksize = 11
        legendsize = 12
        plt.rcParams.update({"font.family": font_family})

        Z_true = torch.as_tensor(Z_true)
        Z_true = Z_true.view(-1, Z_true.shape[-1]) if Z_true.ndim == 1 else Z_true
        N, d = Z_true.shape
        if dim < 0 or dim >= d:
            print(f"[plot_all_statistics] dim={dim} out of range for d={d}.")
            return
        device, dtype = Z_true.device, Z_true.dtype

        # Step 1: Build ECDFs and true quantiles from Z_true
        t_list, F_true_list, qs_true_list = [], [], []
        for j in range(d):
            zvals_j = Z_true[:, j]
            lo, hi  = float(zvals_j.min()), float(zvals_j.max())
            pad     = 0.05 * (hi - lo + 1e-9)
            t_j     = torch.linspace(lo - pad, hi + pad, 5000, device=device, dtype=dtype)
            F_true_j = (zvals_j.view(1, -1) <= t_j.view(-1, 1)).float().mean(dim=1)
            qs_true_j = np.array([
                float(torch.quantile(zvals_j, q=torch.tensor(tau, device=device, dtype=dtype)))
                for tau in taus
            ])
            t_list.append(t_j)
            F_true_list.append(F_true_j)
            qs_true_list.append(qs_true_j)

        mu_true = torch.mean(Z_true, dim=0).cpu().numpy()
        var_true = torch.var(Z_true, dim=0, unbiased=True).cpu().numpy()
        med_true = torch.median(Z_true, dim=0).values.cpu().numpy()

        #---------------------------------------------------
        # Step 2: Sweep over runs
        # Extract suffix from filename using regex
        def extract_index(fname, pattern):
            match = re.search(pattern, os.path.basename(fname))
            return int(match.group(1)) if match else None

        # Gather available files
        weight_files = glob.glob(os.path.join(data_dir, "weights_*.csv"))
        zgrid_files  = glob.glob(os.path.join(Zgrid_dir, "Zgrid_*.pt"))

        # Map: index -> filepath
        weights_by_idx = {extract_index(f, r"weights_(\d+)\.csv"): f for f in weight_files}
        zgrids_by_idx  = {extract_index(f, r"Zgrid_(\d+)\.pt"): f for f in zgrid_files}

        # Intersect available indices
        matched_indices = sorted(set(weights_by_idx) & set(zgrids_by_idx))
        if not matched_indices:
            print("No matching weight/grid file pairs found.")
            return

        ########################################
        mu_list, var_list, Med_list = [], [], []
        F_lists_by_dim    = [[] for _ in range(d)]
        Q_lists_by_dim    = [[] for _ in range(d)]
        used = 0
        ########################################
        # Now loop over matched pairs only
        for idx in matched_indices:
            w_file = weights_by_idx[idx]
            z_file = zgrids_by_idx[idx]

            beta_np     = np.ravel(np.loadtxt(w_file, delimiter=","))
            beta         = torch.as_tensor(beta_np, device=device, dtype=dtype).view(-1)
            Z_grid_full  = torch.load(z_file).to(device=device, dtype=dtype)
            if Z_grid_full.shape[0] == 0:
                continue
        #---------------------------------------------------
            mu_run, var_run, med_run = [], [], []
            for j in range(d):
                idx = torch.argsort(Z_grid_full[:, j])
                z_sorted = Z_grid_full[idx].contiguous()
                beta_sorted = beta[idx]
                cache_full = {"Z_grid": z_sorted, "beta": beta_sorted}

                # CDF estimate
                F_hat_t = self.cdf_from_embedding_projection(
                    cache_full, t_list[j], j, lambda_rec, nu, length_scale, sigma
                ).detach().cpu().numpy()
                F_lists_by_dim[j].append(F_hat_t)

                # Quantiles estimate
                qs_hat_j = [
                    self.quantile_from_cdf_projection(cache_full, float(tau), j,
                                                      lambda_rec, nu, length_scale, sigma)
                    for tau in taus
                ]
                Q_lists_by_dim[j].append(np.array(qs_hat_j, dtype=float))
                med_j = self.quantile_from_cdf_projection(cache_full,float(0.5),j,lambda_rec, nu, length_scale, sigma)
                # Mean & Variance
                mu_j = self.moment_k(cache_full, k=1, dims=[j]).item()
                center_vec = torch.zeros(d, device=z_sorted.device)
                center_vec[j] = mu_j
                var_j = self.moment_k(cache_full, k=2, dims=[j], center=center_vec).item()

                mu_run.append(mu_j)
                var_run.append(var_j)
                med_run.append(med_j)

            mu_list.append(np.array(mu_run))
            var_list.append(np.array(var_run))
            Med_list.append(np.array(med_run))
            used += 1

        mu_arr  = np.vstack(mu_list)
        var_arr = np.vstack(var_list)
        med_arr = np.vstack(Med_list)
        mu_mean, mu_sd = mu_arr.mean(axis=0), mu_arr.std(axis=0, ddof=1)
        var_mean, var_sd = var_arr.mean(axis=0), var_arr.std(axis=0, ddof=1)
        print("mu_arr shape:", mu_arr.shape)
        print("mu_arr std per dim:", mu_arr.std(axis=0))

        if used == 0:
            print("[plot_all_statistics] No usable runs.")
            return

        # Step 3: Aggregate statistics
        Fm_list, Fsd_list, Qm_list, Qsd_list = [], [], [], []
        for j in range(d):
            F_arr_j   = np.vstack(F_lists_by_dim[j])
            Q_arr_j   = np.vstack(Q_lists_by_dim[j])
            Fm_list.append(F_arr_j.mean(axis=0))
            Fsd_list.append(F_arr_j.std(axis=0, ddof=1))
            Qm_list.append(Q_arr_j.mean(axis=0))
            Qsd_list.append(Q_arr_j.std(axis=0, ddof=1))

        # Step 4: Plot
        fig, axes = plt.subplots(3, d, figsize=(16, 12))

        # Row 1: CDFs
        for j in range(d):
            ax = axes[0, j]
            t_np = t_list[j].cpu().numpy()
            Fm, Fsd = Fm_list[j], Fsd_list[j]
            interp = interp1d(t_np, Fm, kind="linear", bounds_error=False, fill_value=(0, 1))
            t_dense = np.linspace(t_np.min(), t_np.max(), 5000)
            F_dense = interp(t_dense)

            upper = Fm + 1.96 * Fsd
            lower = Fm - 1.96 * Fsd
            interp_u = interp1d(t_np, upper, kind="linear", bounds_error=False, fill_value=(0, 1))
            interp_l = interp1d(t_np, lower, kind="linear", bounds_error=False, fill_value=(0, 1))
            ax.plot(t_dense, F_dense, color='#FF5F05', lw=2, label=r"$\widehat F$ mean")
            ax.fill_between(t_dense, interp_l(t_dense), interp_u(t_dense), alpha=0.2, label="±1.96 SD")
            ax.plot(t_np, F_true_list[j].cpu().numpy(), lw=1, color="k", alpha=0.8, label=r"$F$ (MC True)")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(rf"(A{j + 1}) Averaged CDF of $Z[{j + 1}]$", fontsize=titlesize)
            ax.set_xlabel(f"$z_{{{j + 1}}}$", fontsize=labelsize)
            ax.set_ylabel("CDF", fontsize=labelsize)
            ax.grid(alpha=0.3)
            ax.legend(prop={"family": font_family, "size": legendsize})

        # Row 2: Quantiles
        for j in range(d):
            ax = axes[1, j]
            t_np = t_list[j].cpu().numpy()
            Qm, Qsd = Qm_list[j], Qsd_list[j]
            ax.plot(t_np, F_true_list[j].cpu().numpy(), lw=0.7, ls="--", color="k", label="true CDF")
            for tau, q_mean, q_sd in zip(taus, Qm, Qsd):
                ax.axvline(q_mean, color='#FF5F05', lw=2)
                ax.axvspan(q_mean - 1.96 * q_sd, q_mean + 1.96 * q_sd, alpha=0.05)
                ax.text(q_mean, 0.7, fr"$\widehat q_{{{int(100 * tau)}}}$", color='#FF5F05',
                        rotation=90, ha="right", va="bottom", fontsize=ticksize)
            for tau, q in zip(taus, qs_true_list[j]):
                ax.axvline(q, color="k", ls="--", lw=1.2)
                ax.text(q, 0.4, f"q{int(100 * tau)}", rotation=90, ha="left", va="bottom", fontsize=ticksize)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(rf"(B{j + 1}) Averaged Quantiles of $Z[{j + 1}]$", fontsize=titlesize)
            ax.set_xlabel(f"$z_{{{j + 1}}}$", fontsize=labelsize)
            ax.set_ylabel("CDF", fontsize=labelsize)
            ax.grid(alpha=0.3)

        # Row 3: Means and Variances
        axC, axD, axE = axes[2, 0], axes[2, 1], axes[2, 2]
        x = np.arange(d)

        axC.scatter(x + 0.25, mu_true, lw=3, color='k', marker='x', label=r"$\mu$ (MC)")
        axC.boxplot(mu_arr, positions=x, showmeans=True,
                    meanprops=dict(marker='x', lw=2, markeredgecolor='#FF5F05'),
                    boxprops=dict(linewidth=1, color='k'),
                    whiskerprops=dict(linewidth=1, color='k'),
                    capprops=dict(linewidth=1, color='k'),
                    medianprops=dict(linewidth=2, color='k'))
        axC.set_xticks(x)
        axC.set_xticklabels([i + 1 for i in range(d)])
        axC.set_xlabel("Z dimension", fontsize=labelsize)
        axC.set_ylabel("Mean", fontsize=labelsize)
        axC.set_title(r"(C) Means: $\widehat{\mu}$ vs MC-True", fontsize=titlesize)
        axC.grid(axis="y", alpha=0.25)
        ######
        axD.scatter(x + 0.25, var_true, lw=3, marker='x', color='k', label=r"$\mathrm{Var}$ (MC)")
        axD.boxplot(var_arr, positions=x, showmeans=True,
                    meanprops=dict(marker='x', lw=3, markeredgecolor='#FF5F05'),
                    boxprops=dict(linewidth=1, color='k'),
                    whiskerprops=dict(linewidth=1, color='k'),
                    capprops=dict(linewidth=1, color='k'),
                    medianprops=dict(linewidth=2, color='k'))
        axD.set_xticks(x)
        axD.set_xticklabels([i + 1 for i in range(d)])
        axD.set_xlabel("Z dimension", fontsize=labelsize)
        axD.set_ylabel("Variance", fontsize=labelsize)
        axD.set_title(r"(D) Variances: $\widehat{\mu}$ vs MC-True", fontsize=titlesize)
        axD.grid(axis="y", alpha=0.25)
        ######
        axE.scatter(x + 0.25, med_true, lw=3, marker='x', color='k', label=r"$\mathrm{Median}$ (MC)")
        axE.boxplot(med_arr, positions=x, showmeans=False,
                    boxprops=dict(linewidth=1, color='k'),
                    whiskerprops=dict(linewidth=1, color='k'),
                    capprops=dict(linewidth=1, color='k'),
                    medianprops=dict(linewidth=3,marker='x', color='#FF5F05'))
        axE.set_xticks(x)
        axE.set_xticklabels([i + 1 for i in range(d)])
        axE.set_xlabel("Z dimension", fontsize=labelsize)
        axE.set_ylabel("Median", fontsize=labelsize)
        axE.set_title(r"(E) Median: $\widehat{\mu}$ vs MC-True", fontsize=titlesize)
        axE.grid(axis="y", alpha=0.25)

        plt.subplots_adjust(bottom=0.2, wspace=0.35, hspace=0.4)
        plt.figtext(0.5, 0.05, self._footer(f"Runs used={used}"), ha="center", fontsize=legendsize, wrap=True)
        fname = self._fname("stats_allDims")
        plt.savefig(os.path.join(outdir, fname), dpi=dpi)
        plt.close()

    # ================= END =================




    # ---------- ONE-RUN PAGE ----------
    @torch.no_grad()
    def plot_statistics(self, cache, Z_true,lambda_rec: float, nu: float, length_scale: float, sigma: float,
                        taus=(0.1, 0.25, 0.5, 0.75, 0.9), dim: int = 0 ,
                        outdir="./plots/ind_plots", dpi=600):
        """
        (A) CDF via RKHS projection; (B) Quantiles from (A);
        (C) Mean match; (D) Variance match.
        """
        os.makedirs(outdir, exist_ok=True)
        Zg = cache["Z_grid"];
        device, dtype = Zg.device, Zg.dtype
        Z_true = torch.as_tensor(Z_true, device=device, dtype=dtype);
        d = Zg.shape[1]

        font_family = "Bitstream Charter";
        titlesize = labelsize = 14;
        ticksize = legendsize = 10

        # common t-grid
        z_hat, z_true = Zg[:, dim], Z_true[:, dim]
        lo = float(torch.minimum(z_hat.min(), z_true.min()));
        hi = float(torch.maximum(z_hat.max(), z_true.max()))
        pad = 0.05 * (hi - lo + 1e-9)
        t = torch.linspace(lo - pad, hi + pad, 2000, device=device, dtype=dtype)

        # CDFs
        # CDFs (compute on grid -> interpolate to t for continuity)
        prep = cache.get("_cdf_proj_cache")
        if (prep is None or not (prep["dim"] == dim
                                 and abs(prep["lambda_rec"] - lambda_rec) < 1e-14
                                 and abs(prep["nu"] - nu) < 1e-12
                                 and abs(prep["length_scale"] - length_scale) < 1e-12
                                 and abs(prep["sigma"] - sigma) < 1e-12)):
            prep = self._prepare_cdf_projection(cache, dim, lambda_rec, nu, length_scale, sigma)

        z_sorted = prep["z_sorted"];
        F_grid   = prep["prefix_v"]
        # F_grid = self.isotonic_project01(z_sorted, F_grid)

        F_hat = torch.as_tensor(
            np.interp(t.cpu().numpy(),
                      z_sorted.cpu().numpy(),
                      F_grid.cpu().numpy(),
                      left=float(F_grid[0].item()),
                      right=float(F_grid[-1].item())),
            device=device, dtype=dtype)

        F_true = (z_true.view(1, -1) <= t.view(-1, 1)).to(dtype).mean(dim=1)

        # Quantiles
        qs_hat = [
            (tau, float(self.quantile_from_cdf_projection(cache, float(tau), dim, lambda_rec, nu, length_scale, sigma)))
            for tau in taus]
        qs_true = [(tau, float(torch.quantile(z_true, q=torch.tensor(tau, device=device, dtype=dtype)))) for tau in
                   taus]

        # Means & variances
        mu_hat   = self.moment_k(cache, k=1, dims=None)
        mu_true  = torch.mean(Z_true, dim=0)
        var_hat  = self.moment_k(cache, k=2, dims=None, center=mu_hat).cpu().numpy()
        var_true = torch.var(Z_true, dim=0, unbiased=True).cpu().numpy()

        # ---- plot (2x2) ----
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
        plt.subplots_adjust(wspace=0.35, hspace=0.45, bottom=0.18)

        # (A)
        ax = axes[0, 0]
        ax.plot(t.cpu().numpy(), F_hat.cpu().numpy(), lw=2, label=r"$\widehat F$ (proj)")
        ax.plot(t.cpu().numpy(), F_true.cpu().numpy(), lw=1.8, ls="--", label=r"$F$ (MC)")
        ax.set_ylim(-0.02, 1.02);
        ax.set_title(f"(A) CDF of $Z[{dim + 1}]$",
                     fontname=font_family, fontsize=titlesize)
        ax.set_xlabel(f"$z_{{{dim + 1}}}$", fontname=font_family, fontsize=labelsize)
        ax.set_ylabel("F(t)", fontname=font_family, fontsize=labelsize)
        ax.grid(alpha=0.3);
        ax.tick_params(labelsize=ticksize)
        ax.legend(frameon=False, prop={"family": font_family, "size": legendsize})

        # (B)
        ax = axes[0, 1]
        ax.plot(t.cpu().numpy(), F_hat.cpu().numpy(), lw=2, label=r"$\widehat F$")
        ax.plot(t.cpu().numpy(), F_true.cpu().numpy(), lw=1.2, ls="--", label=r"$F$ (MC)")
        for tau, q in qs_hat:
            ax.axvline(q, ls="-", lw=1.2, alpha=0.9);
            ax.text(q, 0.02, f"q̂{int(100 * tau)}", rotation=90, fontsize=8,
                    ha="right", va="bottom", fontname=font_family)
        for tau, q in qs_true:
            ax.axvline(q, ls="--", lw=1.2, alpha=0.9);
            ax.text(q, 0.02, f"q{int(100 * tau)}", rotation=90, fontsize=8,
                    ha="left", va="bottom", fontname=font_family)
        ax.set_ylim(-0.02, 1.02);
        ax.set_title("(B) Quantiles: μ̂ (solid) vs MC (dashed)",
                     fontname=font_family, fontsize=titlesize)
        ax.set_xlabel(f"$z_{{{dim + 1}}}$", fontname=font_family, fontsize=labelsize)
        ax.set_ylabel("F(t)", fontname=font_family, fontsize=labelsize)
        ax.grid(alpha=0.3);
        ax.tick_params(labelsize=ticksize)
        ax.legend(frameon=False, prop={"family": font_family, "size": legendsize}, loc="lower right")

        # (C) mean match
        ax = axes[1, 0];
        x = np.arange(d)
        # plot thick estimated and true lines
        ax.plot(x, mu_hat.cpu().numpy(), lw=5, marker='o', label=r"$\hat\mu$")
        ax.plot(x, mu_true.cpu().numpy(), lw=5, marker='x', linestyle='--', label=r"$\mu$ (MC)")
        ax.set_xticks(x);
        ax.set_xticklabels(range(d))
        ax.set_xlabel("Z dimension", fontname=font_family, fontsize=labelsize)
        ax.set_ylabel("Mean", fontname=font_family, fontsize=labelsize)
        ax.set_title("(C) Means: μ̂ vs MC", fontname=font_family, fontsize=titlesize)
        ax.legend(frameon=False, prop={"family": font_family, "size": legendsize})
        ax.grid(axis="y", alpha=0.25);
        ax.tick_params(labelsize=ticksize)

        # (D) variance match
        ax = axes[1, 1]
        ax.plot(x, var_hat, lw=5, marker='o', label=r"$\widehat{\mathrm{Var}}$")
        ax.plot(x, var_true, lw=5, marker='x', linestyle='--', label=r"$\mathrm{Var}$ (MC)")
        ax.set_xticks(x);
        ax.set_xticklabels(range(d))
        ax.set_xlabel("Z dimension", fontname=font_family, fontsize=labelsize)
        ax.set_ylabel("Variance", fontname=font_family, fontsize=labelsize)
        ax.set_title(r"(D) Variances: $\widehat{\mu}$ vs MC-True", fontname=font_family, fontsize=titlesize)
        ax.legend(frameon=False, prop={"family": font_family, "size": legendsize})
        ax.grid(axis="y", alpha=0.25);
        ax.tick_params(labelsize=ticksize)

        plt.subplots_adjust(bottom=0.2)
        fig.text(0.5, 0.05, self._footer(), ha="center", fontsize=10, fontname=font_family, wrap=True)
        fname = f"Summary_stats_compare_dim{dim}"
        plt.savefig(f"{outdir}/{self._fname(fname)}", dpi=dpi);
        plt.close()
        del Z_true

