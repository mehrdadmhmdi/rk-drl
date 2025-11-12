import os, glob
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from functools import partial
from .matern_kernel import matern_kernel
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
import re


class RecoverAndPlot:
    """
    PLOTS toolkit (estimate-only):
      - recover_joint_beta
      - marginals_from_beta
      - mean_embeddings_full         # now computes/saves ONLY μ̂
      - mean_embedding_joint_2d_hat  # 2D slice of μ̂ (no μ_true)
      - plot_densities               # only estimated marginals
      - plot_mu_per_dim              # only μ̂ ribbons
      - plot_mu_joint_contour        # only μ̂ contour
      - plot_mean_embedding_3d_slice # only μ̂ surface
      - plot_bellman_error
      - plot_total_loss
      - plot_operator_check_2d       # compares μ̂ vs T μ̂ (no truth)
      - plot_all_mu                  # across-run ribbons for μ̂ only
      - export_metrics_tables        # unchanged; also exports moments from weights
    """

    def __init__(self, config):
        self.cfg = config
        self.true_kde_grids = {}

    # -------------------- helpers --------------------
    def _footer(self, extra: str = "") -> str:
        c = self.cfg or {}

        def gj(k, default):
            return c.get(k, default)

        # numbers/arrays with safe formatting
        nu = gj('nu', 'N/A')
        ell = gj('length_scale', 'N/A')
        sig = gj('sigma_k', 'N/A')
        try:
            import numpy as np
            sig_str = np.array(sig).round(2).tolist() if isinstance(sig, (list, tuple)) else (
                round(float(sig), 2) if isinstance(sig, (int, float)) else sig)
        except Exception:
            sig_str = sig

        # text fields
        tp   = gj('target_policy', 'N/A')
        sdim = gj('state_dim', 'N/A')
        rdim = gj('reward_dim', 'N/A')
        adim = gj('action_dim', 'N/A')

        lr   = gj('lr', 'N/A')
        lam  = gj('lambda_reg', 'N/A')
        fpc  = gj('fixed_point_constraint', 'N/A')
        fpl  = gj('FP_penalty_lambda', 'N/A')
        lrec = gj('lambda_rec', 'N/A')
        bw   = gj('bandwidth', 'N/A')
        hef  = gj('hull_expand_factor', 'N/A')
        sst  = gj('s_star', 'N/A')
        ast  = gj('a_star', 'N/A')

        base = (
            f"Target Policy: {tp}"
            rf" Matérn($\nu,\ell,\sigma$)=({nu},{ell},{sig_str}) | "
            f"dims(S,R,A)=({sdim},{rdim},{adim})"
            f"\nlr={lr} | Training Reg-$\\lambda$={lam} | "
            f"FP_constraint={fpc} | $FP_\\lambda$={fpl} | "
            f"Recovery Reg-$\\lambda$={lrec}"
            f"Density G-Bandwidth={bw} | Z grid expand factor = {hef}"
            f"\n(s*,a*)=({sst}, {ast})"
        )
        return base + (f" | {extra}" if extra else "")

    def _fname(self, plot_type: str) -> str:
        c = self.cfg
        return f"{plot_type}.png"

    @staticmethod
    def _trapz1(u: torch.Tensor, x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.ndim != 1:
            raise ValueError("x must be 1D for _trapz1.")
        if x.numel() != u.size(dim):
            raise ValueError(f"x length ({x.numel()}) must equal u.size(dim) ({u.size(dim)}).")
        dx = x[1:] - x[:-1]
        sl0 = [slice(None)] * u.ndim; sl0[dim] = slice(0, -1)
        sl1 = [slice(None)] * u.ndim; sl1[dim] = slice(1, None)
        u0 = u[tuple(sl0)]; u1 = u[tuple(sl1)]
        shape = [1] * u.ndim; shape[dim] = dx.numel()
        dxv = dx.view(shape)
        return (0.5 * (u0 + u1) * dxv).sum(dim=dim)

    # -------------------- single joint solve --------------------
    @staticmethod
    @torch.no_grad()
    def _beta_full(B, k_sa, phi, Z_grid, K_sa, *, method, nu, length_scale, sigma_k, lambda_reg):
        if method == "song":
            return (B.T @ k_sa.view(-1)).contiguous()
        if method == "bellman":
            return (B.T @ phi.view(-1)).contiguous()
        if method == "Schuster":
            # NOTE: left as-is; untouched
            L = torch.linalg.cholesky(A)
            L_inv = torch.linalg.inv(L)
            A_inv_cholesky = torch.matmul(L_inv.T, L_inv)

            m = Kgg.size(0)
            n = K_sa.size(0)

            Kgg = matern_kernel(Z_grid, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma_k)
            Aux_Kgg = torch.linalg.inv(torch.linalg.cholesky(Kgg +  lambda_reg * torch.eye(m, device=Z_grid.device, dtype=Z_grid.dtype)))
            Kgg_inv = torch.matmul(Aux_Kgg.T, Aux_Kgg)

            Aux_K_sa = torch.linalg.inv(torch.linalg.cholesky(K_sa + n*lambda_reg * torch.eye(n, device=Z_grid.device, dtype=Z_grid.dtype)))
            K_sa_inv = torch.matmul(Aux_K_sa.T, Aux_K_sa)

            num = (Kgg_inv**2) * Kgg * K_sa_inv * k_sa
            OP  = num/(m**2)
            return OP*Kgg
        raise ValueError(f"unknown method: {method}")

    @torch.no_grad()
    def recover_joint_beta(self, B, k_sa, Z_grid, phi, K_sa, *, nu, length_scale, sigma_k, method, lambda_reg):
        beta = self._beta_full(B, k_sa, phi, Z_grid, K_sa,
                               method=method, nu=nu, length_scale=length_scale,
                               sigma_k=sigma_k, lambda_reg=lambda_reg)
        self.save_csv(beta.detach().cpu().numpy(), f"weights.csv", base_path="./mu")
        return beta, Z_grid

    # -------------------- evaluation (no re-solves) --------------------
    @torch.no_grad()
    def marginals_from_beta(self, beta_full, Z_grid_full, reward_dim, *,
                            nu, length_scale, sigma_k, lambda_rec, bandwidth=0.5, n_grid=400, margin_factor=0.35):
        device, dtype = Z_grid_full.device, Z_grid_full.dtype
        n = Z_grid_full.shape[0]
        F_cols, grids = [], {}
        for j in range(reward_dim):
            Zj = Z_grid_full[:, [j]]
            zmin, zmax = Zj.min().item(), Zj.max().item()
            r = (zmax - zmin) * margin_factor
            u = torch.linspace(zmin - r, zmax + r, n_grid, device=device, dtype=dtype).unsqueeze(1)

            KZ = matern_kernel(Zj, Zj, nu=nu, length_scale=length_scale, sigma=sigma_k)  # [n,n]
            KZ = 0.5 * (KZ + KZ.T)
            M = torch.linalg.cholesky(KZ + (n * lambda_rec) * torch.eye(n, device=device, dtype=dtype))

            z = Zj.view(-1)
            h = bandwidth
            T = (z - u) / h
            g = torch.exp(-0.5 * T ** 2) / (math.sqrt(2.0 * math.pi) * h)  # [n_grid,n]

            C = torch.cholesky_solve(g.T, M)  # [n, n_grid]

            omega = beta_full.view(-1)  # [n]
            Kw = KZ @ omega  # [n]
            f = (Kw.unsqueeze(0) @ C).view(-1)  # [n_grid]

            f = f.clamp_min(0)
            mass = self._trapz1(f, u.view(-1), dim=0)
            f = f / (mass + torch.finfo(dtype).eps)

            F_cols.append(f)
            grids[j] = u.view(-1).detach().cpu().numpy()
        F = torch.stack(F_cols, dim=1)
        return F, grids

    # -------------------- plots (estimate-only) --------------------
    def plot_densities(self, fz, grid_dict, outdir="./plots/"):
        """
        plot estimated marginals.
        """
        d = self.cfg['reward_dim']
        fig, axes = plt.subplots(1, d, figsize=(6 * d, 6))
        axes = np.atleast_1d(axes)

        fz_np = fz.detach().cpu().numpy() if hasattr(fz, "detach") else np.asarray(fz)

        for j, ax in enumerate(axes):
            ax.plot(np.asarray(grid_dict[j]), fz_np[:, j], color="#FF5F05", lw=2, label="Estimated Marginal")
            ax.set_title(f"Z-dim {j + 1}")
            ax.set_xlabel("Z"); ax.set_ylabel("Density")
            ax.legend(fontsize=8)
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(0.5, 0.04, self._footer(), ha="center", fontsize=11, wrap=True)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/{self._fname('marginal_fZ')}", dpi=600)
        plt.close()

    def plot_bellman_error(self, hist_be, outdir="./plots/"):
        plt.figure(figsize=(7, 5))
        plt.plot(np.asarray(hist_be), color="#FF5F05")
        plt.xlabel("Iteration"); plt.ylabel("Log ‖Bellman Error‖")
        plt.title("Log Bellman Error (ADAM)")
        plt.grid(alpha=0.3, linestyle="--")
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(0.5, 0.04, self._footer(f"Final log BE: {hist_be[-1]:.6e} | Final BE: {math.exp(hist_be[-1]):.6e} "), ha="center", fontsize=10, wrap=True)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/{self._fname('BellmanError_ADAM')}", dpi=600)
        plt.close()

    def plot_total_loss(self, hist_obj, outdir="./plots/"):
        plt.figure(figsize=(7, 5))
        plt.plot(np.asarray(hist_obj), color="#FF5F05")
        plt.xlabel("Iteration"); plt.ylabel("Log Loss (objective + penalty)")
        plt.title("Log Total Loss (ADAM)", fontsize=12)
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
    ##  Mean Embedding (estimate-only)
    ##========================================
    @torch.no_grad()
    def mean_embeddings_full(self, beta, Z_grid, *, nu, length_scale, sigma_k):
        """
        Computes μ̂ on the given full d-D grid (no μ_true).
        Saves μ̂ CSV and returns cache.
        """
        Kgg = matern_kernel(Z_grid, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma_k)
        mu_hat = (Kgg @ beta.view(-1)).contiguous()

        self.save_csv(mu_hat.detach().cpu().numpy(), f"mu_hat.csv", base_path="./mu")

        return {
            "Z_grid": Z_grid, "beta": beta, "mu_hat": mu_hat,
            "nu": nu, "length_scale": length_scale, "sigma_k": sigma_k
        }

    @torch.no_grad()
    def mean_embedding_joint_2d_slice_hat(self, cache, *, dims=(0, 1), ref="median",
                                          n1=120, n2=120, margin_factor=0.35):
        """
        2-D slice of μ̂ only on (dims=j,k) using the cache. Returns (μ̂_2d, Xg, Yg).
        """
        beta, Z_grid = cache["beta"], cache["Z_grid"]
        nu, length_scale, sigma_k = cache["nu"], cache["length_scale"], cache["sigma_k"]

        device, dtype = Z_grid.device, Z_grid.dtype
        j, k = dims

        ref_vec = Z_grid.median(0).values if ref == "median" else Z_grid.mean(0)

        x_min, x_max = Z_grid[:, j].min().item(), Z_grid[:, j].max().item()
        y_min, y_max = Z_grid[:, k].min().item(), Z_grid[:, k].max().item()
        rx = (x_max - x_min) * margin_factor
        ry = (y_max - y_min) * margin_factor
        x = torch.linspace(x_min - rx, x_max + rx, n1, device=device, dtype=dtype)
        y = torch.linspace(y_min - ry, y_max + ry, n2, device=device, dtype=dtype)
        Xg, Yg = torch.meshgrid(x, y, indexing="xy")

        Q = ref_vec.repeat(n1 * n2, 1)
        Q[:, j] = Xg.reshape(-1); Q[:, k] = Yg.reshape(-1)

        Kqg = matern_kernel(Q, Z_grid, nu=nu, length_scale=length_scale, sigma=sigma_k)
        mu_hat2 = (Kqg @ beta.view(-1)).view(n2, n1).contiguous()

        # Save slice CSV (hat only)
        self.save_csv(mu_hat2.detach().cpu().numpy(), f"mu2D_hat_dims{j}{k}.csv", base_path="./mu")
        return mu_hat2, Xg, Yg

    def plot_mean_embedding_3d_slice(self, cache, *, dims=(0, 1), ref="median",
                                     n1=120, n2=120, margin_factor=0.35,
                                     outdir="./plots/"):
        """
        Single 3D surface for μ̂ on a 2-D slice (dims=j,k).
        """
        import os, numpy as np, matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        os.makedirs(outdir, exist_ok=True)
        font_family = "Bitstream Charter"
        plt.rcParams.update({
            "font.family": font_family,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 12,
        })

        mu_hat2, Xg_t, Yg_t = self.mean_embedding_joint_2d_slice_hat(
            cache, dims=dims, ref=ref, n1=n1, n2=n2, margin_factor=margin_factor
        )

        Xg = Xg_t.detach().cpu().numpy()
        Yg = Yg_t.detach().cpu().numpy()
        Z1 = mu_hat2.detach().cpu().numpy()

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        surf1 = ax1.plot_surface(Xg, Yg, Z1, rcount=100, ccount=100,
                                 cmap="viridis", linewidth=0, antialiased=True)
        ax1.set_title(r"Estimated mean embedding $\hat{\mu}$")
        ax1.set_xlabel(f"Z[{dims[0]+1}]"); ax1.set_ylabel(f"Z[{dims[1]+1}]")
        ax1.zaxis.set_rotate_label(False)
        ax1.set_zlabel(r"$\hat{\mu}$", rotation=90, labelpad=8)
        for lab in ax1.get_xticklabels() + ax1.get_yticklabels() + ax1.get_zticklabels():
            lab.set_fontname(font_family)

        cbar = fig.colorbar(surf1, shrink=0.7, pad=0.1)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("value", fontsize=12)
        plt.subplots_adjust(bottom=0.25)
        plt.figtext(0.5, 0.02, self._footer(), ha="center", fontsize=10, wrap=True)
        fname = self._fname(f"mu_3D")
        plt.savefig(os.path.join(outdir, fname), dpi=400)
        plt.close()
        print(f"Saved {os.path.join(outdir, fname)}")

    def plot_mu_joint_contour(self, mu_hat_2d, Xg, Yg, dims=(0, 1), outdir="./plots/"):
        os.makedirs(outdir, exist_ok=True)
        Z1, Z2 = Xg.detach().cpu().numpy(), Yg.detach().cpu().numpy()
        EH = mu_hat_2d.detach().cpu().numpy()
        vmin, vmax = float(EH.min()), float(EH.max())
        levels = np.linspace(vmin, vmax, 15)

        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        c1 = ax1.contourf(Z1, Z2, EH, levels=levels, cmap="Blues")
        fig.colorbar(c1, ax=ax1)
        ax1.set(title=fr"$\hat\mu$ contour", xlabel=f"Z[{dims[0]+1}]", ylabel=f"Z[{dims[1]+1}]")
        plt.subplots_adjust(bottom=0.45)
        plt.figtext(0.5, 0.02, self._footer(), ha="center", fontsize=8, wrap=True)
        fname = self._fname(f"mu2D")
        plt.savefig(f"{outdir}/{fname}", dpi=600)
        plt.close()

    # ========== Orchestrator (estimate-only) =========================
    def mean_embedding_all(self, beta, Z_grid, *,
                           nu, length_scale, sigma_k,
                           do_joint_dims=(0, 1), n1=120, n2=120, margin_factor=0.35,
                           outdir="./plots/"):
        """
        - Computes μ̂ on full grid once (CSV) and plots per-dim.
        - Computes a 2-D slice (CSV) and plots (contour + 3D) for μ̂ only.
        """
        cache = self.mean_embeddings_full(
            beta, Z_grid, nu=nu, length_scale=length_scale, sigma_k=sigma_k
        )

        # per-dim plot (μ̂ only)
        self.plot_mu_per_dim(cache, outdir=outdir)

        # joint 2-D slice (μ̂ only)
        mu2d_hat, Xg, Yg = self.mean_embedding_joint_2d_slice_hat(
            cache, dims=do_joint_dims, n1=n1, n2=n2, margin_factor=margin_factor
        )
        self.plot_mu_joint_contour(mu2d_hat, Xg, Yg, dims=do_joint_dims, outdir=outdir)
        self.plot_mean_embedding_3d_slice(cache, dims=do_joint_dims, ref="median",
                                          n1=n1, n2=n2, margin_factor=margin_factor, outdir=outdir)
        return cache, (mu2d_hat, Xg, Yg)

    # ---------- estimate-only per-dim ribbons ----------
    def plot_mu_per_dim(self, cache, outdir="./plots/"):
        os.makedirs(outdir, exist_ok=True)
        Zg = cache["Z_grid"].detach().cpu().numpy()  # (m,d)
        mh = cache["mu_hat"].detach().cpu().numpy()  # (m,)
        m, d = Zg.shape

        # simple smoothing
        k = max(5, (m // 50) | 1)
        ker = np.hanning(k); ker /= ker.sum()
        smooth = lambda y: np.convolve(y, ker, mode="same")

        fig, axes = plt.subplots(1, d, figsize=(5 * d, 4), sharey=True)
        axes = np.atleast_1d(axes)
        for j in range(d):
            x = Zg[:, j]; idx = np.argsort(x)
            ax = axes[j]
            ax.plot(x[idx], smooth(mh[idx]), label=fr"$\hat \mu$ (dim {j+1})", lw=2, color="#FF5F05")
            ax.set_xlabel(f"Z[{j+1}]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
            if j == 0: ax.set_ylabel("Mean embedding value")

        fig.suptitle("Estimated Mean Embedding (per dimension)")
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(0.5, 0.02, self._footer(), ha="center", fontsize=10, wrap=True)
        fname = self._fname(f"mu")
        plt.savefig(f"{outdir}/{fname}", dpi=600); plt.close()

    # ---------- All runs (estimate-only) ----------
    def plot_all_mu(self, data_dir="./mu", outdir="./plots/", n_bins=20):
        """
        Across-run summary (μ̂ only):
          [A] mean±SD ribbons of μ̂ (over grid index)
          [B] ECDF of |μ̂ - mean(μ̂)| pooled
        """
        import os, numpy as np, matplotlib.pyplot as plt
        os.makedirs(outdir, exist_ok=True)
        data_dir = os.fspath(data_dir)

        plt.rcParams.update({
            "font.family": "Bitstream Charter",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
        })

        hat_files = sorted(glob.glob(os.path.join(data_dir, "mu_hat_*.csv")))
        if not hat_files:
            print(f"No mu_hat_*.csv in {data_dir}")
            return

        H = [np.loadtxt(f, delimiter=",").reshape(-1) for f in hat_files]
        H = np.vstack(H)
        R, m = H.shape
        xidx = np.arange(m)

        Hm, Hsd = H.mean(axis=0), H.std(axis=0, ddof=1)
        abs_dev = np.abs(H - Hm)
        ecdf_x = np.sort(abs_dev.ravel())
        ecdf_y = np.arange(1, ecdf_x.size + 1) / ecdf_x.size

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axA, axB = axs

        axA.plot(xidx, Hm, lw=2, color="#FF5F05", label=r"$\bar{\hat\mu}$")
        axA.fill_between(xidx, Hm - 1.96*Hsd, Hm + 1.96*Hsd, color="#FF5F05", alpha=0.2, label="±1.96·SD")
        axA.set_title("(A) Mean ±1.96 SD Across Runs")
        axA.set_xlabel("Index on Z grid"); axA.set_ylabel("Mean embedding"); axA.grid(alpha=0.3); axA.legend()

        axB.plot(ecdf_x, ecdf_y, lw=2)
        axB.set_title(r"(B) ECDF of $|\hat\mu-\bar{\hat\mu}}|$ (pooled)")
        axB.set_xlabel(r"$|\hat\mu-\bar{\hat\mu}}|$"); axB.set_ylabel("ECDF"); axB.grid(alpha=0.3)

        plt.subplots_adjust(bottom=0.18, wspace=0.25)
        plt.figtext(0.5, 0.05, self._footer(f"Runs: {R}"), ha="center", fontsize=14, wrap=True)
        fname = self._fname("mu_summary_hat_only")
        plt.savefig(f"{outdir}/{fname}", dpi=600); plt.close()

    # ========== Operator check on 2D slice (needs rewards R & γ; no truths) ==========
    @torch.no_grad()
    def plot_operator_check_2d(self, cache, R, gamma, dims=(0,1),
                               n1=80, n2=80, margin_factor=0.35,
                               max_rewards=2000, outdir="./plots/"):
        os.makedirs(outdir, exist_ok=True)
        beta, Zg = cache["beta"], cache["Z_grid"]
        nu, ell, sig = cache["nu"], cache["length_scale"], cache["sigma_k"]
        dev, dt = Zg.device, Zg.dtype

        mu2d_hat, Xg, Yg = self.mean_embedding_joint_2d_slice_hat(
            cache, dims=dims, n1=n1, n2=n2, margin_factor=margin_factor
        )
        Q = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1).to(dev, dt)
        d = Zg.shape[1]
        ref = Zg.median(0).values.clone()
        Qd = ref.repeat(Q.shape[0], 1)
        Qd[:, dims[0]] = Q[:,0]; Qd[:, dims[1]] = Q[:,1]

        if R.shape[0] > max_rewards:
            ridx = torch.randperm(R.shape[0], device=R.device)[:max_rewards]
            Ruse = R[ridx].to(dev, dt)
        else:
            Ruse = R.to(dev, dt)

        Q_out = torch.zeros(Qd.shape[0], dtype=dt, device=dev)
        m = Zg.shape[0]
        for r in Ruse:
            shifted = (gamma * Zg) + r
            K = matern_kernel(Qd, shifted, nu=nu, length_scale=ell, sigma=sig)
            Q_out += K @ beta.view(-1)
            del K, shifted
        Tmu2d = (Q_out / float(Ruse.shape[0])).view_as(mu2d_hat).contiguous()

        Xn, Yn = Xg.detach().cpu().numpy(), Yg.detach().cpu().numpy()
        A = mu2d_hat.detach().cpu().numpy()
        B = Tmu2d.detach().cpu().numpy()
        RZ = (A - B)

        fig, axs = plt.subplots(1,3, figsize=(14,4))
        im0 = axs[0].pcolormesh(Xn, Yn, A, cmap="Blues", shading="auto"); fig.colorbar(im0, ax=axs[0]); axs[0].set(title=rf"$\widehat \mu$ (slice)")
        im1 = axs[1].pcolormesh(Xn, Yn, B, cmap="Greens", shading="auto"); fig.colorbar(im1, ax=axs[1]); axs[1].set(title="$T\widehat \mu$  (slice)")
        im2 = axs[2].pcolormesh(Xn, Yn, np.abs(RZ), cmap="magma", shading="auto"); fig.colorbar(im2, ax=axs[2]); axs[2].set(title="|$\widehat \mu-T\widehat \mu$|")
        for ax in axs: ax.set(xlabel=f"Z[{dims[0]+1}]", ylabel=f"Z[{dims[1]+1}]")
        plt.subplots_adjust(bottom=0.35)
        plt.figtext(0.5, 0.02, self._footer(), ha="center", fontsize=10, wrap=True)
        plt.savefig(f"{outdir}/{self._fname('OperatorCheck2D_hat_only')}", dpi=600); plt.close()

    # ================= END =================
