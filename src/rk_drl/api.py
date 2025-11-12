# src/rkdrl/api.py
from __future__ import annotations
import os, sys, json
from typing import Any, Dict, Optional, Tuple
import torch

from .RK_DRL import RK_DRL
from .density_recovery import RecoverAndPlot

# ========= FIT (thin wrapper over RK_DRL) =========
def estimate_embedding(
    *,
    s0, s1, a0, a1,
    s_star, a_star,
    r,
    target_p_choice, target_p_params,
    nu, length_scale, sigma,
    gamma_val, lambda_reg,
    num_grid_points,
    # optional (kept identical to RK_DRL defaults)
    hull_expand_factor: float = 1.8,
    lr: float = 1e-3, weight_decay: float = 0.0, num_steps: int = 5000,
    FP_penalty_lambda: float = 1e2,
    use_low_rank: bool = False, rank_for_low_rank: Optional[int] = None,
    B_positive: bool = False, fixed_point_constraint: bool = True, exact_projection: bool = False,
    ortho_lambda: float = 1e1, B_conv: bool = False, Sum_one_W: bool = False, NonNeg_W: bool = False,
    mass_anchor_lambda: float = 10.0, target_mass: float = 1.0,
    B_ridge_penalty: bool = False,
    H_batch_size: int = 10,
    device: Optional[str] = None, dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> Tuple[torch.Tensor, list, list, Dict[str, torch.Tensor]]:
    return RK_DRL(
        s0=s0, s1=s1, a0=a0, a1=a1,
        s_star=s_star, a_star=a_star, r=r,
        target_p_choice=target_p_choice, target_p_params=target_p_params,
        nu=nu, length_scale=length_scale, sigma=sigma,
        gamma_val=gamma_val, lambda_reg=lambda_reg,
        num_grid_points=num_grid_points,
        hull_expand_factor=hull_expand_factor,
        lr=lr, weight_decay=weight_decay, num_steps=num_steps,
        FP_penalty_lambda=FP_penalty_lambda,
        use_low_rank=use_low_rank, rank_for_low_rank=rank_for_low_rank,
        B_positive=B_positive, fixed_point_constraint=fixed_point_constraint, exact_projection=exact_projection,
        ortho_lambda=ortho_lambda, B_conv=B_conv, Sum_one_W=Sum_one_W, NonNeg_W=NonNeg_W,
        mass_anchor_lambda=mass_anchor_lambda, target_mass=target_mass,
        B_ridge_penalty=B_ridge_penalty,
        H_batch_size=H_batch_size,
        device=device, dtype=dtype, verbose=verbose,
    )

# ========= PLOTTING CONFIG =========
def build_plot_config(
    *,
    lr: float,
    fixed_point_constraint: bool,
    FP_penalty_lambda: float,
    Sum_one_W: bool,
    NonNeg_W: bool,
    mass_anchor_lambda: float,
    target_mass: float,
    num_steps: int,
    nu: float,
    length_scale: float,
    sigma_k: float,
    gamma_val: float,
    num_grid_points: int,
    hull_expand_factor: float,
    lambda_reg: float,
    bandwidth: float,
    lambda_rec: float,
    method: str,
    state_dim: int,
    reward_dim: int,
    action_dim: int,
    s_star,
    a_star,
    target_policy: str,
) -> Dict[str, Any]:
    return {
        'lr': lr,
        'fixed_point_constraint': fixed_point_constraint,
        'FP_penalty_lambda': FP_penalty_lambda,
        'Sum_one_W': Sum_one_W,
        'NonNeg_W': NonNeg_W,
        'mass_anchor_lambda': mass_anchor_lambda,
        'target_mass': target_mass,
        'num_steps': int(num_steps),
        'nu': float(nu),
        'length_scale': float(length_scale),
        'sigma_k': float(sigma_k),
        'gamma_val': float(gamma_val),
        'num_grid_points': int(num_grid_points),
        'hull_expand_factor': float(hull_expand_factor),
        'lambda_reg': float(lambda_reg),
        'bandwidth': float(bandwidth),
        'lambda_rec': float(lambda_rec),
        'method': str(method),
        'state_dim': int(state_dim),
        'reward_dim': int(reward_dim),
        'action_dim': int(action_dim),
        's_star': (s_star.detach().cpu().tolist() if isinstance(s_star, torch.Tensor) else s_star),
        'a_star': (a_star.detach().cpu().tolist() if isinstance(a_star, torch.Tensor) else a_star),
        'target_policy': str(target_policy),
    }

# ========= INDIVIDUAL PLOT / RECOVERY HELPERS =========
def plot_bellman_error(history_be: list, outdir: str = "./plots"):
    tool = RecoverAndPlot({})
    os.makedirs(outdir, exist_ok=True)
    tool.plot_bellman_error(history_be, outdir=outdir)

def plot_total_loss(history_obj: list, outdir: str = "./plots"):
    tool = RecoverAndPlot({})
    os.makedirs(outdir, exist_ok=True)
    tool.plot_total_loss(history_obj, outdir=outdir)

def recover_joint_beta(
    *, B: torch.Tensor, k_sa: torch.Tensor, Z_grid: torch.Tensor, Phi: torch.Tensor, K_sa: torch.Tensor,
    config: Dict[str, Any],
):
    tool = RecoverAndPlot(config)
    return tool.recover_joint_beta(
        B, k_sa, Z_grid, Phi, K_sa,
        nu=config['nu'], length_scale=config['length_scale'], sigma_k=config['sigma_k'],
        method=config['method'], lambda_reg=config['lambda_reg']
    )

def compute_marginals_from_beta(
    *, beta_full: torch.Tensor, Z_grid: torch.Tensor, config: Dict[str, Any],
    n_grid: int = 400, margin_factor: float = 0.25
):
    tool = RecoverAndPlot(config)
    return tool.marginals_from_beta(
        beta_full, Z_grid, reward_dim=config['reward_dim'],
        nu=config['nu'], length_scale=config['length_scale'], sigma_k=config['sigma_k'],
        lambda_rec=config['lambda_rec'], bandwidth=config['bandwidth'],
        n_grid=n_grid, margin_factor=margin_factor
    )

def plot_densities(
    *, fz: torch.Tensor, grid_dict: Dict[str, Any], config: Dict[str, Any], outdir: str = "./plots"
):
    tool = RecoverAndPlot(config)  # needs reward_dim for subplot count
    os.makedirs(outdir, exist_ok=True)
    tool.plot_densities(fz, grid_dict, outdir=outdir)

def mean_embedding_all(
    *, beta_full: torch.Tensor, Z_grid: torch.Tensor, config: Dict[str, Any],
    do_joint_dims=(0,1), n1: int = 120, n2: int = 120, margin_factor: float = 0.35, outdir: str = "./plots"
):
    tool = RecoverAndPlot(config)
    return tool.mean_embedding_all(
        beta_full, Z_grid,
        nu=config['nu'], length_scale=config['length_scale'], sigma_k=config['sigma_k'],
        do_joint_dims=do_joint_dims, n1=n1, n2=n2, margin_factor=margin_factor, outdir=outdir
    )

def plot_operator_check_2d(cache: Dict[str, Any], *, r_obs: torch.Tensor, gamma: float, dims=(0,1), outdir: str = "./plots"):
    tool = RecoverAndPlot({})
    tool.plot_operator_check_2d(cache, R=r_obs, gamma=gamma, dims=dims, outdir=outdir)

def save_weights_and_grid(beta_full: torch.Tensor, Z_grid: torch.Tensor, run_id: int, mu_dir="./mu", data_dir="./data"):
    os.makedirs(mu_dir, exist_ok=True); os.makedirs(data_dir, exist_ok=True)
    torch.save(Z_grid, os.path.join(data_dir, f"Zgrid_{run_id}.pt"))
    import numpy as np
    np.savetxt(os.path.join(mu_dir, f"weights_{run_id}.csv"),
               beta_full.detach().cpu().view(-1).numpy(), delimiter=",", fmt="%.8e")

# ========= CLI (estimate-only) =========
def _shape(x): return tuple(x.shape) if hasattr(x, "shape") else x

def cli():
    """
    stdin JSON:
    {
      "fit": { ... RK_DRL kwargs ... },
      "plots": {
        "config": { ... build_plot_config kwargs ... },
        "r_obs": null or array,
        "what": ["bellman","loss","beta","marginal","mean","op2d"]
      }
    }
    """
    cfg = json.load(sys.stdin)
    B, hist_obj, hist_be, pre = estimate_embedding(**cfg["fit"])
    print("OK fit:", {"B": _shape(B), "hist_obj": len(hist_obj), "hist_be": len(hist_be),
                      **{k: _shape(v) for k, v in pre.items()}})

    if "plots" in cfg:
        pc = cfg["plots"]
        config = build_plot_config(**pc["config"])
        r_obs  = (torch.as_tensor(pc["r_obs"]) if pc.get("r_obs") is not None else None)

        what = set(pc.get("what", []))
        if "bellman" in what: plot_bellman_error(hist_be)
        if "loss"    in what: plot_total_loss(hist_obj)

        if {"beta","marginal","mean","op2d"} & what:
            beta, Zg = recover_joint_beta(B=B, k_sa=pre["k_sa"], Z_grid=pre["Z_grid"],
                                          Phi=pre["Phi"], K_sa=pre["K_sa"], config=config)
            print("OK beta:", {"beta": _shape(beta), "Zg": _shape(Zg)})

            if "marginal" in what:
                fz, grid = compute_marginals_from_beta(beta_full=beta, Z_grid=Zg, config=config)
                plot_densities(fz=fz, grid_dict=grid, config=config)

            if "mean" in what or "op2d" in what:
                cache, _ = mean_embedding_all(beta_full=beta, Z_grid=Zg, config=config)
                if "op2d" in what and r_obs is not None:
                    plot_operator_check_2d(cache, r_obs=r_obs, gamma=config["gamma_val"])
