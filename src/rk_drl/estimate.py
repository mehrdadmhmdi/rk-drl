import torch
od = torch.load("offline_data.pt", map_location="cpu")
zt = torch.load("Z_true.pt",     map_location="cpu")

s0, s1, a0, a1, r = od["s0"], od["s1"], od["a0"], od["a1"], od["r0"]
s_star, a_star    = zt["metadata"]["s_star"], zt["metadata"]["a_star"]
target_p_choice   = zt["metadata"]["policy"]
tp_raw            = zt["metadata"]["policy_params"][target_p_choice]
target_p_params   = {target_p_choice: {k: v for k, v in tp_raw.items() if k != "name"}}



## ============================================
# 1) import the API
from rk_drl.api import estimate_embedding, build_plot_config, recover_joint_beta, compute_marginals_from_beta, plot_densities

# 2) fit (gets the embedding basis B and intermediates)
B, hist_obj, hist_be, pre = estimate_embedding(
    s0=s0, s1=s1, a0=a0, a1=a1,
    s_star=s_star, a_star=a_star,
    r=r,
    target_p_choice=target_p_choice,
    target_p_params=target_p_params,
    nu=4.5, length_scale=2, sigma=1.0,
    gamma_val=0.9, lambda_reg=8e-3,
    num_grid_points=400, hull_expand_factor=1.8
)

# 3) weights at your query point(s): w = Bᵀ k
w = (B.T @ pre["k_sa"])   # shape: (m, n_queries)

# 4) (optional) choose plots — each is its own function
cfg = build_plot_config(
    lr=1e-3, fixed_point_constraint=True, FP_penalty_lambda=1e2,
    Sum_one_W=False, NonNeg_W=False, mass_anchor_lambda=10., target_mass=1.,
    num_steps=5000, nu=1.5, length_scale=1.0, sigma_k=1.0, gamma_val=0.9,
    num_grid_points=400, hull_expand_factor=1.8,
    lambda_reg=1e-6, bandwidth=0.5, lambda_rec=1e-3, method="song",
    state_dim=s0.shape[1], reward_dim=r.shape[1], action_dim=a0.shape[1],
    s_star=s_star, a_star=a_star, target_policy=target_p_choice,
)
beta, Zg = recover_joint_beta(B=B, k_sa=pre["k_sa"], Z_grid=pre["Z_grid"], Phi=pre["Phi"], K_sa=pre["K_sa"], config=cfg)
fz, grid = compute_marginals_from_beta(beta_full=beta, Z_grid=Zg, config=cfg)
plot_densities(fz=fz, Z_true_tensor=None, r_obs=r, Z_grid=Zg, grid_dict=grid, outdir="./plots/")
