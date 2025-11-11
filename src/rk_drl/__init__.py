# src/rk_drl/__init__.py
import os
os.environ.setdefault("VISPY_GL_BACKEND", "egl")
os.environ.setdefault("VISPY_USE_APP", "headless")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8"
)

from .RK_DRL import RK_DRL
from .api import (
    run_RKDRL, build_plot_config,
    plot_bellman_error, plot_total_loss,
    recover_joint_beta, compute_marginals_from_beta, plot_densities,
    compute_L2_marginal_error, mean_embedding_all,
    plot_bland_altman, plot_quantile_calibration,
    plot_error_vs_distance_from_mode, plot_operator_check_2d,
    plot_error_heatmap, plot_statistics, save_weights_and_grid,
)

__all__ = [
    "RK_DRL", "run_RKDRL", "build_plot_config",
    "plot_bellman_error", "plot_total_loss",
    "recover_joint_beta", "compute_marginals_from_beta", "plot_densities",
    "compute_L2_marginal_error", "mean_embedding_all",
    "plot_bland_altman", "plot_quantile_calibration",
    "plot_error_vs_distance_from_mode", "plot_operator_check_2d",
    "plot_error_heatmap", "plot_statistics", "save_weights_and_grid",
]
