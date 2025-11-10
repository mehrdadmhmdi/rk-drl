import torch

def Gamma_sa(K_sa: torch.Tensor, k_sa: torch.Tensor, lambda_reg: float):
    """
    Calculation of Gamma_sa = (K + n*λ I)^(-1) k
    Args:
        K_sa (torch.Tensor): (n, n)
        k_sa (torch.Tensor): (n,)
    Returns:
         Gamma_sa (torch.Tensor): (n, )
    """
    n = K_sa.size(0)
    reg_mat = K_sa + (lambda_reg * n) * torch.eye(n, device=K_sa.device, dtype=K_sa.dtype)

    # RHS to 2D
    k_is_vec = (k_sa.ndim == 1)
    rhs      = k_sa.unsqueeze(-1) if k_is_vec else k_sa

    # Cholesky with fallback jitter
    try:
        L = torch.linalg.cholesky(reg_mat)
    except RuntimeError:
        jitter = 1e-6 * reg_mat.diagonal().mean().clamp(min=1.0)
        L = torch.linalg.cholesky(reg_mat + jitter * torch.eye(n, device=reg_mat.device, dtype=reg_mat.dtype))

    x = torch.cholesky_solve(rhs, L)
    return x.squeeze(-1) if k_is_vec else x
