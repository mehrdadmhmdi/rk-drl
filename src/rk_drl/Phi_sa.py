import torch

def Phi_sa(K_sa_prime: torch.Tensor,
           Gamma_sa  : torch.Tensor,
           alpha     : torch.Tensor) -> torch.Tensor:
    """
    Torch version of Phi_RKHS:
      Phi = K_sa_prime @ (Gamma * (K_sa_prime @ alpha))

    Args:
        K_sa_prime : (n, n) tensor
        Gamma_sa: (n,) or (n,1) tensor
        alpha      : (n,) or (n,1) tensor

    Returns:
        Phi        : (n, 1) tensor
    """
    # to ensure column vectors
    if Gamma_sa.ndim == 1:
        Gamma_sa = Gamma_sa.unsqueeze(1)   # (n,1)
    if alpha.ndim    == 1:
        alpha    = alpha.unsqueeze(1)               # (n,1)

    inner    = K_sa_prime @ alpha                # (n,1)
    weighted = Gamma_sa * inner               # elementwise
    Phi      = K_sa_prime @ weighted             # (n,1)
    return Phi
