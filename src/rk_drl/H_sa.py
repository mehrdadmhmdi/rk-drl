import torch
from .matern_kernel import matern_kernel

def compute_H_rows(i_batch, Gamma_sa, gamma, R, Z, kernel):
    """
    For each i in i_batch (where i < m), compute H[i, :] of length m:

      H[i,j] = sum_{p=0..(n-1)}  Gamma_sa[p] * k( R[p], Z[i] - gamma * Z[j] ).

    Returns: (i_batch, chunk_result) with chunk_result.shape = (len(i_batch), m).
    """
    # Shapes
    nR = R.shape[0]          # n
    nZ = Z.shape[0]          # m
    d  = R.shape[1]
    device, dtype = Z.device, Z.dtype

    # ensure tensors on same device/dtype
    if not isinstance(i_batch, torch.Tensor):
        i_batch_t = torch.as_tensor(i_batch, device=device, dtype=torch.long)
    else:
        i_batch_t = i_batch.to(device=device, dtype=torch.long)

    Gamma_sa  = Gamma_sa.to(device=device, dtype=dtype).reshape(-1)     # (n,)
    R         = R.to(device=device, dtype=dtype)                         # (n,d)
    Z         = Z.to(device=device, dtype=dtype)                         # (m,d)
    gamma_t   = torch.as_tensor(gamma, device=device, dtype=dtype)       # scalar

    b = i_batch_t.numel()  # batch length
    chunk_result = torch.zeros((b, nZ), device=device, dtype=dtype)

    # shifts: (b, m, d) with shifts[b_i, j, :] = Z[i_batch[b_i]] - gamma * Z[j]
    Zi      = Z.index_select(dim=0, index=i_batch_t)                     # (b,d)
    shifts  = Zi.unsqueeze(1) - gamma_t * Z.unsqueeze(0)                 # (b,m,d)
    shifts2 = shifts.reshape(b * nZ, d)                                   # (b*m, d)

    # kernel matrix between R (n,d) and shifts2 (b*m,d): (n, b*m)
    K_full = kernel(R, shifts2)                                          # (n, b*m)

    # row_data for all i in batch at once: (1,n) @ (n, b*m) -> (1, b*m)
    row_data_flat = (Gamma_sa.unsqueeze(0) @ K_full)                     # (1, b*m)
    row_data = row_data_flat.reshape(b, nZ)                               # (b, m)

    chunk_result.copy_(row_data)
    return i_batch, chunk_result


def H_sa(Gamma_sa, gamma, R, Z, nu, length_scale, sigma, batch_size=10):
    """
    Build H of shape (m,m) where:
      R: shape (n, d)
      Z: shape (m, d)
    H[i,j] = sum_{p=0..n-1} Gamma_sa[p] * k( R[p],  Z[i] - gamma*Z[j] ).

    - Chunked over i in [0..(m-1)]
    - Time complexity: O(n * m^2)
    - Memory-friendly row-wise batch approach
    """
    # ensure Torch tensors
    R        = torch.as_tensor(R)
    Z        = torch.as_tensor(Z)
    Gamma_sa = torch.as_tensor(Gamma_sa).reshape(-1)

    device, dtype = Z.device, Z.dtype
    nR = R.shape[0]
    d  = R.shape[1]
    mZ = Z.shape[0]  # m

    # kernel closure to match original call pattern kernel(R, shift)
    kernel = lambda x1, x2: matern_kernel(x1, x2, length_scale=length_scale, nu=nu, sigma=sigma)

    # chunk row indices
    row_indices = torch.arange(mZ, device=device)
    chunks = [row_indices[i: i + batch_size] for i in range(0, mZ, batch_size)]

    # assemble final H (on device)
    H = torch.zeros((mZ, mZ), device=device, dtype=dtype)

    for i_batch in chunks:
        _, chunk_mat = compute_H_rows(i_batch, Gamma_sa, gamma, R, Z, kernel)  # (len, m)
        H.index_copy_(0, i_batch.to(dtype=torch.long), chunk_mat)

    # Optional checks (Torch)
    is_symmetric = torch.allclose(H, H.T, atol=1e-8)
    evals = torch.linalg.eigvalsh((H + H.T) * 0.5)  # symmetrize for safety
    is_spd = bool(torch.all(evals >= -1e-8))
    print(f"H is SPD: {is_spd}")
    print(f"H is symmetric: {is_symmetric}")
    return H

#=================================
##  usage Example
#=================================
# if __name__ == "__main__":
#     torch.manual_seed(0)
#
#     # dimensions
#     n = 5   # |R|
#     m = 6   # |Z|
#     d = 2   # dimension
#
#     # toy data
#     R = torch.randn(n, d)
#     Z = torch.randn(m, d)
#     Gamma_sa = torch.rand(n)           # weights
#     gamma = 0.9
#
#     # kernel parameters
#     nu = 1.5
#     length_scale = 1.0
#     sigma = 1.0
#
#     # build H
#     H = H_sa(Gamma_sa, gamma, R, Z, nu, length_scale, sigma, batch_size=2)
#
#     print("H shape:", H.shape)
#     print("H (rounded):\n", H.cpu().numpy().round(3))
