import torch
import numpy as np
import time
import math
from typing import Optional
from .matern_kernel import matern_kernel


def compute_transformed_grid_pytorch(Z_grid, reward_set, gamma_val):
    # Z_grid: (m,d) -> (m,1,d), reward_set: (n,d) -> (1,n,d)
    return gamma_val * Z_grid.unsqueeze(1) + reward_set.unsqueeze(0)

##=======================================================================
def compute_G_pytorch_semivectorized(transformed, Gamma_sa, nu, length_scale, sigma=1.0):
    m, n, _ = transformed.shape
    G = torch.empty((m, m), device=transformed.device, dtype=transformed.dtype)
    for i in range(m):
        for j in range(i, m):
            Kij = matern_kernel(transformed[i], transformed[j], nu, length_scale, sigma)
            tmp = torch.mv(Kij, Gamma_sa)
            val = torch.dot(Gamma_sa, tmp)
            G[i, j] = G[j, i] = val
    _check_G_properties(G.cpu().numpy())
    return G
# -----------------------------------------------------------------------------

def compute_G_pytorch_fully_vectorized(transformed, Gamma_sa, nu, length_scale, sigma=1.0):
    m, n, d = transformed.shape
    flat = transformed.reshape(m*n, d)
    Kgiant = matern_kernel(flat, flat, nu, length_scale, sigma)
    Kt = Kgiant.reshape(m, n, m, n)
    G = torch.einsum('u,iujv,v->ij', Gamma_sa, Kt, Gamma_sa)
    if transformed.device.type == "cuda":
        torch.cuda.synchronize()


    _check_G_properties(G.cpu().numpy())
    return G

##================================================
def compute_G_pytorch_batched(transformed:   torch.Tensor,Gamma_sa: torch.Tensor, nu: float,length_scale:  float,sigma:float = 1.0,block_i:int= 1,block_j:int= None) -> torch.Tensor:
    """
    Batched G-matrix computation in 2D blocks to limit peak memory:

      G[i,j] = sum_{u,v} Gamma_sa[u] * k(trans[i,u], trans[j,v]) * Gamma_sa[v]

    Defaults:
      block_i = 1
      block_j = 1000 * n
    """
    m, n, d = transformed.shape
    device, dtype = transformed.device, transformed.dtype

    # internal default for block_j
    if block_j is None:
        block_j = 1000 * n

    # flatten all (i,u) points once: (m*n, d)
    flat_all = transformed.reshape(m*n, d)

    # ensure Gamma_sa is 1-D
    Gamma1d = Gamma_sa.reshape(-1)        # (n,)

    # precompute full repeated Gamma for flattened columns
    Gamma_cols_full = Gamma1d.repeat(m)    # (m*n,)

    G = torch.zeros((m, m), device=device, dtype=dtype)

    # outer loop over rows
    for i0 in range(0, m, block_i):
        i1 = min(m, i0 + block_i)
        bi = i1 - i0

        # slice and flatten this Z-block
        blk = transformed[i0:i1].reshape(bi*n, d)   # (bi*n, d)
        Gamma_rows_blk = Gamma1d.repeat(bi)         # (bi*n,)

        # inner loop over flattened columns
        for j0 in range(0, m*n, block_j):
            j1 = min(m*n, j0 + block_j)
            bj = j1 - j0

            flat_cols_blk = flat_all[j0:j1]            # (bj, d)
            Gamma_cols_blk = Gamma_cols_full[j0:j1]    # (bj,)

            Kblk = matern_kernel(
                blk,               # (bi*n, d)
                flat_cols_blk,     # (bj, d)
                nu,
                length_scale,
                sigma
            )                                             # (bi*n, bj)
            # ------------------------------------------------------

            # weight by Gamma_rows_blk[u] * Kblk * Gamma_cols_blk[v]
            W = (Gamma_rows_blk.unsqueeze(1) * Kblk) * Gamma_cols_blk.unsqueeze(0)  # (bi*n, bj)

            # reshape to (bi, n, bjm, n) and sum over u,v dims
            assert bj % n == 0, "block_j must be a multiple of n"
            bjm    = bj // n
            G_block = W.view(bi, n, bjm, n).sum(dim=(1, 3))  # (bi, bjm)

            # map bj columns back to j-indices
            j_i0 = j0 // n
            G[i0:i1, j_i0:j_i0 + bjm] += G_block
            del Kblk,W
    _check_G_properties(G.detach().cpu().numpy())
    torch.cuda.empty_cache()
    return G



# --------------------- Utility ---------------------
def _check_G_properties(G):
    """
    Print symmetry and positive-(semi)definiteness diagnostics.
    """
    print("G is symmetric:", np.allclose(G, G.T, atol=1e-8))
    print("G is semi-positive definite:", np.all(np.linalg.eigvalsh(G) >= -1e-8))

# -----------------------------------------------------------------------------
# Section 4: example
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     print("\n example")
#     M, N, D = 100, 30, 10
#     gamma_val, NU, LS, SIG = 0.95, 4.5, 1.0, 1.0
#     print("M, N, D =", M, N, D , "\ngamma_val, NU, LS, SIG =", gamma_val, NU, LS, SIG )
#     # Generate random data
#     np.random.seed(0)
#     Z_np = np.random.rand(M, D)
#     R_np = np.random.rand(N, D)
#     Gamma_np = np.random.rand(N)
#
#
#     # 3) PyTorch semi-vectorized
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}\n")
#     dtype = torch.float64
#     Z_pt = torch.from_numpy(Z_np).to(device, dtype=dtype)
#     R_pt = torch.from_numpy(R_np).to(device, dtype=dtype)
#     G_pt = torch.from_numpy(Gamma_np).to(device, dtype=dtype)
#
#     t0 = time.time()
#     T_pt = compute_transformed_grid_pytorch(Z_pt, R_pt, gamma_val)
#     G3 = compute_G_pytorch_semivectorized(T_pt, G_pt, NU, LS, SIG)
#     torch.cuda.synchronize() if device.type=="cuda" else None
#     print(f"PyTorch semi-vec took {time.time()-t0:.3f}s\n")
#
#     # 4) PyTorch fully-vectorized
#     t0 = time.time()
#     T_pt = compute_transformed_grid_pytorch(Z_pt, R_pt, gamma_val)
#     G4 = compute_G_pytorch_fully_vectorized(T_pt, G_pt, NU, LS, SIG)
#     torch.cuda.synchronize() if device.type=="cuda" else None
#     print(f"PyTorch full-vec took {time.time()-t0:.3f}s\n")
#
#     # 5) compute_G_pytorch_batched
#     t0 = time.time()
#     T_pt = compute_transformed_grid_pytorch(Z_pt, R_pt, gamma_val)
#     G5 = compute_G_pytorch_batched(T_pt, G_pt, NU, LS, SIG)
#     torch.cuda.synchronize() if device.type=="cuda" else None
#     print(f"PyTorch full-vec took {time.time()-t0:.3f}s\n")
#
#     # print differences
#     print("np.max(abs(G1-G2))",np.max(abs(G1-G2)))
#     print("np.max(abs(G1-G3))",np.max(abs(G1-G3)))
#     print("np.max(abs(G1-G4))",np.max(abs(G1-G4)))
#     print("np.max(abs(G1-G5))",np.max(abs(G1-G5)))
#

