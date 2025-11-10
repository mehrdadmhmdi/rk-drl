import torch
import numpy as np
from scipy.spatial import ConvexHull

class ZGrid:
    @staticmethod
    def _kmeans_torch(r: torch.Tensor, n_clusters: int, max_iter: int = 100, tol: float = 1e-4) -> torch.Tensor:
        device, dtype = r.device, r.dtype
        N, d = r.shape
        idx = torch.randperm(N, device=device)[:n_clusters]
        centers = r[idx].clone()
        for _ in range(max_iter):
            dists  = torch.cdist(r, centers)          # (N,K)
            labels = torch.argmin(dists, dim=1)       # (N,)
            new_centers = torch.zeros_like(centers)
            counts = torch.bincount(labels, minlength=n_clusters).clamp_min(1).to(dtype)
            new_centers.index_add_(0, labels, r)
            new_centers = new_centers / counts.unsqueeze(1)
            shift = (new_centers - centers).norm(dim=1).max()
            centers = new_centers
            if shift <= tol: break
        return centers

    @staticmethod
    def _convex_hull_vertices_numpy(points_t: torch.Tensor) -> torch.Tensor:
        """
        Convert to NumPy, get hull vertices via SciPy, return Torch indices on original device.
        """
        device = points_t.device
        pts_np = points_t.detach().cpu().numpy()   # (K,D)
        hull = ConvexHull(pts_np)
        verts_np = hull.vertices                   # np indices
        return torch.as_tensor(verts_np, dtype=torch.long, device=device)

    @staticmethod
    def Z_kmeans(r: torch.Tensor, n_clusters: int, constant_factor: float) -> torch.Tensor:
        """
        Cluster reward samples and expand hull vertices radially.

        Parameters:
            - r: (N, D) torch tensor with observed (or finite discounted) r.
            - n_clusters: number of k-means clusters (atoms).
            - constant_factor: expansion factor > 0 for hull points.

        Returns:
            - expanded_centers: (n_clusters, D) torch tensor.
        """
        if r.ndim != 2:
            raise ValueError("r must be a 2D tensor of shape (N, D).")
        if constant_factor <= 0:
            raise ValueError("constant_factor must be > 0.")

        device, dtype = r.device, r.dtype

        # 1) K-means (Torch)
        centers = ZGrid._kmeans_torch(r, n_clusters=n_clusters).to(device=device, dtype=dtype)  # (K,D)

        # 2) Global centroid
        mu = centers.mean(dim=0)  # (D,)

        # 3) Convex hull vertices via SciPy (NumPy hop)
        vertices = ZGrid._convex_hull_vertices_numpy(centers)  # (H,)

        # 4) Radial expansion
        expanded = centers.clone()
        expanded[vertices] = mu + constant_factor * (centers[vertices] - mu)
        return expanded


##=========================
##### usage #####
# Z_grids = ZGrid.Z_kmeans(r, n_clusters=num_grid_points, constant_factor=1.8)
