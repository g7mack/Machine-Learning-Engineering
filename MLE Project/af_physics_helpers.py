
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


# Physics helpers
def compute_divergence_from_edges(x_pred, edge_index, edge_attr, eps=1e-12):
    """
    Args:
      x_pred: [N,2] predicted velocity (torch.Tensor)
      edge_index: [2, E] (torch.LongTensor)
      edge_attr: [E, K] where first two cols are rel_pos (dx,dy) and last col may be dist.
      eps: small numeric guard

    Returns:
      per_node_div: [N] tensor (units: velocity/length)
    Notes:
      - returns an approximation of the divergence per-node using directional derivatives
        projected on edges and aggregated by target node (incoming edges).
      - for empty/missing edge_attr returns zeros
    """
    device = x_pred.device
    N = x_pred.size(0)

    # safety checks
    if edge_attr is None:
        return torch.zeros(N, device=device)

    if edge_attr.dim() != 2 or edge_attr.size(0) == 0:
        return torch.zeros(N, device=device)

    if edge_attr.size(1) < 2:
        return torch.zeros(N, device=device)

    row, col = edge_index  # source, target

    rel = edge_attr[:, :2].to(device)          # (E,2)
    # squared length of edge (E,)
    r2 = (rel.pow(2).sum(dim=1) + eps).unsqueeze(1)  # (E,1)

    # velocity on edge endpoints
    u_i = x_pred[row]   # (E,2)
    u_j = x_pred[col]   # (E,2)
    du = u_j - u_i      # (E,2)

    # directional derivative along edge (scalar per edge)
    # (du · rel) / r^2; division guard included via r2
    proj = ( (du * rel).sum(dim=1, keepdim=True) ) / r2   # (E,1)

    # aggregate (mean) per target node
    per_node = scatter_mean(proj.squeeze(1), col, dim=0, dim_size=N)  # (N,)

    return per_node  # units: velocity / length


def bc_losses(x_pred, x_true, surface_mask=None, normals=None, inlet_mask=None, inlet_true=None):
    """
    Compute boundary condition related losses safely.
    Returns (bc_full_mse, bc_normal_mse, inlet_mse) all torch scalars on same device as x_pred.

    Notes:
      - surface_mask: boolean tensor shape [N] indicating wall nodes (True on wall)
      - normals: tensor [N,2] or [N,3] containing normals at nodes (zeros where no normal)
      - inlet_mask: boolean tensor shape [N] marking inlet nodes (True at inlet)
      - inlet_true: tensor [N,2] containing true inlet velocity on nodes (or None)
    """
    device = x_pred.device
    bc_full   = torch.tensor(0., dtype=x_pred.dtype, device=device)
    bc_normal = torch.tensor(0., dtype=x_pred.dtype, device=device)
    inlet_mse = torch.tensor(0., dtype=x_pred.dtype, device=device)

    # --- wall/surface BC ---
    if isinstance(surface_mask, torch.Tensor) and surface_mask.numel() == x_pred.size(0) and surface_mask.any():
        idx = torch.nonzero(surface_mask, as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            pred_surf = x_pred[idx]
            if x_true is not None:
                true_surf = x_true[idx]
                bc_full = F.mse_loss(pred_surf, true_surf)
            else:
                # push to zero if no explicit truth
                bc_full = torch.mean(pred_surf.pow(2))

            if isinstance(normals, torch.Tensor) and normals.size(0) == x_pred.size(0):
                n_surf = normals[idx]
                n_norm = n_surf / (n_surf.norm(dim=1, keepdim=True) + 1e-8)
                n_dot = (pred_surf * n_norm).sum(dim=1)
                bc_normal = torch.mean(n_dot.pow(2))

    # --- inlet BC ---
    # require: inlet_mask hits and inlet_true provided with same shape
    if (
        isinstance(inlet_mask, torch.Tensor)
        and inlet_mask.numel() == x_pred.size(0)
        and inlet_mask.any()
        and inlet_true is not None
        and inlet_true.shape == x_pred.shape
    ):
        idx_i = torch.nonzero(inlet_mask, as_tuple=False).squeeze(1)
        if idx_i.numel() > 0:
            inlet_mse = F.mse_loss(x_pred[idx_i], inlet_true[idx_i])

    return bc_full, bc_normal, inlet_mse