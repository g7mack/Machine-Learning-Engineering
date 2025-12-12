
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import airfrans as af
from torch_geometric.nn import SAGEConv
from torch_scatter import scatter_mean
from torch import nn
from torch_geometric.loader import DataLoader as PyGDataLoader

from af_physics_helpers import compute_divergence_from_edges, bc_losses


# GNN model
class VelocityGNN_Upgraded(nn.Module):
    def __init__(self, in_channels, edge_attr_dim=3, hidden=128, num_layers=4, dropout=0.1, use_gat=False, heads=4):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden)
        )
        self.edge_attr_dim = edge_attr_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gat = use_gat
        self.heads = heads

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden + edge_attr_dim
            self.layers.append(SAGEConv(in_dim, hidden))
            self.norms.append(nn.LayerNorm(hidden))

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 2)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_channels]
        h = self.node_encoder(x)  # [N, hidden]
        row, col = edge_index

        for i, conv in enumerate(self.layers):
            # aggregate edge_attr for incoming edges
            agg_edge = scatter_mean(edge_attr, col, dim=0, dim_size=h.size(0))  # [N, edge_attr_dim]
            h_cat = torch.cat([h, agg_edge], dim=1)
            out = conv(h_cat, edge_index)
            if out.shape == h.shape:
                h = h + out
            else:
                h = out
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.readout(h)
        return out
    

# Training loop with physics losses
def train_gnn_physics(
    train_dataset,
    val_dataset,
    epochs=60,
    batch_size=1,
    lr=1e-4,                 
    weight_decay=1e-4,
    lambda_bc=0.1,
    lambda_norm=0.1,
    lambda_inlet=0.1,
    lambda_div=5e-4,         # small to balance weights
    patience=8,
    lr_patience=4,
    factor=0.5,
    device=None,
    ramp_div_epochs=10,
    max_grad_norm=1.0,       # gradient clipping
    diag_batches=2           # how many batches to print diagnostics for
):
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # enforce float32 on normalization stats if present
    if hasattr(train_dataset, "X_mean") and train_dataset.X_mean is not None:
        train_dataset.X_mean = train_dataset.X_mean.astype(np.float32) if isinstance(train_dataset.X_mean, np.ndarray) else train_dataset.X_mean

    if hasattr(train_dataset, "X_std") and train_dataset.X_std is not None:
        # guard tiny stds
        if isinstance(train_dataset.X_std, np.ndarray):
            train_dataset.X_std = np.maximum(train_dataset.X_std, 1e-6).astype(np.float32)
        else:
            # torch or tensor-like: leave it but will clamp when used
            pass

    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=1, shuffle=False)

    in_dim = train_dataset.graphs[0].x.shape[1]
    edge_attr_dim = train_dataset.graphs[0].edge_attr.shape[1]

    model = VelocityGNN_Upgraded(in_channels=in_dim, edge_attr_dim=edge_attr_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=factor, patience=lr_patience, verbose=True)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": [], "sup_loss": [], "bc_loss": [], "div_loss": []}
    global_eps = 1e-12

    # tiny helper
    def safe_num(x):
        if torch.is_tensor(x):
            return float(torch.nan_to_num(x, posinf=1e30, neginf=-1e30).cpu().item())
        else:
            try:
                return float(np.nan_to_num(x, posinf=1e30, neginf=-1e30))
            except Exception:
                return x

    printed_diag = False

    for epoch in range(epochs):
        model.train()
        running = 0.0
        total_nodes = 0
        sup_running = 0.0
        bc_running = 0.0
        div_running = 0.0

        # linear ramp for divergence weight
        if ramp_div_epochs and epoch < ramp_div_epochs:
            w_div = lambda_div * (epoch + 1) / ramp_div_epochs
        else:
            w_div = lambda_div

        for bidx, batch in enumerate(train_loader):
            # ensure float32 and device
            batch = batch.to(device)
            # cast important tensors to float32 to avoid dtype mixing
            batch.x = batch.x.float()
            batch.y = batch.y.float()
            if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr.float()
            if hasattr(batch, "inlet_velocity_field") and batch.inlet_velocity_field is not None:
                batch.inlet_velocity_field = batch.inlet_velocity_field.float()

            opt.zero_grad()
            pred = model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None))  # (N,2)

            # compute U_ref robustly and clamp to safe range
            U_ref = 1.0
            if hasattr(batch, "inlet_velocity_field") and getattr(batch, "inlet_velocity_field") is not None:
                inlet_field = batch.inlet_velocity_field
                if inlet_field.shape == pred.shape:
                    if hasattr(batch, "inlet_mask") and batch.inlet_mask is not None and batch.inlet_mask.any():
                        idx_i = torch.nonzero(batch.inlet_mask, as_tuple=False).squeeze(1)
                        if idx_i.numel() > 0:
                            U_ref = float(inlet_field[idx_i].norm(dim=1).mean().item())
                        else:
                            U_ref = float(inlet_field.norm(dim=1).mean().item())
                    else:
                        U_ref = float(inlet_field.norm(dim=1).mean().item())
                else:
                    # single vector
                    try:
                        U_ref = float(inlet_field.reshape(-1,2).norm(dim=1).mean().item())
                    except Exception:
                        U_ref = 1.0
            else:
                if batch.y.numel() > 0:
                    U_ref = float(batch.y.norm(dim=1).max().item())
                    if U_ref <= 0:
                        U_ref = float(batch.y.norm(dim=1).mean().item())
            # final guard
            if not (isinstance(U_ref, float) and np.isfinite(U_ref) and U_ref > 1e-6):
                U_ref = 1.0
            U_ref = max(U_ref, 1e-3)   # ensure not too small

            # supervised loss scale
            sup_loss = loss_fn(pred, batch.y)

            # bc losses
            bc_full, bc_normal, inlet_mse = bc_losses(
                pred / U_ref,
                batch.y / U_ref if batch.y is not None else None,
                surface_mask=getattr(batch, "surface_mask", None),
                normals=getattr(batch, "normals", None),
                inlet_mask=getattr(batch, "inlet_mask", None),
                inlet_true=(getattr(batch, "inlet_velocity_field", None) / U_ref) if getattr(batch, "inlet_velocity_field", None) is not None and getattr(batch, "inlet_velocity_field", None).shape == pred.shape else getattr(batch, "inlet_velocity_field", None)
            )

            # divergence loss
            edge_attr = getattr(batch, "edge_attr", None)
            if (edge_attr is None) or (edge_attr.numel() == 0):
                div_loss = torch.tensor(0.0, device=device)
            else:
                per_node_div = compute_divergence_from_edges(pred, batch.edge_index, edge_attr, eps=global_eps)
                # fluid mask prefer surface_mask
                if getattr(batch, "surface_mask", None) is not None:
                    fluid_mask_bool = ~batch.surface_mask
                    fluid_mask = fluid_mask_bool.to(dtype=per_node_div.dtype)
                elif getattr(batch, "sdf", None) is not None:
                    sdf = batch.sdf.float()
                    fluid_mask_bool = (sdf.squeeze(-1) > 0)
                    fluid_mask = fluid_mask_bool.to(dtype=per_node_div.dtype)
                else:
                    fluid_mask = torch.ones_like(per_node_div, dtype=per_node_div.dtype)

                # mean edge length nondim
                if edge_attr.size(1) >= 3:
                    dist = edge_attr[:, -1].clamp(min=global_eps)
                    mean_edge_len = float(dist.mean().item())
                else:
                    rel = edge_attr[:, :2]
                    mean_edge_len = float((rel.norm(dim=1) + global_eps).mean().item())
                mean_edge_len = max(mean_edge_len, 1e-6)

                div_scaled = (mean_edge_len * per_node_div) / (U_ref + 1e-12)
                denom = fluid_mask.sum().clamp(min=1.0)
                div_loss = ( (div_scaled.pow(2) * fluid_mask).sum() / denom )

            total_loss = sup_loss + lambda_bc * bc_full + lambda_norm * bc_normal + lambda_inlet * inlet_mse + w_div * div_loss

            # check NaN/INF
            if not torch.isfinite(total_loss):
                print("Non-finite total_loss detected — aborting. Inspect batch and stats.")
                print("sup_loss, bc_full, bc_normal, inlet_mse, div_loss:", safe_num(sup_loss), safe_num(bc_full), safe_num(bc_normal), safe_num(inlet_mse), safe_num(div_loss))
                raise RuntimeError("Non-finite loss; aborting to prevent wasted compute.")

            total_loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            opt.step()

            running += float(total_loss.item()) * batch.num_nodes
            total_nodes += batch.num_nodes
            sup_running += float(sup_loss.item()) * batch.num_nodes
            bc_running += (float(bc_full.item()) + float(bc_normal.item()) + float(inlet_mse.item())) * batch.num_nodes
            div_running += float(div_loss.item()) * batch.num_nodes

            # print first-batch-only diagnostics (once)
            if not printed_diag:
                printed_diag = True

        # epoch metrics
        train_loss = running / max(1, total_nodes)
        sup_loss_epoch = sup_running / max(1, total_nodes)
        bc_loss_epoch = bc_running / max(1, total_nodes)
        div_loss_epoch = div_running / max(1, total_nodes)

        # validation
        model.eval()
        vrunning = 0.0; vtotal = 0; vsup = 0.0; vbc = 0.0; vdiv = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch.x = batch.x.float(); batch.y = batch.y.float()
                if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
                    batch.edge_attr = batch.edge_attr.float()
                pred = model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None))
                # compute U_ref safely (same code as training)
                U_ref = 1.0
                if hasattr(batch, "inlet_velocity_field") and getattr(batch, "inlet_velocity_field") is not None:
                    inlet_field = batch.inlet_velocity_field
                    try:
                        if inlet_field.shape == pred.shape and hasattr(batch, "inlet_mask") and batch.inlet_mask is not None and batch.inlet_mask.any():
                            idx_i = torch.nonzero(batch.inlet_mask, as_tuple=False).squeeze(1)
                            if idx_i.numel() > 0:
                                U_ref = float(inlet_field[idx_i].norm(dim=1).mean().item())
                            else:
                                U_ref = float(inlet_field.norm(dim=1).mean().item())
                        else:
                            U_ref = float(inlet_field.reshape(-1,2).norm(dim=1).mean().item())
                    except Exception:
                        U_ref = 1.0
                else:
                    if batch.y.numel() > 0:
                        U_ref = float(batch.y.norm(dim=1).max().item())
                        if U_ref <= 0:
                            U_ref = float(batch.y.norm(dim=1).mean().item())
                if not (isinstance(U_ref, float) and np.isfinite(U_ref) and U_ref > 1e-6):
                    U_ref = 1.0
                U_ref = max(U_ref, 1e-3)

                sup_loss = loss_fn(pred, batch.y)
                bc_full, bc_normal, inlet_mse = bc_losses(
                    pred / U_ref,
                    batch.y / U_ref if batch.y is not None else None,
                    surface_mask=getattr(batch, "surface_mask", None),
                    normals=getattr(batch, "normals", None),
                    inlet_mask=getattr(batch, "inlet_mask", None),
                    inlet_true=(getattr(batch, "inlet_velocity_field", None) / U_ref) if getattr(batch, "inlet_velocity_field", None) is not None and getattr(batch, "inlet_velocity_field", None).shape == pred.shape else getattr(batch, "inlet_velocity_field", None)
                )

                # divergence (same safe routine)
                edge_attr = getattr(batch, "edge_attr", None)
                if (edge_attr is None) or (edge_attr.numel() == 0):
                    div_loss = torch.tensor(0.0, device=device)
                else:
                    per_node_div = compute_divergence_from_edges(pred, batch.edge_index, edge_attr, eps=global_eps)
                    if getattr(batch, "surface_mask", None) is not None:
                        fluid_mask_bool = ~batch.surface_mask
                        fluid_mask = fluid_mask_bool.to(dtype=per_node_div.dtype)
                    elif getattr(batch, "sdf", None) is not None:
                        sdf = batch.sdf
                        fluid_mask_bool = (sdf.squeeze(-1) > 0)
                        fluid_mask = fluid_mask_bool.to(dtype=per_node_div.dtype)
                    else:
                        fluid_mask = torch.ones_like(per_node_div, dtype=per_node_div.dtype)

                    if edge_attr.size(1) >= 3:
                        dist = edge_attr[:, -1].clamp(min=global_eps)
                        mean_edge_len = dist.mean().item()
                    else:
                        rel = edge_attr[:, :2]
                        dist = (rel.norm(dim=1) + global_eps)
                        mean_edge_len = dist.mean().item()

                    mean_edge_len = max(mean_edge_len, 1e-12)
                    div_scaled = (mean_edge_len * per_node_div) / (U_ref + 1e-12)
                    denom = fluid_mask.sum().clamp(min=1.0)
                    div_loss = ( (div_scaled.pow(2) * fluid_mask).sum() / denom )

                total_loss = sup_loss + lambda_bc * bc_full + lambda_norm * bc_normal + lambda_inlet * inlet_mse + w_div * div_loss

                vrunning += float(total_loss.item()) * batch.num_nodes
                vtotal += batch.num_nodes
                vsup += float(sup_loss.item()) * batch.num_nodes
                vbc += (float(bc_full.item()) + float(bc_normal.item()) + float(inlet_mse.item())) * batch.num_nodes
                vdiv += float(div_loss.item()) * batch.num_nodes

        val_loss = vrunning / max(1, vtotal)
        val_sup = vsup / max(1, vtotal)
        val_bc = vbc / max(1, vtotal)
        val_div = vdiv / max(1, vtotal)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["sup_loss"].append(sup_loss_epoch)
        history["bc_loss"].append(bc_loss_epoch)
        history["div_loss"].append(div_loss_epoch)

        print(f"Epoch {epoch:02d} | train={train_loss:.6f} | val={val_loss:.6f} | sup={sup_loss_epoch:.6f} bc={bc_loss_epoch:.6f} div={div_loss_epoch:.6f} | lr={opt.param_groups[0]['lr']:.2e}")

        scheduler.step(val_loss)

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history