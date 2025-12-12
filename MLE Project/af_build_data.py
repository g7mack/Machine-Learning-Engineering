
import numpy as np
import torch
import airfrans as af
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.nn import knn_graph


# Helper: build node features
def build_node_features_full(sim, eps=1e-12):
    """
    builder that returns X,pos,y and masks/aux arrays.
    """
    pos = np.asarray(sim.position).astype(np.float32)                      # (N,2)
    sdf = np.squeeze(np.asarray(sim.sdf)).astype(np.float32).reshape(-1,1) # (N,1)
    surface_flag = np.asarray(sim.surface).astype(np.float32).reshape(-1,1) # (N,1)
    normals = np.asarray(sim.normals).astype(np.float32)                   # (N,2) (zero off-surface)
    inlet_vel_field = np.asarray(sim.input_velocity).astype(np.float32)    # (N,2)
    inlet_vel_scalar = np.asarray(sim.inlet_velocity)                      

    # y target
    y = np.asarray(sim.velocity).astype(np.float32)  # (N,2)

    # surface mask (boolean)
    surface_mask = np.asarray(sim.surface).astype(bool)

    # inlet mask: prefer per-node inlet field if it has shape (N,2)
    inlet_mask = np.zeros(pos.shape[0], dtype=bool)
    try:
        if inlet_vel_field is not None and inlet_vel_field.ndim == 2 and inlet_vel_field.shape[0] == pos.shape[0]:
            # use per-node non-zero velocity as indicator of inlet nodes
            norms = np.linalg.norm(inlet_vel_field, axis=1)
            inlet_mask = norms > (eps * 10)
        else:
            # no per-node inlet field; do not attempt to synthesize a mask from a scalar
            inlet_mask = np.zeros(pos.shape[0], dtype=bool)
    except Exception:
        inlet_mask = np.zeros(pos.shape[0], dtype=bool)

    # build features: pos (2), sdf(1), surface_flag(1), normals(2), inlet_field(2), inlet_vel_mag(1), aoa(1)
    aoa_col = np.full((pos.shape[0], 1), float(sim.angle_of_attack), dtype=np.float32)
    # ensure inlet_vel_field shape: if single vector, repeat it per node (but note inlet_mask will be False)
    if inlet_vel_field.ndim == 1 or (inlet_vel_field.ndim == 2 and inlet_vel_field.shape[0] != pos.shape[0]):
        # create per-node field by repeating
        try:
            vec = np.asarray(inlet_vel_field).reshape(-1)
            if vec.size == pos.shape[1]: 
                inlet_vel_field = np.tile(vec, (pos.shape[0], 1)).astype(np.float32)
            else:
                inlet_vel_field = np.zeros((pos.shape[0], pos.shape[1]), dtype=np.float32)
        except Exception:
            inlet_vel_field = np.zeros((pos.shape[0], pos.shape[1]), dtype=np.float32)

    inlet_vel_mag = np.linalg.norm(inlet_vel_field, axis=1, keepdims=True).astype(np.float32)

    X = np.concatenate([pos, sdf, surface_flag, normals, inlet_vel_field, inlet_vel_mag, aoa_col], axis=1)

    return {
        "X": X,
        "pos": pos,
        "y": y,
        "surface_mask": surface_mask,
        "inlet_mask": inlet_mask,
        "normals": normals,
        "inlet_velocity_field": inlet_vel_field
    }


# Dataset with masks and normals
class AirfransGraphDatasetFull(PyGDataset):
    def __init__(self, root, sim_names, T=298.15, k=12, normalize=True, X_mean=None, X_std=None, Y_mean=None, Y_std=None, transform=None):
        super().__init__(root=None, transform=transform)
        self.ds_root = root
        self.sim_names = sim_names
        self.T = T
        self.k = k
        self.normalize = normalize

        # compute normalization stats across train sims
        all_feats = []
        if self.normalize:

            if X_mean is not None and X_std is not None:
                # use precomputed training stats
                self.X_mean = X_mean
                self.X_std = X_std

            else:
                for name in self.sim_names:
                    sim = af.Simulation(root=self.ds_root, name=name, T=self.T)
                    d = build_node_features_full(sim)
                    all_feats.append(d["X"])
                all_feats_cat = np.concatenate(all_feats, axis=0)
                self.X_mean = all_feats_cat.mean(axis=0, keepdims=True).astype(np.float32)
                self.X_std  = all_feats_cat.std(axis=0, keepdims=True).astype(np.float32)

                # clamp tiny stds to safe floor
                min_std = 1e-2
                self.X_std = np.maximum(self.X_std, min_std).astype(np.float32)

        # Compute Y normalization stats
        # ----------------------------
        all_targets = []
        if self.normalize:

            if Y_mean is not None and Y_std is not None:
                self.Y_mean = Y_mean
                self.Y_std = Y_std
            else:
                for name in self.sim_names:
                    sim = af.Simulation(root=self.ds_root, name=name, T=self.T)
                    d = build_node_features_full(sim)
                    all_targets.append(d["y"])
                all_targets_cat = np.concatenate(all_targets, axis=0)

                self.Y_mean = all_targets_cat.mean(axis=0, keepdims=True).astype(np.float32)
                self.Y_std  = all_targets_cat.std(axis=0, keepdims=True).astype(np.float32)
                self.Y_std  = np.maximum(self.Y_std, 1e-6).astype(np.float32)


        # build PyG Data objects
        self.graphs = []
        for name in self.sim_names:
            sim = af.Simulation(root=self.ds_root, name=name, T=self.T)
            d = build_node_features_full(sim)
            X = d["X"]
            pos = d["pos"]
            y = d["y"]
            surface_mask = d["surface_mask"]
            inlet_mask = d["inlet_mask"]
            normals = d["normals"]
            inlet_velocity_field = d["inlet_velocity_field"]

            if self.normalize:
                X = (X - self.X_mean) / self.X_std
                y = (y - self.Y_mean) / self.Y_std


            # convert to tensors
            x_tensor = torch.tensor(X, dtype=torch.float32)
            pos_tensor = torch.tensor(pos, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            surface_mask_t = torch.tensor(surface_mask, dtype=torch.bool)
            inlet_mask_t = torch.tensor(inlet_mask, dtype=torch.bool)
            normals_t = torch.tensor(normals, dtype=torch.float32)
            inlet_vel_field_t = torch.tensor(inlet_velocity_field, dtype=torch.float32)

            # knn edges
            edge_index = knn_graph(pos_tensor, k=self.k, loop=False)

            # edge attributes: rel_pos and dist
            row, col = edge_index
            rel_pos = pos_tensor[col] - pos_tensor[row]
            dist = rel_pos.norm(dim=-1, keepdim=True)
            edge_attr = torch.cat([rel_pos, dist], dim=1)

            data = Data(x=x_tensor, pos=pos_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
            data.surface_mask = surface_mask_t
            # attach inlet mask only if any True or shape good
            data.inlet_mask = inlet_mask_t if inlet_mask_t.shape[0] == x_tensor.shape[0] else torch.zeros(x_tensor.shape[0], dtype=torch.bool)
            data.normals = normals_t
            # attach inlet_velocity_field but ensure shape correct
            if inlet_vel_field_t.shape[0] == x_tensor.shape[0] and inlet_vel_field_t.shape[1] == x_tensor.shape[1] or inlet_vel_field_t.shape[1] in (2,3):
                data.inlet_velocity_field = inlet_vel_field_t
            else:
                data.inlet_velocity_field = torch.zeros((x_tensor.shape[0], inlet_vel_field_t.shape[1] if inlet_vel_field_t.ndim>1 else 2), dtype=torch.float32)

            data.sim_name = name

            self.graphs.append(data)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]