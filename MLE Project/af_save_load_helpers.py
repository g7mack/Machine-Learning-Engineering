import torch

from af_gnn_model import VelocityGNN_Upgraded

# Save / load helpers
def save_physics_checkpoint(path, model, history, dataset=None, optimizer=None):
    payload = {"model_state": model.state_dict(), "history": history}
    if dataset is not None:
        payload["X_mean"] = dataset.X_mean
        payload["X_std"] = dataset.X_std
        payload["in_channels"] = dataset.graphs[0].x.shape[1]
        payload["edge_attr_dim"] = dataset.graphs[0].edge_attr.shape[1]

        # Save y stats if available
        if hasattr(dataset, "Y_mean") and dataset.Y_mean is not None:
            payload["Y_mean"] = dataset.Y_mean
            payload["Y_std"]  = dataset.Y_std

    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)
    print(f"Saved checkpoint to {path}")

def load_physics_checkpoint(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = VelocityGNN_Upgraded(in_channels=ckpt["in_channels"], edge_attr_dim=ckpt["edge_attr_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt