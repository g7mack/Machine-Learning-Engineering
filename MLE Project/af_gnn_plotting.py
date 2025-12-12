
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from af_gnn_model import VelocityGNN_Upgraded
from af_build_data import AirfransGraphDatasetFull


# Load a trained model + normalization stats
def load_trained_model(path, device=None):
    """
    Load a trained VelocityGNN_Upgraded model checkpoint.
    Returns: model, checkpoint_dict
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)

    model = VelocityGNN_Upgraded(
        in_channels=ckpt["in_channels"],
        edge_attr_dim=ckpt["edge_attr_dim"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, ckpt


# Plot training history
def plot_training(history):
    train = history["train_loss"]
    val = history["val_loss"]

    plt.figure(figsize=(7,5))
    plt.plot(train, label="Train Loss")
    plt.plot(val, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training History")
    plt.grid(True)
    plt.legend()
    plt.show()


# Downsampling Utility
def downsample(pos, vel, factor=10):
    """
    Pick every 'factor'-th point for visualization.
    Works for both MLP arrays and GNN node arrays.
    """
    return pos[::factor], vel[::factor]


# Full GNN Plotting Suite
def plot_gnn_results(pos, vel_true, vel_pred, factor=10):
    """
    pos       : (N, 2) node positions
    vel_true  : (N, 2) ground truth velocity
    vel_pred  : (N, 2) predicted velocity
    factor    : downsampling factor
    """

    # ---------- Downsample ----------
    pos_ds, vel_pred_ds = downsample(pos, vel_pred, factor=factor)
    _, vel_true_ds = downsample(pos, vel_true, factor=factor)

    # ---------- Scatter: True vs Pred ----------
    plt.figure(figsize=(8,6))
    plt.scatter(vel_true_ds[:,0], vel_pred_ds[:,0], s=2, alpha=0.5, label='u component')
    plt.scatter(vel_true_ds[:,1], vel_pred_ds[:,1], s=2, alpha=0.5, label='v component')

    lo = min(vel_true_ds.min(), vel_pred_ds.min())
    hi = max(vel_true_ds.max(), vel_pred_ds.max())

    plt.plot([lo,hi],[lo,hi], "k--", label="Perfect prediction")
    plt.xlabel("True velocity")
    plt.ylabel("Predicted velocity")
    plt.title("Predicted vs True Velocities (sparse)")
    plt.legend()

    # ---------- True field ----------
    plt.figure(figsize=(8,6))
    sc = plt.scatter(pos_ds[:, 0], pos_ds[:, 1],
                     c = vel_true_ds[:, 0], s = 0.75)
    plt.title("True X-Velocity (sparse)")
    plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal")
    plt.colorbar(sc, label="X-Velocity")

    # ---------- Predicted field ----------
    plt.figure(figsize=(8,6))
    sc = plt.scatter(pos_ds[:, 0], pos_ds[:, 1],
                     c = vel_pred_ds[:, 0], s = 0.75)
    plt.title("Predicted X-Velocity (sparse)")
    plt.xlabel("x"); plt.ylabel("y"); plt.axis("equal")
    plt.colorbar(sc, label="X-Velocity")

    # ---------- Error field ----------
    error = np.linalg.norm(vel_true_ds - vel_pred_ds, axis=1)

    plt.figure(figsize=(8,6))
    sc = plt.scatter(pos_ds[:,0], pos_ds[:,1], c=error, s=1, cmap='inferno')
    plt.colorbar(sc, label='Velocity Error Magnitude')
    plt.xlabel("x"); plt.ylabel("y"); plt.title("Pointwise Velocity Error")
    plt.axis("equal")

    # ---------- Histogram ----------
    plt.figure(figsize=(6,4))
    plt.hist(error, bins=50, color='skyblue')
    plt.xlabel("Velocity Error Magnitude")
    plt.ylabel("Count")
    plt.title("Velocity Error Distribution")

    plt.show()



root = r"C:\Users\...\AirfRANS\af\Dataset" #where dataset is saved, change as necessary

# load trained model
model, ckpt = load_trained_model("velocity_gnn_physics22.pth") #current best saved model is velocity_gnn_physics22.pth
X_mean = ckpt.get("X_mean", None)
X_std  = ckpt.get("X_std", None)
Y_mean = ckpt.get("Y_mean", None)
Y_std  = ckpt.get("Y_std", None)

# load data (randomly chosen simulation)
val_ds = AirfransGraphDatasetFull(root, ['airFoil2D_SST_75.03_3.282_5.553_1.947_9.749'], normalize=True, k=12, 
    X_mean=X_mean,
    X_std=X_std,
    Y_mean=Y_mean,
    Y_std=Y_std,)

# plot training history
plot_training(ckpt["history"])


# run inference
device = "cuda" if torch.cuda.is_available() else "cpu"
sample = val_ds.get(0).to(device)
pos = sample.pos.cpu().numpy()
vel_true = (sample.y.cpu().numpy())


model.eval()
with torch.no_grad():
    pred = model(sample.x.to(device),
                 sample.edge_index.to(device),
                 sample.edge_attr.to(device))

Y_mean_t = torch.tensor(val_ds.Y_mean, dtype=torch.float32, device=pred.device)
Y_std_t  = torch.tensor(val_ds.Y_std, dtype=torch.float32, device=pred.device)

# unscale
pred_unscaled = pred * Y_std_t + Y_mean_t 
pred_np = pred_unscaled.cpu().numpy()
vel_true1 = sample.y * Y_std_t + Y_mean_t
vel_true_np = vel_true1.cpu().numpy()

plot_gnn_results(pos, vel_true_np, pred_np, factor=10)