import os
import random
import torch

from af_build_data import AirfransGraphDatasetFull
from af_gnn_model import train_gnn_physics
from af_save_load_helpers import save_physics_checkpoint


root = r"C:\Users\...\AirfRANS\af\Dataset" #where dataset is saved, change as necessary
n_train = 40
n_val = 10
shuffle_sims = True

all_sims = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
if shuffle_sims:
    random.shuffle(all_sims)
assert len(all_sims) >= n_train + n_val

train_sims = all_sims[:n_train]
val_sims = all_sims[n_train:n_train + n_val]

train_ds = AirfransGraphDatasetFull(root, train_sims, normalize=True, k=12)
val_ds = AirfransGraphDatasetFull(root, val_sims, normalize=True, k=12, 
                                X_mean=train_ds.X_mean,
                                X_std=train_ds.X_std,
                                Y_mean=train_ds.Y_mean,
                                Y_std=train_ds.Y_std)
    
model, history = train_gnn_physics(
    train_ds,
    val_ds,
    epochs=60,
    batch_size=1,
    lr=5e-4,
    lambda_bc=0.05,
    lambda_norm=0.05,
    lambda_inlet=0.2,
    lambda_div=5e-4,
    patience=8
)

save_physics_checkpoint("velocity_gnn_physics22.pth", model, history, dataset=train_ds) #current model save name

# quick inference
device = "cuda" if torch.cuda.is_available() else "cpu"
sample = val_ds.get(0).to(device)
with torch.no_grad():
    pred = model(sample.x, sample.edge_index, sample.edge_attr)
print("Pred shape:", pred.shape)