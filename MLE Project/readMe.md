Physics-Informed Graph Neural Networks for 2D Airfoil Flow-Field Prediction
Gavin MacKenzie

Description:
This project develops and evaluates a Physics-Informed GNN designed to predict 3D velocity fields around aerodynamic bodies using CFD simulation data.
The objective was to create a model that:

Accurately approximates CFD solutions at significantly reduced computational cost
Respects key physical constraints (e.g., divergence-free velocity fields, boundary conditions)
Generalizes across a diverse set of flow conditions and geometries

This project demonstrates how geometric deep learning can accelerate fluid-mechanics-based design processes in aerospace engineering.

Dataset:
This project used the AirFRANS dataset, a collection of 2D RANS CFD simulations

Model Overview:
The core model is a GNN using GraphSAGE message-passing layers. Key architectural components include:

GraphSAGE Convolution Layers to enable node-wise feature aggregation over mesh connectivity, preserving spatial relationships in unstructured CFD grids
Physics-Informed Loss Terms: Divergence penalty (∇·u ≈ 0) and Boundary condition losses via masked supervision
Smoothness and consistency regularization
MLP Decoder to map latent node embeddings to predicted velocity/pressure
Dropout & Layer Normalization to improve generalization and stabilize training.

Training Procedure:
AdamW optimizer
Mini-batching with PyTorch Geometric’s DataLoader
Gradient clipping
Monitoring of training/validation losses with LR scheduling and early stopping

How to run:
Creat a virtual environment (recommended)
Install required libraries and packages (see requirements.text)
Download code or clone repository
Run af_download to download dataset
Run af_run_training to train and save a new model and/or download velocity_gnn_physics22.pth for already trained model
Run af_plotting to load the trained model, perform inferencing, and plot relevant results

Results:
The GNN successfully learns high-fidelity velocity predictions from unstructured CFD meshes
Error histograms show that most predictions fall within a small deviation from ground truth
Scatter plots indicate strong linear correlation between predicted and true velocity components.\
Point-wise comparisons reveal the model captures major flow features (wake regions, stagnation zones), with largest errors occurring near high-gradient boundary layers
The physics-informed loss significantly reduces divergence and improves physical consistency compared to a purely data-driven model
Overall, the model offers a computationally efficient surrogate to accelerate aerodynamic design loops
More improvement can be made by increasing number of training samples or improving the physics constraints
