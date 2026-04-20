import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import os

# =====================================================================
# GUIDANCE: STEP 3 - VISUALIZING NONLINEAR DECISION BOUNDARIES
# This script creates a heatmap mapping how the model shifts its attention 
# (attribution) across two features (x1 and x3) while keeping others static.
# This visually proves the pipeline's ability to decode complex dependencies.
# =====================================================================

DATA_DIR = os.path.join('..', 'data', 'synthetic')
os.makedirs(DATA_DIR, exist_ok=True)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = 'cpu'
torch.manual_seed(3)

# Load Model
model = MLP(input_dim=4, hidden_dim=256, output_dim=1).to(device)
model_path = os.path.join(DATA_DIR, 'Synthetic_MLP.pth')
model.load_state_dict(torch.load(model_path))
model.eval()

ig = IntegratedGradients(model)

# Define grid for visualization
n_points = 50
x1_range = np.linspace(-3, 3, n_points)
x3_range = np.linspace(-3, 3, n_points)
x1_grid, x3_grid = np.meshgrid(x1_range, x3_range)

# Keep x0 and x2 fixed
x0_val = 0.0
x2_val = 1.0

X_grid = np.zeros((n_points * n_points, 4))
X_grid[:, 0] = x0_val
X_grid[:, 1] = x1_grid.flatten()
X_grid[:, 2] = x2_val
X_grid[:, 3] = x3_grid.flatten()

X_tensor = torch.tensor(X_grid, dtype=torch.float32, requires_grad=True).to(device)

print("Calculating IG across 2D Grid...")
attributions = ig.attribute(X_tensor, target=0, n_steps=50)

# We are visualizing the attribution map for X2 (index 2)
feature_idx = 2
attr_map = attributions[:, feature_idx].detach().numpy().reshape(n_points, n_points)

# Define custom colormap
colors = ["#2B3B60", "#4B6B9E", "#7291B8", "#A4BEDA", "#E8F0F9", "#FFFFFF",
          "#FFF0E6", "#FFC7A8", "#FF9260", "#E45A34", "#B51A18"]
my_blue_cmap = LinearSegmentedColormap.from_list("my_custom_blue", colors)

# Plot
plt.figure(figsize=(8, 6))
cp = plt.contourf(x1_grid, x3_grid, attr_map, levels=30, cmap=my_blue_cmap)

plt.colorbar(cp, label=f'Attribution Score (Feature X{feature_idx})')
plt.xlabel('X1')
plt.ylabel('X3')
plt.title(f'Heatmap of Feature X{feature_idx} Attribution')
plt.tight_layout()

heatmap_path = os.path.join(DATA_DIR, 'attribution_heatmap_x2=1.svg')
plt.savefig(heatmap_path, format='svg', bbox_inches='tight')
print(f"Saved Heatmap to: {heatmap_path}")