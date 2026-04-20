import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import umap

# =====================================================================
# GUIDANCE: DISCRETE LOGIC GATE UMAP PROJECTION
# This script takes the human-readable logic rules extracted by the 
# Decision Tree (e.g., Feature A > X AND Feature B < Y) and applies 
# them as a strict Boolean filter to the UMAP. This acts as a visual 
# ground-truth check that the mathematically extracted rules cleanly 
# partition distinct, biologically cohesive cellular subpopulations.
# =====================================================================

DATA_DIR = os.path.join('..', 'data', 'EGF_perturb')
OUT_DIR = os.path.join(DATA_DIR, 'UMAP_Results')
os.makedirs(OUT_DIR, exist_ok=True)

mat = h5py.File(os.path.join(DATA_DIR, 'Data_1.mat'), 'r')
LinearIndex = np.array(mat['LinearIndex']).T.squeeze()
data_idx = np.where(np.isin(LinearIndex, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20]))[0]
LinearIndex = LinearIndex[data_idx]
FeatureData = np.load(os.path.join(DATA_DIR, 'FeatureData-z-510.npy'))
d = FeatureData.shape[1]

concentration = [0, 1, 6.25, 10, 25, 100]
datanum = [[13, 14, 15], [16, 17, 18], [10, 19, 20], [7, 8, 9], [4, 5, 6], [1, 2, 3]]

response_idx = 3
concent_idx = 1 # Example: 1 ng/mL
feature_idx = 75

print(f"Generating Discrete Logic UMAPs for 1 ng/mL EGF...")

attr = np.load(os.path.join(DATA_DIR, f'attr_{response_idx}-{concentration[concent_idx]}-dim{d}-z.npy'))
X = np.load(os.path.join(DATA_DIR, f'X_ig_{response_idx}-{concentration[concent_idx]}-dim{d}-z.npy'))

reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean', random_state=42)
feature_reduced = reducer.fit_transform(X)

all_feature_names = []
try:
    with open(os.path.join(DATA_DIR, 'feature_name.txt'), 'r') as file:
        for line in file:
            all_feature_names.append(line.strip())
except FileNotFoundError:
    all_feature_names = [f"Feature_{i}" for i in range(d)]

custom_cmap = ListedColormap(['lightgrey', 'red'])

# Logic Gate 1: Based on specific biological thresholds extracted by the tree
y1 = np.where((X[:, 67] <= 0.577) & (X[:, 50] <= 1.389) & (X[:, 151] > 0.052), 1, 0)
plt.figure(figsize=(10, 10))
plt.scatter(feature_reduced[:, 0], feature_reduced[:, 1], s=2, c=y1, cmap=custom_cmap)
plt.title(f'Logic Gate 1 Projection\nTarget Feature: {all_feature_names[feature_idx]} | EGF {concentration[concent_idx]} ng/ml', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'umap_LogicGate1_{response_idx}-{concentration[concent_idx]}-feat{feature_idx}.svg'), format='svg')

# Logic Gate 2: A different subpopulation logic branch
y2 = np.where((X[:, 67] <= 0.577) & (X[:, 50] > 1.389) & (X[:, 0] > 1.039), 1, 0)
plt.figure(figsize=(10, 10))
plt.scatter(feature_reduced[:, 0], feature_reduced[:, 1], s=2, c=y2, cmap=custom_cmap)
plt.title(f'Logic Gate 2 Projection\nTarget Feature: {all_feature_names[feature_idx]} | EGF {concentration[concent_idx]} ng/ml', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'umap_LogicGate2_{response_idx}-{concentration[concent_idx]}-feat{feature_idx}.svg'), format='svg')

print(f"✅ Discrete Logic UMAPs saved successfully to {OUT_DIR}")