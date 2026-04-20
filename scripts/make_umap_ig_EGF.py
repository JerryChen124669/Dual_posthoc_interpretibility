import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import umap

# =====================================================================
# GUIDANCE: CONTINUOUS UMAP ATTRIBUTION PROJECTION
# Instead of binary thresholds, this script maps the raw, continuous 
# Integrated Gradient (IG) scores onto the UMAP embedding. This provides 
# a visual confirmation of "gradients of activation" within the cellular 
# population, ensuring that the model's learned rules map smoothly onto 
# biological topography.
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
concent_idx = 3
response_name = {3: 'pERK', 18: 'pEGFR', 33: 'pAKT', 48: 'pFAK', 63: 'FoxO3a', 78: 'pS6'}

# Note: Assumes attributions were saved to the data folder by `get_ig.py`
attr = np.load(os.path.join(DATA_DIR, f'attr_{response_idx}-{concentration[concent_idx]}-dim{d}-z.npy'))
W = np.load(os.path.join(DATA_DIR, f'W_{response_idx}-{concentration[concent_idx]}-dim{d}-z.npy'))
X = np.load(os.path.join(DATA_DIR, f'X_ig_{response_idx}-{concentration[concent_idx]}-dim{d}-z.npy'))

feature_idx = 75
interp = 'IG value' # 'IG value' or 'Feature value'

print(f"Generating Continuous UMAP for {interp}...")

reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean', random_state=42)
feature_reduced = reducer.fit_transform(X)

all_feature_names = []
try:
    with open(os.path.join(DATA_DIR, 'feature_name.txt'), 'r') as file:
        for line in file:
            all_feature_names.append(line.strip())
except FileNotFoundError:
    all_feature_names = [f"Feature_{i}" for i in range(d)]

if interp == 'IG value':
    y = attr[:, feature_idx]
else:
    y = X[:, feature_idx]

# Center color map on 0
max_val = y.max()
min_val = y.min()
vmin, vmax = min(min_val, -max_val), max(max_val, -min_val)

# Custom diverging colormap (Ordered Blue -> Yellow/White -> Red)
custom_hex_colors = [
    "#00441b", "#1b7837", "#5aae61", "#a6dba0", 
    "#d9f0d3",  
    "#e7d4e8", "#c2a5cf", "#9970ab", "#762a83"
]
my_cmap = LinearSegmentedColormap.from_list("custom_rdybl", custom_hex_colors)
norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

plt.figure(figsize=(12, 10))
plt.scatter(feature_reduced[:, 0], feature_reduced[:, 1], s=2, c=y, cmap=my_cmap, norm=norm)

plt.title(f'UMAP colored by {interp} of {all_feature_names[feature_idx]} \nEGF: {concentration[concent_idx]} ng/ml', fontsize=16)
plt.colorbar()
plt.axis('off')
plt.tight_layout()

out_file = os.path.join(OUT_DIR, f'umap_{interp.replace(" ","")}_{response_idx}-{concentration[concent_idx]}-feat{feature_idx}.svg')
plt.savefig(out_file, format='svg')
print(f"Saved Continuous UMAP to: {out_file}")