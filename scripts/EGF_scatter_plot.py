import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from model import MultiLinearRegression

# =====================================================================
# GUIDANCE: COMPARING LINEAR VS NON-LINEAR DRIVERS
# This script visualizes the discrepancy between features deemed important 
# by a baseline Linear model versus the MLP (via Integrated Gradients).
# Features falling far off the diagonal highlight complex biological rules
# (e.g., threshold effects or protein interactions) that purely linear 
# models fail to capture.
# =====================================================================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA_DIR = os.path.join('..', 'data', 'EGF_perturb')
MODEL_DIR = os.path.join('..', 'models')
PLOT_DIR = os.path.join(DATA_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

device = 'cpu'
seed = 42 
d = 510
concentration = [0, 1, 6.25, 10, 25, 100]
datanum = [[13, 14, 15], [16, 17, 18], [10, 19, 20], [7, 8, 9], [4, 5, 6], [1, 2, 3]]
con_idx = 5 # Example: 100 ng/mL
response_idx = 3 # 3 = pERK
response_name = {3: 'pERK', 18: 'pEGFR', 33: 'pAKT', 48: 'pFAK', 63: 'FoxO3a', 78: 'pS6'}

def main():
    print("Loading data for Linear vs Non-Linear feature comparison...")

    # Load Non-Linear attributions (IG matrix)
    try:
        W_mlp = np.load(os.path.join(DATA_DIR, f'W_{response_idx}-{concentration[con_idx]}-dim{d}-z.npy'))
    except FileNotFoundError:
        print("ERROR: Run get_ig.py to generate W_ matrices first.")
        return

    # Extract global importance ranks for the MLP
    mean_abs_attr_mlp = np.mean(np.abs(W_mlp), axis=0)
    ranks_mlp = len(mean_abs_attr_mlp) - np.argsort(np.argsort(mean_abs_attr_mlp))

    # Load Linear Model and extract weights directly
    model_lin = MultiLinearRegression(input_size=d, n_models=6, n_egf=6, device=device).to(device)
    model_lin.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'MultiLinear_{response_idx}-{seed}-{d}-z.pth'), map_location=device))
    
    linear_weights = model_lin.models[con_idx].weight.data.numpy().squeeze()
    mean_abs_attr_lin = np.abs(linear_weights)
    ranks_lin = len(mean_abs_attr_lin) - np.argsort(np.argsort(mean_abs_attr_lin))

    # Plot Setup
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(ranks_lin, ranks_mlp, color='blue', alpha=0.5)

    ax.set_xlim(0, d + 10)
    ax.set_ylim(0, d + 10)
    
    # Y = X line indicates agreement between Linear and Non-Linear
    ax.plot([0, d + 10], [0, d + 10], 'r--', alpha=0.7) 
    
    ax.set_title(f'Feature Importance Rank Shift: Linear vs Non-Linear\nTarget: {response_name[response_idx]} | EGF: {concentration[con_idx]} ng/mL', fontsize=18)
    ax.set_xlabel('Rank derived from Linear Regression Weights', fontsize=16)
    ax.set_ylabel('Rank derived from MLP Integrated Gradients', fontsize=16)
    
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Highlight zones of discrepancy
    ax.fill_between([0, d+10], [0, d+10], [0, 0], color='lightgrey', alpha=0.3)
    
    legend_patch = mpatches.Patch(color='lightgrey', label='Features Underestimated by Linear Model')
    ax.legend(handles=[legend_patch], loc='lower right', fontsize=14)

    plot_path = os.path.join(PLOT_DIR, f'ScatterRank_{response_name[response_idx]}_{concentration[con_idx]}ng.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to {plot_path}")

if __name__ == '__main__':
    main()