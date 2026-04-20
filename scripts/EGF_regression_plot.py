import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import os
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score
from model import MultiMLPRegression

# =====================================================================
# GUIDANCE: REGRESSION VALIDATION
# Before interpreting a deep learning model, it is critical to verify 
# its predictive performance. This script visualizes the correlation 
# between ground-truth protein levels and the MLP's predictions to 
# confirm the biological relationships have been properly learned.
# =====================================================================

DATA_DIR = os.path.join('..', 'data', 'EGF_perturb')
MODEL_DIR = os.path.join('..', 'models')
PLOT_DIR = os.path.join(DATA_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

seed = 42
d = 510
concentration = [0, 1, 6.25, 10, 25, 100]
datanum = [[13, 14, 15], [16, 17, 18], [10, 19, 20], [7, 8, 9], [4, 5, 6], [1, 2, 3]]
response_idx = 78  
model_name = 'MultiMLP'

response_name = {3: 'pERK', 18: 'pEGFR', 33: 'pAKT', 48: 'pFAK', 63: 'FoxO3a', 78: 'pS6',
                 93: 'FoxO1', 108: 'pMTOR', 123: 'pRSK', 153: 'pGSK3B', 168: 'pMEK', 183: 'pPI3K', 198: 'pS6K'}

# Limits for plotting axes
lim = {78: [(0, 5), (0, 7), (0, 12), (0, 12), (0, 12), (0, 12)]}

def main():
    device = torch.device('cpu')
    print("Loading data for Regression Plot...")
    
    mat = h5py.File(os.path.join(DATA_DIR, 'Data_1.mat'), 'r')
    LinearIndex = np.array(mat['LinearIndex']).T.squeeze()
    ResponseData = np.array(mat['ResponseData'])
    FeatureData = np.load(os.path.join(DATA_DIR, 'FeatureData-z-510.npy'))

    X, Y, EGF = [], [], []
    for i in range(len(datanum)):
        data_idx = np.where(np.isin(LinearIndex, datanum[i]))[0]
        X.append(FeatureData[data_idx, :])
        Y.append(ResponseData[response_idx, data_idx])
        EGF.append(np.full((FeatureData[data_idx, :].shape[0],), i))

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    EGF = np.concatenate(EGF)

    # Load specific test indices generated during training
    rand_int = np.load(os.path.join(DATA_DIR, f'rand_int_{response_idx}-all-dim510-z.npy'))
    X, Y, EGF = X[rand_int, :], Y[rand_int], EGF[rand_int]

    X_test_t = torch.tensor(X, dtype=torch.float32).to(device)
    EGF_test_t = torch.tensor(EGF, dtype=torch.long).to(device)

    model = MultiMLPRegression(input_size=d, hidden_size=256, n_models=6, n_egf=6, dropout=0.5, device=device).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{model_name}_{response_idx}-{seed}-{d}-z.pth'), map_location=device))
    model.eval()

    with torch.no_grad():
        y_pred_test = model(X_test_t, EGF_test_t).cpu().numpy()

    for i in range(len(datanum)):
        data_idx = np.where(np.isin(EGF, i))[0]
        sp_Pearson_test = pearsonr(Y[data_idx], y_pred_test[data_idx])
        
        plt.figure(figsize=(8, 8))
        plt.scatter(Y[data_idx], y_pred_test[data_idx], s=4, alpha=0.8)
        
        # Plot ideal y=x line
        lim_min, lim_max = lim[response_idx][i]
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r') 
        
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('Measured Protein Expression', fontsize=20)
        plt.ylabel('MLP Predicted Expression', fontsize=20)
        plt.title(f'Target: {response_name[response_idx]} | EGF: {concentration[i]} ng/mL\nPearson r: {sp_Pearson_test.statistic:.3f}', fontsize=16)
        
        plot_path = os.path.join(PLOT_DIR, f'Regression_{response_name[response_idx]}_{concentration[i]}ng.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Regression plots saved to {PLOT_DIR}")

if __name__ == '__main__':
    main()