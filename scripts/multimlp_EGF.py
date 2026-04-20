import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from model import MultiMLPRegression, MultiLinearRegression

# =====================================================================
# GUIDANCE: MULTIMODAL MLP TRAINING FOR EGF PERTURBATION
# This script trains the core predictive model of the PITCH pipeline.
# By predicting specific kinase activities (e.g., pERK) from 510 
# multimodal features (transcriptomic + morphological), we force the 
# network to learn the underlying biological signal transduction logic.
# =====================================================================

def parse_arguments():
    parser = argparse.ArgumentParser()
    # PATH UPDATES for repository structure
    parser.add_argument('--data_dir', type=str, default=os.path.join('..', 'data', 'EGF_perturb'))
    parser.add_argument('--model_dir', type=str, default=os.path.join('..', 'models'))
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='MultiMLP')
    parser.add_argument('--response', type=int, default=78)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_model', type=int, default=6)
    parser.add_argument('--n_epochs', type=int, default=40000)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--model_save', type=bool, default=True)
    parser.add_argument('--plot_save', type=bool, default=True)
    return parser.parse_args()

def main():
    args = parse_arguments()
    os.makedirs(args.model_dir, exist_ok=True)
    
    plot_dir = os.path.join(args.data_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Loading biological data from {args.data_dir}...")
    mat = h5py.File(os.path.join(args.data_dir, 'Data_1.mat'), 'r')
    LinearIndex = np.array(mat['LinearIndex']).T.squeeze()
    ResponseData = np.array(mat['ResponseData'])
    FeatureData = np.load(os.path.join(args.data_dir, 'FeatureData-z-510.npy'))

    concentration = [0, 1, 6.25, 10, 25, 100]
    datanum = [[13, 14, 15], [16, 17, 18], [10, 19, 20], [7, 8, 9], [4, 5, 6], [1, 2, 3]]
    response_idx = args.response

    # Prepare datasets for 6 different EGF concentrations
    X, Y, EGF = [], [], []
    for i in range(len(datanum)):
        data_idx = np.where(np.isin(LinearIndex, datanum[i]))[0]
        X.append(FeatureData[data_idx, :])
        Y.append(ResponseData[response_idx, data_idx])
        EGF.append(np.full((FeatureData[data_idx, :].shape[0],), i))

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    EGF = np.concatenate(EGF)
    d = X.shape[1]

    # Subsample data array indices for train/test splits
    rand_int = np.random.choice(range(X.shape[0]), 20000, replace=False)
    np.save(os.path.join(args.data_dir, f'rand_int_{response_idx}-all-dim{d}-z.npy'), rand_int)

    X, Y, EGF = X[rand_int, :], Y[rand_int], EGF[rand_int]
    X_train, X_test, Y_train, Y_test, EGF_train, EGF_test = train_test_split(X, Y, EGF, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    EGF_train_t = torch.tensor(EGF_train, dtype=torch.long).to(device)
    
    train_dataset = TensorDataset(X_train_t, Y_train_t, EGF_train_t)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

    # Initialize model
    model_name = args.model
    print(f"Initializing {model_name}...")
    if model_name == 'MultiLinear':
        model = MultiLinearRegression(input_size=d, n_models=args.n_model, n_egf=len(concentration), device=device).to(device)
    else:
        model = MultiMLPRegression(input_size=d, hidden_size=args.hidden_size, n_models=args.n_model,
                                   n_egf=len(concentration), dropout=args.dropout, device=device).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training Loop
    for epoch in range(args.n_epochs):
        model.train()
        for batch_X, batch_y, batch_EGF in train_loader:
            outputs = model(batch_X, batch_EGF)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{args.n_epochs}], Loss: {loss.item():.4f}')

    # Model Saving
    if args.model_save:
        save_path = os.path.join(args.model_dir, f'{model_name}_{response_idx}-{seed}-{d}-z.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    # Validation
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        EGF_test_t = torch.tensor(EGF_test, dtype=torch.long).to(device)
        y_pred_test = model(X_test_t, EGF_test_t).cpu().numpy()

    sk_RSquared_test = explained_variance_score(Y_test, y_pred_test)
    sp_Pearson_test = pearsonr(Y_test, y_pred_test)
    print(f'Final Validation: RSquared = {sk_RSquared_test:.4f}, Pearson = {sp_Pearson_test.statistic:.4f}')

if __name__ == '__main__':
    main()