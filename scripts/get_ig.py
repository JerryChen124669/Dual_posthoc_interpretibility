import os
import h5py
import numpy as np
import torch
from captum.attr import IntegratedGradients
import argparse
from model import MultiMLPRegression, MLPRegression

# =====================================================================
# GUIDANCE: INTEGRATED GRADIENTS (IG) COMPUTATION
# This script computes feature attributions using the trained MLP. 
# IG calculates the mathematical integral of the model's gradients 
# from a baseline state to the cell's observed state. This safely 
# decomposes the model's non-linear predictions into exact, per-feature 
# importance scores.
# =====================================================================

# Fix OpenMP error for local compute
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def parse_arguments():
    parser = argparse.ArgumentParser()
    # PATH UPDATES
    parser.add_argument('--data_dir', type=str, default=os.path.join('..', 'data', 'EGF_perturb'))
    parser.add_argument('--model_dir', type=str, default=os.path.join('..', 'models'))
    parser.add_argument('--device', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='MultiMLP')
    parser.add_argument('--response', type=int, default=78)
    parser.add_argument('--concentrate_idx', type=int, default=5)
    parser.add_argument('--attr_save', type=bool, default=True)
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    device = torch.device("cpu") # Force CPU to avoid CUDA memory limits during large attribution arrays
    seed = args.seed
    
    print(f"Loading data from {args.data_dir}...")
    mat = h5py.File(os.path.join(args.data_dir, 'Data_1.mat'), 'r')
    LinearIndex = np.array(mat['LinearIndex']).T.squeeze()
    FeatureData = np.load(os.path.join(args.data_dir, 'FeatureData-z-510.npy'))
    
    concentration = [0, 1, 6.25, 10, 25, 100]
    datanum = [[13, 14, 15], [16, 17, 18], [10, 19, 20], [7, 8, 9], [4, 5, 6], [1, 2, 3]]
    con_idx = args.concentrate_idx
    response_idx = args.response
    model_name = args.model
    d = FeatureData.shape[1]

    print(f"Loading {model_name} from {args.model_dir}...")
    if model_name == 'MLP':
        model = MLPRegression(input_size=d, hidden_size=256, dropout=0.5).to(device)
        model.load_state_dict(torch.load(os.path.join(args.model_dir, f'MLP_{response_idx}.pth'), map_location=device))
    else:
        model = MultiMLPRegression(input_size=d, hidden_size=256, n_models=6, n_egf=6, dropout=0.5, device=device).to(device)
        model.load_state_dict(torch.load(os.path.join(args.model_dir, f'{model_name}_{response_idx}-{seed}-{d}-z.pth'), map_location=device))
    model.eval()

    # Data Subsetting
    data_idx = np.where(np.isin(LinearIndex, datanum[con_idx]))[0]
    X = FeatureData[data_idx, :]
    
    # Subsample to 20,000 cells for compute efficiency
    rand_int = np.random.choice(range(data_idx.shape[0]), 20000, replace=False)
    X = X[rand_int, :]
    EGF = torch.from_numpy(np.full((X.shape[0],), con_idx)).to(torch.long)
    X_tensor = torch.from_numpy(X).to(torch.float32).to(device)
    X_tensor.requires_grad = True

    print(f"Computing Integrated Gradients for {concentration[con_idx]} ng/mL...")
    intpmodel = IntegratedGradients(model, multiply_by_inputs=False)
    W = intpmodel.attribute(X_tensor, additional_forward_args=(EGF,), n_steps=50)
    
    # Final Attribution = Gradients * Inputs
    W_np = W.cpu().detach().numpy()
    attr_np = torch.mul(X_tensor, W).cpu().detach().numpy()
    X_np = X_tensor.cpu().detach().numpy()

    if args.attr_save:
        # Save indices so UMAP plotting scripts match the exact same cells
        np.save(os.path.join(args.data_dir, f'rand_int_{response_idx}-{concentration[con_idx]}-dim{d}-z.npy'), rand_int)
        np.save(os.path.join(args.data_dir, f'W_{response_idx}-{concentration[con_idx]}-dim{d}-z.npy'), W_np)
        np.save(os.path.join(args.data_dir, f'attr_{response_idx}-{concentration[con_idx]}-dim{d}-z.npy'), attr_np)
        np.save(os.path.join(args.data_dir, f'X_ig_{response_idx}-{concentration[con_idx]}-dim{d}-z.npy'), X_np)
        print(f"Successfully saved IG matrices to {args.data_dir}")

if __name__ == '__main__':
    main()