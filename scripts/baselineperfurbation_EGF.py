import os
import h5py
import numpy as np
import torch
from captum.attr import IntegratedGradients
import argparse
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imblearn.under_sampling import RandomUnderSampler
import umap  

# =====================================================================
# GUIDANCE: COMPREHENSIVE BASELINE PERTURBATION
# This script computes attributions across THREE different baselines:
# 1. Global Baseline (Zero vector)
# 2. Biological Neutral Baseline (0 ng/mL EGF)
# 3. Local Condition Baseline (10 ng/mL EGF)
# It trains a decision tree for each to evaluate how baseline selection 
# impacts rule extraction, and validates the extracted rules via UMAP.
# =====================================================================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from model import MultiMLPRegression, MLPRegression

def parse_arguments():
    parser = argparse.ArgumentParser()
    # PATH UPDATES for the repository structure
    parser.add_argument('--data_dir', type=str, default=os.path.join('..', 'data', 'EGF_perturb'))
    parser.add_argument('--model_dir', type=str, default=os.path.join('..', 'models')) 
    parser.add_argument('--device', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='MultiMLP')
    parser.add_argument('--response', type=int, default=3)
    parser.add_argument('--concentrate_idx', type=int, default=3) # 10 ng/mL EGF
    parser.add_argument('--feature_idx', type=int, default=75)    
    parser.add_argument('--percentage', type=int, default=90)
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--balance', type=bool, default=False)
    parser.add_argument('--select_feature', type=bool, default=True)
    parser.add_argument('--self_exclude', type=bool, default=True)
    return parser.parse_args()

def get_best_leaf_node(dt_model, X_eval, y_eval):
    leaf_ids = dt_model.apply(X_eval)
    unique_leaves = np.unique(leaf_ids)
    
    best_precision = -1.0
    best_leaf = -1
    
    for leaf in unique_leaves:
        in_leaf = (leaf_ids == leaf)
        n_samples = np.sum(in_leaf)
        if n_samples > 0:
            class_1_count = np.sum(y_eval[in_leaf] == 1)
            precision = class_1_count / n_samples
            if precision > best_precision:
                best_precision = precision
                best_leaf = leaf
            elif precision == best_precision and precision > 0:
                if n_samples > np.sum(leaf_ids == best_leaf):
                    best_leaf = leaf
                    
    if best_leaf == -1:
        best_leaf = unique_leaves[0] 
                
    return best_leaf, best_precision

def extract_and_apply_rules(dt_model, feature_names, best_leaf_id, X_data, baseline_name):
    """
    GUIDANCE: Traces the optimal decision path from root to leaf, translates 
    sklearn's threshold arrays into human-readable logical bounds, and extracts 
    the targeted cellular subpopulation.
    """
    tree = dt_model.tree_
    parent_map = {}
    for i in range(tree.node_count):
        if tree.children_left[i] != -1:
            parent_map[tree.children_left[i]] = (i, "left")
        if tree.children_right[i] != -1:
            parent_map[tree.children_right[i]] = (i, "right")
            
    path = []
    current_node = best_leaf_id
    while current_node in parent_map:
        parent, direction = parent_map[current_node]
        path.insert(0, (parent, direction))
        current_node = parent
        
    mask = np.ones(X_data.shape[0], dtype=bool)
    
    print(f"\n--- Extracted Rules for {baseline_name} (Node ID: {best_leaf_id}) ---")
    if not path:
        print("RULE: Root Node (No rules applied)")
        
    for node, direction in path:
        feat_idx = tree.feature[node]
        threshold = np.float32(tree.threshold[node])
        feat_name = feature_names[feat_idx]
        
        if direction == "left":
            print(f"RULE: {feat_name} <= {threshold:.4f}")
            mask = mask & (X_data[:, feat_idx] <= threshold)
        else:
            print(f"RULE: {feat_name} > {threshold:.4f}")
            mask = mask & (X_data[:, feat_idx] > threshold)
            
    return mask.astype(int)

def main():
    args = parse_arguments()
    out_dir = os.path.join(args.data_dir, 'Perturbation_Results')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cpu")
    seed = args.seed
    
    print(f"Loading biological data from {args.data_dir}...")
    mat = h5py.File(os.path.join(args.data_dir, 'Data_1.mat'), 'r')
    LinearIndex = np.array(mat['LinearIndex']).T.squeeze()
    FeatureData = np.load(os.path.join(args.data_dir, 'FeatureData-z-510.npy'))
    
    concentration = [0, 1, 6.25, 10, 25, 100]
    datanum = [[13, 14, 15], [16, 17, 18], [10, 19, 20], [7, 8, 9], [4, 5, 6], [1, 2, 3]]
    con_idx = args.concentrate_idx
    response_idx = args.response  
    model_name = args.model
    d = FeatureData.shape[1]

    data_idx = np.where(np.isin(LinearIndex, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20]))[0]
    LinearIndex = LinearIndex[data_idx]

    idx_0 = np.where(np.isin(LinearIndex, datanum[0]))[0]
    baseline_0_raw = FeatureData[idx_0, :].mean(axis=0) 
    
    idx_10 = np.where(np.isin(LinearIndex, datanum[3]))[0]
    baseline_10_raw = FeatureData[idx_10, :].mean(axis=0)

    print(f"\nProcessing Analyzed Dataset: {concentration[con_idx]} ng/mL...")
    
    rand_int = np.load(os.path.join(args.data_dir, f'rand_int_{response_idx}-{concentration[con_idx]}-dim{d}-z.npy'))
    data_idx = np.where(np.isin(LinearIndex, datanum[con_idx]))[0]
    X_subset = FeatureData[data_idx, :][rand_int, :] 

    print(f"Loading model {model_name} from {args.model_dir}...")
    if model_name == 'MLP':
        nn_model = MLPRegression(input_size=d, hidden_size=256, dropout=0.5).to(device)
        nn_model.load_state_dict(torch.load(os.path.join(args.model_dir, f'MLP_{response_idx}.pth'), map_location=device))
    else:
        nn_model = MultiMLPRegression(input_size=d, hidden_size=256, n_models=6, n_egf=6, dropout=0.5, device=device).to(device)
        nn_model.load_state_dict(torch.load(os.path.join(args.model_dir, f'{model_name}_{response_idx}-{seed}-{d}-z.pth'), map_location=device))
    nn_model.eval()

    # Calculate IG for three different baselines
    EGF = torch.from_numpy(np.full((X_subset.shape[0],), con_idx)).to(torch.long)
    X_tensor = torch.from_numpy(X_subset).to(torch.float32).to(device)
    baseline_tensor_0 = torch.from_numpy(baseline_0_raw).to(torch.float32).to(device).unsqueeze(0).expand_as(X_tensor)
    baseline_tensor_10 = torch.from_numpy(baseline_10_raw).to(torch.float32).to(device).unsqueeze(0).expand_as(X_tensor)

    intpmodel = IntegratedGradients(nn_model, multiply_by_inputs=False)
    
    W_global = intpmodel.attribute(X_tensor, additional_forward_args=(EGF,), n_steps=50)
    W_0ng = intpmodel.attribute(X_tensor, baselines=baseline_tensor_0, additional_forward_args=(EGF,), n_steps=50)
    W_10ng = intpmodel.attribute(X_tensor, baselines=baseline_tensor_10, additional_forward_args=(EGF,), n_steps=50)
        
    attr_global = torch.mul(X_tensor, W_global).cpu().detach().numpy()
    attr_0ng = torch.mul((X_tensor - baseline_tensor_0), W_0ng).cpu().detach().numpy()
    attr_10ng = torch.mul((X_tensor - baseline_tensor_10), W_10ng).cpu().detach().numpy()

    # Feature Selection Configuration (Hardcoded subsets for this specific EGF paper figure)
    if args.select_feature:
        use_list = [0, 1, 7, 13, 14, 20, 25, 26, 32, 37, 38, 44, 50, 51, 57, 63, 67, 75, 101, 102, 110, 134, 151]
        use_map = {0: 0, 1: 1, 7: 2, 13: 3, 14: 4, 20: 5, 25: 6, 26: 7, 32: 8, 37: 9,
                   38: 10, 44: 11, 50: 12, 51: 13, 57: 14, 63: 15, 67: 16, 75: 17, 101: 18, 102: 19,
                   110: 20, 134: 21, 151: 22}
        final_feature_indices = use_list.copy()
        X_use = X_subset[:, use_list] 
        
        if args.self_exclude:
            idx_to_drop = use_map[args.feature_idx]
            final_feature_indices.pop(idx_to_drop)
            X_use = X_use[:, np.r_[0:idx_to_drop, idx_to_drop + 1:22]]
    else:
        final_feature_indices = list(range(314))
        X_use = X_subset[:, 0:314]
        if args.self_exclude:
            final_feature_indices.remove(args.feature_idx)
            X_use = X_use[:, np.r_[0:args.feature_idx, args.feature_idx + 1:314]]

    try:
        with open(os.path.join(args.data_dir, 'feature_name.txt'), 'r') as f:
            all_feature_names = [line.strip() for line in f.readlines()]
        feature_names = [all_feature_names[i] for i in final_feature_indices]
    except Exception:
        feature_names = [f"Feature_{i}" for i in final_feature_indices]

    # Binary Classification generation (Top N%)
    y_global = np.where(attr_global[:, args.feature_idx] > np.percentile(attr_global[:, args.feature_idx], args.percentage), 1, 0)
    y_0ng = np.where(attr_0ng[:, args.feature_idx] > np.percentile(attr_0ng[:, args.feature_idx], args.percentage), 1, 0)
    y_10ng = np.where(attr_10ng[:, args.feature_idx] > np.percentile(attr_10ng[:, args.feature_idx], args.percentage), 1, 0)

    dt_args = {'max_depth': 4, 'max_leaf_nodes': 14, 'min_samples_leaf': 5, 'random_state': 42}
    
    dt_global = DecisionTreeClassifier(**dt_args).fit(X_use, y_global)
    dt_0ng = DecisionTreeClassifier(**dt_args).fit(X_use, y_0ng)
    dt_10ng = DecisionTreeClassifier(**dt_args).fit(X_use, y_10ng)

    leaf_global, prec_global = get_best_leaf_node(dt_global, X_use, y_global)
    leaf_0ng, prec_0ng = get_best_leaf_node(dt_0ng, X_use, y_0ng)
    leaf_10ng, prec_10ng = get_best_leaf_node(dt_10ng, X_use, y_10ng)
    
    y_umap_global = extract_and_apply_rules(dt_global, feature_names, leaf_global, X_use, "Global Baseline")
    y_umap_0ng = extract_and_apply_rules(dt_0ng, feature_names, leaf_0ng, X_use, "0 ng/mL Baseline")
    y_umap_10ng = extract_and_apply_rules(dt_10ng, feature_names, leaf_10ng, X_use, "10 ng/mL Baseline")

    dataset_title = f"{concentration[con_idx]} ng/mL"

    # Generating UMAPs
    print("Generating UMAPs...")
    reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(X_subset) 

    umap_configs = [
        (y_umap_global, "Global", "Baseline: Global Average", leaf_global, prec_global),
        (y_umap_0ng, "0ng", "Baseline: 0 ng/mL Average", leaf_0ng, prec_0ng),
        (y_umap_10ng, "10ng", "Baseline: 10 ng/mL Average (Local)", leaf_10ng, prec_10ng)
    ]

    for y_data, baseline_name, title, leaf_idx, leaf_prec in umap_configs:
        plt.figure(figsize=(10, 10))
        red_count = np.sum(y_data)
        
        plt.scatter(embedding[y_data == 0, 0], embedding[y_data == 0, 1], c='lightgrey', s=5, alpha=0.5, label='Rest')
        plt.scatter(embedding[y_data == 1, 0], embedding[y_data == 1, 1], c='red', s=3, alpha=0.8, 
                    label=f'Best Node (ID: {leaf_idx} | Prec: {leaf_prec*100:.1f}% | n={red_count})')
        
        plt.title(f"UMAP Analyzed Dataset: {dataset_title}\n{title}", fontsize=16)
        plt.legend(loc="upper right")
        plt.axis('off')
        plt.tight_layout()
        
        umap_file = f'UMAP_Dataset_{con_idx}ng_Baseline_{baseline_name}_feat{args.feature_idx}.png'
        plt.savefig(os.path.join(out_dir, umap_file), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()