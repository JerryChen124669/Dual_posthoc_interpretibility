import h5py
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import argparse
import os

# =====================================================================
# GUIDANCE: PITCH PIPELINE PROMPT GENERATOR
# This script bridges the gap between mathematically derived rules and 
# large language models (LLMs). It processes the attributions, identifies 
# the logical thresholds governing the biological pathways, and formats 
# these into a structured textual prompt. The LLM then uses this prompt 
# to synthesize verifiable biological hypotheses.
# =====================================================================

def parse_arguments():
    parser = argparse.ArgumentParser()
    # PATH UPDATES for the repository structure
    parser.add_argument('--data_dir', type=str, default=os.path.join('..', 'data', 'EGF_perturb'))
    parser.add_argument('--response', type=int, default=3)
    parser.add_argument('--concentrate_idx', type=int, default=3)
    parser.add_argument('--feature_idx', type=int, default=75)
    parser.add_argument('--percentage', type=int, default=90) # '90' for top 10%
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--balance', type=bool, default=False)
    parser.add_argument('--select_feature', type=bool, default=True)
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Setup prompt output directory
    prompt_dir = os.path.join(args.data_dir, 'LLM_Prompts')
    os.makedirs(prompt_dir, exist_ok=True)

    print(f"Loading attribution data from {args.data_dir}...")
    
    mat = h5py.File(os.path.join(args.data_dir, 'Data_1.mat'), 'r')
    LinearIndex = np.array(mat['LinearIndex']).T.squeeze()
    FeatureData = np.load(os.path.join(args.data_dir, 'FeatureData-z-510.npy'))
    d = FeatureData.shape[1]
    
    concentration = [0, 1, 6.25, 10, 25, 100]
    con_idx = args.concentrate_idx
    response_idx = args.response

    # Load precomputed matrices
    W = np.load(os.path.join(args.data_dir, f'W_{response_idx}-{concentration[con_idx]}-dim510-z.npy'))
    attr = np.load(os.path.join(args.data_dir, f'attr_{response_idx}-{concentration[con_idx]}-dim510-z.npy'))
    X_ig = np.load(os.path.join(args.data_dir, f'X_ig_{response_idx}-{concentration[con_idx]}-dim510-z.npy'))

    # Hardcoded subset configuration for specific PITCH figures
    if args.select_feature:
        use_list = [0, 1, 7, 13, 14, 20, 25, 26, 32, 37, 38, 44, 50, 51, 57, 63, 67, 75, 101, 102, 110, 134, 151]
        use_map = {0: 0, 1: 1, 7: 2, 13: 3, 14: 4, 20: 5, 25: 6, 26: 7, 32: 8, 37: 9,
                   38: 10, 44: 11, 50: 12, 51: 13, 57: 14, 63: 15, 67: 16, 75: 17, 101: 18, 102: 19,
                   110: 20, 134: 21, 151: 22}
        final_feature_indices = use_list.copy()
        X_use = X_ig[:, use_list]
        idx_to_drop = use_map[args.feature_idx]
        final_feature_indices.pop(idx_to_drop)
        X_use = X_use[:, np.r_[0:idx_to_drop, idx_to_drop + 1:22]]
    else:
        final_feature_indices = list(range(314))
        X_use = X_ig[:, 0:314]
        final_feature_indices.remove(args.feature_idx)
        X_use = X_use[:, np.r_[0:args.feature_idx, args.feature_idx + 1:314]]

    # Load biological feature names
    all_feature_names = []
    try:
        with open(os.path.join(args.data_dir, 'feature_name.txt'), 'r') as file:
            for line in file:
                all_feature_names.append(line.strip())
        feature_names = [all_feature_names[i] for i in final_feature_indices]
        target_feat_name = all_feature_names[args.feature_idx]
    except FileNotFoundError:
        feature_names = [f"Feature_{i}" for i in final_feature_indices]
        target_feat_name = f"Feature_{args.feature_idx}"

    # Binarize Attributions for Tree Target
    label_by = attr[:, args.feature_idx]
    thold = np.percentile(label_by, args.percentage)
    y = np.where(label_by > thold, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(X_use, y, test_size=0.2, random_state=42)

    # Train the Logic Extraction Tree
    print("Training Decision Tree to extract AI rules...")
    model = DecisionTreeClassifier(max_depth=args.max_depth, max_leaf_nodes=14, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)

    # Validate Tree Performance
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Tree Accuracy: {acc:.4f}")

    # ==========================================
    # EXTRACT LEAF NODE CONTEXT FOR THE LLM
    # ==========================================
    leaf_ids = model.apply(X_train)
    unique_leaves = np.unique(leaf_ids)
    
    leaf_context = ""
    for leaf in unique_leaves:
        in_leaf = (leaf_ids == leaf)
        n_samples = np.sum(in_leaf)
        if n_samples > 0:
            class_1_count = np.sum(y_train[in_leaf] == 1)
            class_0_count = np.sum(y_train[in_leaf] == 0)
            precision = class_1_count / n_samples
            leaf_context += f"Zone ID {leaf}: Total Cells = {n_samples}, True Positives (Class 1) = {class_1_count}, False Positives (Class 0) = {class_0_count}, Precision = {precision*100:.1f}%\n"

    # Export tree directly as text rules
    tree_rules = export_text(model, feature_names=feature_names)

    # ==========================================
    # FORMAT AND SAVE LLM PROMPT
    # ==========================================
    ai_readable_text = (
        f"--- CONTEXT ---\n"
        f"You are a computational biologist interpreting rules derived from a neural network using the PITCH pipeline.\n"
        f"The data represents single-cell responses to {concentration[con_idx]} ng/mL EGF stimulation. "
        f"We used Integrated Gradients to identify which cells had {target_feat_name} as a highly influential positive driver of pERK activation.\n"
        f"We then trained a decision tree to extract the specific transcriptomic/morphological signatures of these high-attribution cells.\n\n"
        f"--- TARGET ZONES ---\n"
        f"These are the specific logic zones that successfully isolated 'High Attribution' samples. "
        f"Prioritize analyzing nodes with high True Positives (TP) and high Precision.\n\n"
        f"{leaf_context}\n"
        f"--- FULL DECISION TREE RAW RULES ---\n"
        f"{tree_rules}\n\n"
        f"--- TASK ---\n"
        f"Based on the TARGET ZONES above, formulate a hypothesis explaining HOW these specific feature combinations integratively regulate pERK via {target_feat_name}. "
        f"Highlight which zones are the most biologically robust based on the highest number of True Positives and Precision. "
        f"Categorize your findings into: 1. Confirmations of existing literature, and 2. Novel testable mechanisms supported by biological rationale. "
        f"Please list relevant publications supporting these conclusions."
    )

    txt_filename = os.path.join(prompt_dir, f'pERK_EGF{concentration[con_idx]}ng_DT_prompt.txt')
    
    with open(txt_filename, "w") as f:
        f.write(ai_readable_text)
    print(f"\n✅ AI Readable LLM Prompt saved successfully to: {txt_filename}")

if __name__ == '__main__':
    main()