import h5py
import numpy as np
import pandas as pd
import os
import argparse
from itertools import combinations
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# =====================================================================
# GUIDANCE: EMPIRICAL THRESHOLD OPTIMIZATION
# A common critique of post-hoc interpretability pipelines is the arbitrary 
# selection of binarization thresholds (e.g., picking "Top 10%" randomly). 
# This script systematically evaluates different attribution percentiles 
# using cross-validation. It tracks F1-Score (classification accuracy) and 
# the Jaccard Index (tree rule stability) to mathematically justify the 
# threshold that yields the most robust and verifiable biological rules.
# =====================================================================

def parse_arguments():
    parser = argparse.ArgumentParser()
    # PATH UPDATES for repository structure
    parser.add_argument('--data_dir', type=str, default=os.path.join('..', 'data', 'EGF_perturb'))
    parser.add_argument('--response', type=int, default=3)
    parser.add_argument('--feature_idx', type=int, default=75)
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--balance', type=bool, default=False)
    parser.add_argument('--select_feature', type=bool, default=True)
    parser.add_argument('--self_exclude', type=bool, default=True)
    return parser.parse_args()

def compute_pairwise_jaccard(feature_sets):
    """Calculates the average Jaccard index across multiple sets of tree logic rules."""
    if not feature_sets or len(feature_sets) < 2:
        return 0.0
    
    jaccards = []
    for s1, s2 in combinations(feature_sets, 2):
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        if union == 0:
            jaccards.append(1.0)
        else:
            jaccards.append(intersection / union)
    return np.mean(jaccards)

def main():
    args = parse_arguments()
    out_dir = os.path.join(args.data_dir, 'Threshold_Validation')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading matrices from {args.data_dir}...")
    mat = h5py.File(os.path.join(args.data_dir, 'Data_1.mat'), 'r')
    LinearIndex = np.array(mat['LinearIndex']).T.squeeze()
    FeatureData = np.load(os.path.join(args.data_dir, 'FeatureData-z-510.npy'))
    
    concentration = [0, 1, 6.25, 10, 25, 100]
    datanum = [[13, 14, 15], [16, 17, 18], [10, 19, 20], [7, 8, 9], [4, 5, 6], [1, 2, 3]]
    response_idx = args.response

    # Evaluate multiple percentiles to find the 'sweet spot'
    target_percentages = [50, 60, 70, 80, 90, 95]
    all_results = []

    # Hardcoded subsetting to match exact paper figures
    if args.select_feature:
        use_list = [0, 1, 7, 13, 14, 20, 25, 26, 32, 37, 38, 44, 50, 51, 57, 63, 67, 75, 101, 102, 110, 134, 151]
        use_map = {0: 0, 1: 1, 7: 2, 13: 3, 14: 4, 20: 5, 25: 6, 26: 7, 32: 8, 37: 9,
                   38: 10, 44: 11, 50: 12, 51: 13, 57: 14, 63: 15, 67: 16, 75: 17, 101: 18, 102: 19,
                   110: 20, 134: 21, 151: 22}
        final_feature_indices = use_list.copy()
        if args.self_exclude:
            idx_to_drop = use_map[args.feature_idx]
            final_feature_indices.pop(idx_to_drop)

    for con_idx in [1, 3, 5]: # Test across 1, 10, and 100 ng/mL EGF
        print(f"\n--- Testing EGF Concentration: {concentration[con_idx]} ng/mL ---")
        
        try:
            attr = np.load(os.path.join(args.data_dir, f'attr_{response_idx}-{concentration[con_idx]}-dim510-z.npy'))
            X_ig = np.load(os.path.join(args.data_dir, f'X_ig_{response_idx}-{concentration[con_idx]}-dim510-z.npy'))
        except FileNotFoundError:
            print(f"Skipping {concentration[con_idx]} ng/mL - attributions not yet computed.")
            continue

        if args.select_feature:
            X_use = X_ig[:, use_list]
            if args.self_exclude:
                X_use = X_use[:, np.r_[0:idx_to_drop, idx_to_drop + 1:22]]
        else:
            X_use = X_ig[:, 0:314]
            if args.self_exclude:
                X_use = X_use[:, np.r_[0:args.feature_idx, args.feature_idx + 1:314]]

        # Loop through each threshold
        for pct in target_percentages:
            label_by = attr[:, args.feature_idx]
            thold = np.percentile(label_by, pct)
            y = np.where(label_by > thold, 1, 0)
            
            # Cross-validation Setup
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            fold_f1_scores = []
            fold_feature_sets = []

            for train_idx, test_idx in skf.split(X_use, y):
                X_train, X_test = X_use[train_idx], X_use[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if args.balance:
                    rus = RandomUnderSampler(random_state=42)
                    X_train, y_train = rus.fit_resample(X_train, y_train)

                dt = DecisionTreeClassifier(max_depth=args.max_depth, max_leaf_nodes=14, min_samples_leaf=5, random_state=42)
                dt.fit(X_train, y_train)

                y_pred = dt.predict(X_test)
                fold_f1_scores.append(f1_score(y_test, y_pred, average='binary'))

                # Extract features used by this specific tree
                used_features = set()
                for feat in dt.tree_.feature:
                    if feat != -2: # -2 indicates a leaf node in sklearn
                        used_features.add(feat)
                fold_feature_sets.append(used_features)

            # Aggregate cross-validation results
            mean_f1 = np.mean(fold_f1_scores)
            avg_jaccard = compute_pairwise_jaccard(fold_feature_sets)

            all_results.append({
                'Concentration': concentration[con_idx],
                'Top_Percent_Threshold': pct,
                'Mean_F1_Score': mean_f1,
                'Jaccard_Index_Stability': avg_jaccard
            })

    # ==========================================
    # SAVE AND PLOT RESULTS
    # ==========================================
    if all_results:
        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(out_dir, f'threshold_optimization_results_feat{args.feature_idx}.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n✅ Results exported to: {csv_path}")

        metrics_to_plot = [
            ('Mean_F1_Score', 'F1 Score (Model Accuracy)', 'Measures how well the tree separates the classes'),
            ('Jaccard_Index_Stability', 'Jaccard Index (Rule Stability)', 'Measures if the extracted rules stay the same across CV folds')
        ]

        for conc in [1, 10, 100]:
            df_sub = results_df[results_df['Concentration'] == conc]
            if df_sub.empty:
                continue
                
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f'Threshold Optimization for EGF = {conc} ng/mL\n(Target Feature: {args.feature_idx})', fontsize=16, y=1.05)
            
            for i, (col, title, explanation) in enumerate(metrics_to_plot):
                axes[i].plot(df_sub['Top_Percent_Threshold'], df_sub[col], marker='o', linewidth=2, markersize=8, color='teal')
                
                axes[i].set_title(f"{title}\n", fontsize=14, fontweight='bold')
                axes[i].text(0.5, 1.02, explanation, transform=axes[i].transAxes, 
                             fontsize=11, color='dimgray', ha='center', va='bottom')
                
                axes[i].set_xlabel('Top Percent Threshold (%)', fontsize=12)
                axes[i].set_ylabel('Score', fontsize=12)
                axes[i].set_xticks(target_percentages) 
                axes[i].grid(True, linestyle='--', alpha=0.7)
                axes[i].set_ylim(0, 1.05)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.93]) 
            
            plot_path = os.path.join(out_dir, f'threshold_metrics_plot_{conc}ngmL_feat{args.feature_idx}.svg')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"✅ Plot saved to: {plot_path}")

if __name__ == '__main__':
    main()