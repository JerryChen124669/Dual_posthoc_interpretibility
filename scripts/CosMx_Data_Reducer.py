import pandas as pd
import numpy as np
import os

# =====================================================================
# GUIDANCE: STEP 2 - RUN THIS SCRIPT SECOND
# IMPORTANT: You must run 'preprocess_CosMx.py' before running this script!
# 
# This script takes the massive output file from Step 1 and further reduces 
# the data. It selects the subset of genes most correlated with the target 
# protein and downsamples the cells. This creates a lightweight dataset 
# optimized for training the multi-layer perceptron (MLP).
# =====================================================================

# --- CONFIGURATION ---
# GUIDANCE: Steps back out of the 'scripts' folder and accesses 'data/CosMx_cancer'
DATA_DIR = os.path.join('..', 'data', 'CosMx_cancer')

input_file = os.path.join(DATA_DIR, "cancer_cells_Ki67_dualgaters.csv")
output_file = os.path.join(DATA_DIR, "cancer_cells_Ki67_dualgaters_cleaned_500.csv")
target_column = "Ki-67"

# Target Dimensions
TARGET_GENE_COUNT = 500   # Reduce from 1000 -> 500 features
TARGET_CELL_COUNT = 30000 # Final clean cell count
# ---------------------

print(f"Reading huge file: {input_file} ...")
print("   (This uses ~2GB RAM, please wait...)")

df = pd.read_csv(input_file)
print(f"   Original Shape: {df.shape}")
print("-" * 50)

# Identify gene columns (exclude spatial metadata and target protein)
metadata_cols = ['fov', 'cell_ID']
gene_cols = [c for c in df.columns if c not in metadata_cols and c != target_column]

# =====================================================================
# STEP 1: CELL QUALITY CONTROL 
# =====================================================================
print("1. Skipping outlier cropping (keeping all cells intact)...")

# GUIDANCE: We intentionally bypass outlier removal here to allow the 
# deep learning model to learn robustly across extreme biological states.
df_clean = df.copy()

print(f"   Cells remaining after bypassing QC: {len(df_clean)}")
print("-" * 50)

# =====================================================================
# STEP 2: PROTEIN TARGET FIX
# =====================================================================
print(f"2. Normalizing Protein Target ({target_column})...")

# GUIDANCE: The 99th percentile clipping (np.clip) was deliberately removed.
# Instead, we apply a log1p transformation so the neural network can still 
# learn the gradients smoothly without getting derailed by massive outliers.
df_clean[target_column] = np.log1p(df_clean[target_column])

print(f"   Applied log1p transform (Extreme outlier capping removed).")
print("-" * 50)

# =====================================================================
# STEP 3: GENE REDUCTION (Feature Selection)
# =====================================================================
print(f"3. Selecting Top {TARGET_GENE_COUNT} genes correlated with normalized {target_column}...")

# GUIDANCE: Calculate correlation between all standardized genes and the target protein
correlations = df_clean[gene_cols].corrwith(df_clean[target_column]).abs()

# Pick the top N genes with highest predictive value
top_genes = correlations.sort_values(ascending=False).head(TARGET_GENE_COUNT).index.tolist()

print(f"   Selected {len(top_genes)} features.")
print(f"   Top predictor: {top_genes[0]} (corr: {correlations[top_genes[0]]:.4f})")

# Construct new DataFrame with only these selected features
cols_to_keep = metadata_cols + top_genes + [target_column]
df_reduced = df_clean[cols_to_keep]
print("-" * 50)

# =====================================================================
# STEP 4: CELL REDUCTION (Downsampling)
# =====================================================================
print(f"4. Downsampling to {TARGET_CELL_COUNT} cells...")

if len(df_reduced) > TARGET_CELL_COUNT:
    # GUIDANCE: Randomly sample cells (random_state=42 ensures exact reproducibility for publication)
    df_final = df_reduced.sample(n=TARGET_CELL_COUNT, random_state=42)
else:
    print("   Dataset is already smaller than target count. Keeping all cells.")
    df_final = df_reduced

print(f"   New Shape: {df_final.shape}")

# =====================================================================
# STEP 5: SAVE FINAL REDUCED DATASET
# =====================================================================
print("-" * 50)
print(f"Saving to {output_file}...")
df_final.to_csv(output_file, index=False)

# Check final file size to ensure it is lightweight enough for modeling
size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"Done! Final File Size: {size_mb:.2f} MB")