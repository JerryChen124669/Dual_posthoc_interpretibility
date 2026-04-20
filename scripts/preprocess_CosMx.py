import pandas as pd
import numpy as np
import time
import sys
import os

# =====================================================================
# GUIDANCE: STEP 1 - RUN THIS SCRIPT FIRST
# This script processes the raw CosMx spatial transcriptomics data.
# It identifies the cancer cells via dual-gating, calculates the variance 
# of genes, standardizes the expression, and merges it with the protein target.
# 
# IMPORTANT: Run this script BEFORE running 'Data_Reducer.py'.
# =====================================================================

# --- CONFIGURATION ---
# GUIDANCE: Steps back out of the 'scripts' folder and accesses 'data/CosMx_cancer'
DATA_DIR = os.path.join('..', 'data', 'CosMx_cancer')

# Note: Adjust the internal folder structure below if your unzipped Flatfiles are placed differently inside 'CosMx_cancer'
rna_file_path = os.path.join(DATA_DIR, 'Flatfiles_RNA', 'Flatfiles_RNA', 'flatFiles', 'BreastCancer', 'BreastCancer_exprMat_file.csv.gz')
protein_file_path = os.path.join(DATA_DIR, 'Flatfiles_Protein', 'Flatfiles_Protein', 'flatFiles', 'BreastCancer', 'BreastCancer_exprMat_file.csv.gz') 

target_protein_name = "Ki-67"
positive_marker = "Channel-PanCK"  # Tumor/Epithelial marker
negative_marker = "Channel-CD45"   # Immune/Stromal marker 
CHUNK_SIZE = 20000 
# ---------------------

try:
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def print_status(message):
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}")

def get_gmm_threshold(data_series):
    """
    GUIDANCE: Applies log1p, fits a 2-component Gaussian Mixture Model (GMM), 
    and returns a biological threshold in raw space to isolate positive/negative cells.
    """
    X_log = np.log1p(data_series.values.reshape(-1, 1))
    
    gmm = GaussianMixture(n_components=2, random_state=42)
    labels = gmm.fit_predict(X_log)
    
    mean_0 = X_log[labels == 0].mean()
    mean_1 = X_log[labels == 1].mean()
    
    # Midpoint between the two log-means serves as the gating cutoff
    cutoff_log = np.mean([mean_0, mean_1])
    return np.expm1(cutoff_log)

start_total = time.time()

# =====================================================================
# STEP 0: LOAD PROTEIN & IDENTIFY CANCER POPULATION 
# =====================================================================
print_status("0. POPULATION SELECTION (Filtering for Cancer Cells)...")

if not os.path.exists(protein_file_path):
    print(f"L ERROR: Protein file not found: {protein_file_path}")
    sys.exit()

df_protein = pd.read_csv(protein_file_path)

# Verify markers exist
required_cols = [target_protein_name, positive_marker, negative_marker]
missing = [c for c in required_cols if c not in df_protein.columns]
if missing:
    print(f"L ERROR: Missing columns: {missing}")
    sys.exit()

# --- GATING ALGORITHM ---
# GUIDANCE: We use in-silico flow cytometry gating (PanCK+ / CD45-) to 
# strictly isolate epithelial tumor cells from the complex tumor microenvironment.
print_status(f"   Analyzing {positive_marker} & {negative_marker} expression...")

if SKLEARN_AVAILABLE:
    panck_threshold = get_gmm_threshold(df_protein[positive_marker])
    cd45_threshold = get_gmm_threshold(df_protein[negative_marker])
    
    is_cancer = (
        (df_protein[positive_marker] > panck_threshold) & 
        (df_protein[negative_marker] < cd45_threshold)
    )
    method_name = "Dual-Marker Log-GMM"
else:
    panck_cutoff = np.percentile(df_protein[positive_marker].values, 60) 
    cd45_cutoff = np.percentile(df_protein[negative_marker].values, 40)
    is_cancer = (
        (df_protein[positive_marker] > panck_cutoff) & 
        (df_protein[negative_marker] < cd45_cutoff)
    )
    method_name = "Percentile Fallback"

cancer_cells_df = df_protein[is_cancer].copy()
cancer_index = pd.MultiIndex.from_frame(cancer_cells_df[['fov', 'cell_ID']])
total_cancer_cells = len(cancer_index)

print_status(f"   Method Used: {method_name}")
print_status(f"   Selected {total_cancer_cells:,} Cancer Cells (out of {len(df_protein):,} total).")
print("-" * 60)

if total_cancer_cells == 0:
    print_status("ERROR: No cancer cells found based on thresholds. Exiting.")
    sys.exit()

# =====================================================================
# STEP 1: CALCULATE VARIANCE (ON CANCER CELLS ONLY)
# =====================================================================
# GUIDANCE: Because spatial datasets are massive, we process the RNA file 
# in chunks to avoid overwhelming system RAM.
print_status("1. STARTING PASS 1: Calculating Variance (Cancer subset only)...")

global_n = 0
global_sum = None
global_sum_sq = None

rna_stream = pd.read_csv(rna_file_path, chunksize=CHUNK_SIZE)

for i, chunk in enumerate(rna_stream):
    chunk_indexed = chunk.set_index(['fov', 'cell_ID'])
    
    if i == 0:
        gene_names = chunk_indexed.columns.tolist()
        global_sum = pd.Series(0.0, index=gene_names, dtype='float64')
        global_sum_sq = pd.Series(0.0, index=gene_names, dtype='float64')

    # Fast filter: Keep only cells that passed the protein gating
    cancer_chunk = chunk_indexed[chunk_indexed.index.isin(cancer_index)].astype('float64')
    
    if cancer_chunk.empty:
        continue 

    current_rows = len(cancer_chunk)
    global_n += current_rows
    global_sum += cancer_chunk.sum()
    global_sum_sq += (cancer_chunk ** 2).sum()

    percent = (global_n / total_cancer_cells) * 100
    sys.stdout.write(f"\r   >> Pass 1: {percent:5.1f}% | {global_n} Cancer Cells Processed")
    sys.stdout.flush()

print("\n")
print_status("   Pass 1 Complete. Computing Variance...")

global_mean = global_sum / global_n
global_var = (global_sum_sq - (global_sum**2 / global_n)) / (global_n - 1)
global_std = np.sqrt(global_var)

# Select Top 1000 highly variable genes to remove uninformative/noisy transcripts
top_1000_genes = global_var.sort_values(ascending=False).head(1000).index.tolist()
print_status(f"   Selected Top 1000 Tumor-Variable Genes.")

# =====================================================================
# STEP 2: STANDARDIZE EXPRESSION DATA
# =====================================================================
print("-" * 60)
print_status("2. STARTING PASS 2: Loading & Standardizing...")

cols_to_use = ['fov', 'cell_ID'] + top_1000_genes
processed_chunks = []
processed_count = 0

rna_stream_pass2 = pd.read_csv(rna_file_path, usecols=cols_to_use, chunksize=CHUNK_SIZE)

for i, chunk in enumerate(rna_stream_pass2):
    chunk_indexed = chunk.set_index(['fov', 'cell_ID'])
    cancer_chunk = chunk_indexed[chunk_indexed.index.isin(cancer_index)].copy()
    
    if cancer_chunk.empty:
        continue
        
    # Standardize to Z-scores using the global mean/std calculated in Pass 1
    cancer_chunk[top_1000_genes] = (cancer_chunk[top_1000_genes] - global_mean[top_1000_genes]) / global_std[top_1000_genes]
    
    processed_chunks.append(cancer_chunk.reset_index())
    processed_count += len(cancer_chunk)
    
    percent = (processed_count / total_cancer_cells) * 100
    sys.stdout.write(f"\r   >> Pass 2: {percent:5.1f}%")
    sys.stdout.flush()

print("\n")
print_status("   Concatenating...")
df_rna_final = pd.concat(processed_chunks, ignore_index=True)


# =====================================================================
# STEP 3 & 4: MERGE TARGET AND SAVE
# =====================================================================
print("-" * 60)
print_status("3. MERGING FINAL DATASET...")

df_final = pd.merge(
    df_rna_final,
    df_protein[['fov', 'cell_ID', target_protein_name]],
    on=['fov', 'cell_ID'],
    how='inner'
)

cols = [c for c in df_final.columns if c != target_protein_name] + [target_protein_name]
df_final = df_final[cols]

# GUIDANCE: Output is saved to the data folder so it is ready for Data_Reducer.py
output_filename = os.path.join(DATA_DIR, "cancer_cells_Ki67_dualgaters.csv")
print_status(f"4. Saving to {output_filename}...")
df_final.to_csv(output_filename, index=False)

end_time = time.time()
print_status(f" DONE! Total time: {(end_time - start_total)/60:.1f} minutes.")