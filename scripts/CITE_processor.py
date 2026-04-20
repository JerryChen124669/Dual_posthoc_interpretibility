import scanpy as sc
import pandas as pd
import scipy.sparse as sp
import numpy as np
import os

# =====================================================================
# GUIDANCE: CITE-SEQ PREPROCESSING SCRIPT
# This script processes matched single-cell RNA and surface protein 
# measurements (CITE-seq data). It dynamically identifies the modalities, 
# applies quality control, extracts Highly Variable Genes (HVGs), and 
# standardizes both inputs and targets for downstream deep learning.
# =====================================================================

# --- CONFIGURATION ---
# GUIDANCE: Set to access the 'data/CITEseq_immune' directory relative to the 'scripts' folder
DATA_DIR = os.path.join('..', 'data', 'CITEseq_immune')

FILE_MOD1 = os.path.join(DATA_DIR, 'dataset_mod1.h5ad')
FILE_MOD2 = os.path.join(DATA_DIR, 'dataset_mod2.h5ad')
OUTPUT_FILE = os.path.join(DATA_DIR, 'CITEseq_1500HVG_13TG_FullyScaled.xlsx') 
NUM_GENES = 1500

def extract_dataframe(adata):
    """
    GUIDANCE: Helper function to safely extract the expression matrix 
    (whether sparse or dense) into a Pandas DataFrame for final export.
    """
    matrix = adata.X
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return pd.DataFrame(matrix, index=adata.obs_names, columns=adata.var_names)

def main():
    print(f"Loading CITE-seq data from {DATA_DIR}...")
    try:
        adata1 = sc.read_h5ad(FILE_MOD1)
        adata2 = sc.read_h5ad(FILE_MOD2)
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the files. Details: {e}")
        return

    # =====================================================================
    # STEP 1: DYNAMICALLY IDENTIFY MODALITIES
    # =====================================================================
    # GUIDANCE: Transcriptomics (genes) typically have >10,000 features, 
    # whereas ADT (surface proteins) typically have <200. This logic 
    # automatically assigns the correct AnnData object to the right variable.
    if adata1.n_vars > adata2.n_vars:
        adata_genes = adata1
        adata_proteins = adata2
    else:
        adata_genes = adata2
        adata_proteins = adata1

    # =====================================================================
    # STEP 2: EXTRACT NORMALIZED DATA 
    # =====================================================================
    print("Extracting 'normalized' layer into the main data slot...")
    # GUIDANCE: Single-cell data often contains raw counts in .X and log-normalized
    # counts in .layers. We specifically pull the normalized data to avoid 
    # training models on biased sequencing depth artifacts.
    for name, adata_obj in [("Genes", adata_genes), ("Proteins", adata_proteins)]:
        if 'normalized' in adata_obj.layers:
            adata_obj.X = adata_obj.layers['normalized'].copy()
        else:
            available_layers = list(adata_obj.layers.keys())
            print(f"  -> Warning: 'normalized' not found in {name}. Using {available_layers[0]}")
            adata_obj.X = adata_obj.layers[available_layers[0]].copy()
            
        if sp.issparse(adata_obj.X) and not sp.isspmatrix_csr(adata_obj.X):
            adata_obj.X = adata_obj.X.tocsr()

    # =====================================================================
    # STEP 3: ALIGNMENT AND QUALITY CONTROL (QC)
    # =====================================================================
    print("Aligning cells and applying Quality Control...")
    # GUIDANCE: Ensure we only keep cells that have valid data in both modalities
    common_cells = adata_genes.obs_names.intersection(adata_proteins.obs_names)
    adata_genes = adata_genes[common_cells].copy()
    adata_proteins = adata_proteins[common_cells].copy()
    
    # GUIDANCE: Drop low-quality droplets (e.g., debris or empty droplets) 
    # that express fewer than 200 distinct genes.
    sc.pp.filter_cells(adata_genes, min_genes=200)
    
    valid_cells = adata_genes.obs_names
    adata_proteins = adata_proteins[valid_cells].copy()

    # =====================================================================
    # STEP 4: FILTER GENES BY HIGHLY VARIABLE GENE (HVG) SCORE
    # =====================================================================
    print(f"Looking for 'hvg_score' to extract top {NUM_GENES} genes...")
    # GUIDANCE: Using all ~20,000 genes introduces massive noise and leads 
    # to overfitting. We subset to the most biologically informative (variable) genes.
    if 'hvg_score' in adata_genes.var.columns:
        top_hvg_names = adata_genes.var.nlargest(NUM_GENES, 'hvg_score').index
        adata_genes = adata_genes[:, top_hvg_names].copy()
    elif 'hvg' in adata_genes.var.columns:
        adata_genes = adata_genes[:, adata_genes.var['hvg'] == True].copy()
        if adata_genes.n_vars > NUM_GENES:
            variances = np.var(adata_genes.X.toarray() if sp.issparse(adata_genes.X) else adata_genes.X, axis=0)
            top_indices = np.argsort(variances)[-NUM_GENES:]
            adata_genes = adata_genes[:, top_indices].copy()
    else:
        print("Warning: Neither hvg_score nor hvg found. Please check your var columns.")
        return

    # =====================================================================
    # STEP 5 & 6: STANDARDIZE INPUTS AND TARGETS
    # =====================================================================
    # GUIDANCE: Z-score normalization is essential for Deep Learning. 
    # It ensures that highly expressed genes/proteins do not disproportionately 
    # dominate the network's weight updates compared to low-expression genes.
    print("Scaling (Z-scoring) the Input Genes...")
    if sp.issparse(adata_genes.X):
        adata_genes.X = adata_genes.X.toarray()
    sc.pp.scale(adata_genes, max_value=10)

    print("Scaling (Z-scoring) the Target Proteins...")
    if sp.issparse(adata_proteins.X):
        adata_proteins.X = adata_proteins.X.toarray()
    sc.pp.scale(adata_proteins, max_value=10)

    # =====================================================================
    # STEP 7 & 8: EXTRACT, COMBINE, AND SAVE
    # =====================================================================
    print("Formatting final dataset...")
    df_genes = extract_dataframe(adata_genes)
    df_proteins = extract_dataframe(adata_proteins)
    
    # Concatenate the inputs and targets horizontally for easy loading during ML training
    final_df = pd.concat([df_genes, df_proteins], axis=1)

    print(f"Saving to {OUTPUT_FILE}...")
    final_df.to_excel(OUTPUT_FILE)
    print("Success! Both Inputs and Targets are now standardized for Machine Learning.")

if __name__ == "__main__":
    main()