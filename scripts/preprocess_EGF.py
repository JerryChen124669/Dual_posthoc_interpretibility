import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# =====================================================================
# GUIDANCE: DATA PREPROCESSING & STANDARDIZATION
# This script processes the raw MATLAB data containing multimodal 
# single-cell measurements (transcriptomic + morphological). 
# Z-score standardization is strictly applied to ensure the neural network's 
# gradients do not bias toward features with naturally larger numeric 
# ranges (e.g., raw cell area vs. log-fold gene expression). It also 
# trims redundant features to create the optimized 510-dimension input.
# =====================================================================

DATA_DIR = os.path.join('..', 'data', 'EGF_perturb')
os.makedirs(DATA_DIR, exist_ok=True)

def main():
    print(f"Loading raw MATLAB data from {DATA_DIR}...")
    mat = h5py.File(os.path.join(DATA_DIR, 'Data_1.mat'), 'r')

    # --- Extract FeatureHeader names ---
    # MATLAB cell arrays of strings are stored as object references in h5py.
    # We loop through the references and convert the ascii codes back to strings.
    feature_names = []
    for ref in mat['FeatureHeader'][:].flatten():
        char_array = mat[ref][:]
        name = ''.join(chr(int(c)) for c in char_array.flatten())
        feature_names.append(name)
    feature_names = np.array(feature_names)

    # --- Extract core matrices ---
    LinearIndex = np.array(mat['LinearIndex']).T.squeeze()
    FeatureData = np.array(mat['FeatureData']).T
    
    # Subset to the exact experimental conditions used in the study
    data_idx = np.where(np.isin(LinearIndex, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20]))[0]
    FeatureData = FeatureData[data_idx, :]
    print(f"Original FeatureData shape before pruning: {FeatureData.shape}")

    scaler = StandardScaler()

    # GUIDANCE: The raw dataset contains both log2-transformed (1-650) 
    # and untransformed (651-1300) versions of the same features.
    X = FeatureData[:, 0:650]
    X_names = feature_names[0:650]

    # Revert specific features back to their non-log forms based on biological distribution
    nolog = np.r_[28, 30, 40, 41, 42, 48, 49, 53, 55, 57, 414:418, 433, 446, 447, 452, 461, 462, 463, 487, 493:498,
            501, 542, 543, 544, 565:571, 640, 645, 647, 648]
    nolog2 = nolog + 650
    X[:, nolog] = FeatureData[:, nolog2]

    # Standardize data to stabilize MLP training
    X = scaler.fit_transform(X)

    # Remove irrelevant metadata features (e.g., pure coordinate data like CenterX/CenterY)
    use = np.r_[0:28, 29, 31:40, 43:48, 50:53, 54, 56, 58:414, 418:433, 434:446, 448:452, 453:461,
            464:487, 488:493, 498:501, 502:542, 545:565, 571:640, 641:645, 646, 649]
    
    X = X[:, use]
    final_feature_names = X_names[use]

    print(f"Final optimized dataset shape: {X.shape}")

    # Export finalized ML-ready arrays
    out_npy = os.path.join(DATA_DIR, 'FeatureData-z-510.npy')
    np.save(out_npy, X)
    
    out_txt = os.path.join(DATA_DIR, 'feature_name.txt')
    with open(out_txt, 'w') as f:
        for name in final_feature_names:
            f.write(f"{name}\n")
            
    print(f"✅ Preprocessing complete. Saved inputs to {DATA_DIR}.")

if __name__ == '__main__':
    main()