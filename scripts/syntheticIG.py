import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import os

# =====================================================================
# GUIDANCE: STEP 2 - EXTRACTING RULES VIA INTEGRATED GRADIENTS
# We apply IG to the trained MLP to see if it correctly identifies 
# how the input features interact to produce the synthetic output. We then 
# use a decision tree to extract human-readable logic.
# =====================================================================

DATA_DIR = os.path.join('..', 'data', 'synthetic')
os.makedirs(DATA_DIR, exist_ok=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = 'cpu'
d = 4

# Load the model trained in Step 1
model = MLP(input_dim=d, hidden_dim=256, output_dim=1).to(device)
model_path = os.path.join(DATA_DIR, 'Synthetic_MLP.pth')
model.load_state_dict(torch.load(model_path))
model.eval()

# Recreate the validation dataset (must match Step 1)
np.random.seed(0)
X_val = np.random.randn(2000, d)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

# =====================================================================
# 1. COMPUTE INTEGRATED GRADIENTS
# =====================================================================
print("Computing Integrated Gradients...")
ig = IntegratedGradients(model)
baseline = torch.zeros_like(X_val_tensor) # Zero baseline
attributions = ig.attribute(X_val_tensor, baseline, n_steps=50)

attr = attributions.cpu().detach().numpy()

# Export raw attributions
attr_df = pd.DataFrame(attr, columns=[f'Feature_{i}' for i in range(d)])
csv_path = os.path.join(DATA_DIR, 'attribution_scores.csv')
attr_df.to_csv(csv_path, index=False)
print(f"Attributions exported to {csv_path}")

# =====================================================================
# 2. FEATURE SELECTION AND RULE EXTRACTION
# =====================================================================
# Find the feature with the highest average magnitude of impact
mean_abs_attr = np.mean(np.abs(attr), axis=0)
feature_idx = np.argmax(mean_abs_attr)

print(f"\n--- Automatic Feature Selection ---")
for i, score in enumerate(mean_abs_attr):
    print(f"Feature {i} Average Attribution Magnitude: {score:.4f}")
print(f"Decided Column for targeting: Feature {feature_idx}\n")

# Binarize the target: Top 10% of attribution vs the rest
label_by = attr[:, feature_idx]
thold = np.percentile(label_by, 90)
y = np.where(label_by > thold, 1, 0)

# Balance dataset and train Decision Tree
rus = RandomUnderSampler(random_state=42)
X_used = X_val[:, np.r_[0:feature_idx, feature_idx + 1:d]]
X_res, y_res = rus.fit_resample(X_used, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=14, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)

# Evaluation
y_pred = dt_model.predict(X_test)
print("Decision Tree Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Plot and save
plt.figure(figsize=(20, 10))
feature_names = [f'X{i}' for i in range(d) if i != feature_idx]
plot_tree(dt_model, filled=True, feature_names=feature_names, class_names=['Low Attr', 'High Attr'], rounded=True)
plt.title(f'Decision Tree for Predicting High Attribution of X{feature_idx}')

tree_path = os.path.join(DATA_DIR, 'synthetic_tree.png')
plt.savefig(tree_path, dpi=300, bbox_inches='tight')
print(f"Saved Decision Tree Plot to: {tree_path}")