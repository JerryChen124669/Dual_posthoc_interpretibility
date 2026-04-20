import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
import os

# =====================================================================
# GUIDANCE: STEP 1 - SYNTHETIC DATA GENERATION AND TRAINING
# This script generates a synthetic dataset governed by a known, complex 
# non-linear equation. We train an MLP on this data to prove that the 
# model can accurately capture these dynamics before interpreting it.
# =====================================================================

# --- CONFIGURATION ---
DATA_DIR = os.path.join('..', 'data', 'synthetic')
os.makedirs(DATA_DIR, exist_ok=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def synthetic_function(X):
    """
    GUIDANCE: The ground-truth biological rule we are simulating.
    It contains a linear term (X0), a quadratic term (X1), and a complex 
    nonlinear interaction term (X2, X3).
    """
    return 2 * X[:, 0] - 3 * X[:, 1] ** 2 + 5 * sigmoid(X[:, 2]) * (sigmoid(X[:, 1] + X[:, 3]))

device = 'cpu'
np.random.seed(0)
torch.manual_seed(0)

n_samples = 20000
d = 4

# =====================================================================
# 1. DATA GENERATION
# =====================================================================
X = np.random.randn(n_samples, d)
noise = np.random.randn(n_samples) * 0.05
y = synthetic_function(X) + noise
y = y.reshape(-1, 1)

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# =====================================================================
# 2. MODEL DEFINITION & TRAINING
# =====================================================================
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

model = MLP(input_dim=d, hidden_dim=256, output_dim=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training MLP on Synthetic Data...")
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in dataloader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# =====================================================================
# 3. EVALUATION AND EXPORT
# =====================================================================
# Generate validation set to confirm model generalized well
X_val = np.random.randn(2000, d)
y_val = synthetic_function(X_val)
y_val = y_val.reshape(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    predictions = model(X_val_tensor)
predictions = predictions.cpu().detach().numpy().squeeze()

sp_Pearson_test = pearsonr(y_val.squeeze(), predictions)
r2 = r2_score(y_val.squeeze(), predictions)
print(f'Validation Pearson r = {sp_Pearson_test.statistic:.4f}')
print(f'Validation R-squared = {r2:.4f}')

# Export results
val_df = pd.DataFrame({'True_Values': y_val.squeeze(), 'Predictions': predictions})
csv_path = os.path.join(DATA_DIR, 'Validation_Results.csv')
val_df.to_csv(csv_path, index=False)

model_path = os.path.join(DATA_DIR, 'Synthetic_MLP.pth')
torch.save(model.state_dict(), model_path)
print(f"Saved model to: {model_path}")