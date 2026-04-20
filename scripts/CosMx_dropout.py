import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    # GUIDANCE: Since this script runs from inside the 'scripts/' folder, 
    # we use '../' to step back to the root directory, then enter 'data/CosMx_cancer/'
    parser.add_argument('--data_dir', type=str, default=os.path.join('..', 'data', 'CosMx_cancer')) 
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--hidden_size', type=int, default=256) # Matched to pipeline default
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--model_save', type=bool, default=False) 
    return parser.parse_args()

# =====================================================================
# STEP 1: Define Model Architectures
# GUIDANCE: We define both a baseline Linear Regression and a non-linear 
# Multi-Layer Perceptron (MLP) to demonstrate how non-linear models better
# capture biological complexity, especially when data is missing (dropout).
# =====================================================================

class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.initialize()

    def forward(self, x):
        return self.linear(x).squeeze()

    def initialize(self):
        # GUIDANCE: Kaiming initialization helps gradients flow better at the start of training
        nn.init.kaiming_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.