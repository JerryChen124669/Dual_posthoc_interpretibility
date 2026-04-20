import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        return self.linear(x).squeeze()

    def initialize(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)


class MultiLinearRegression(nn.Module):
    def __init__(self, input_size, n_models, n_egf, dropout, device):
        super(MultiLinearRegression, self).__init__()
        self.dev = device

        self.linear = nn.Linear(input_size, n_models)
        self.emb = torch.nn.Embedding(n_egf, n_models)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x, egf_idx):
        x = self.linear(x)
        # original
        # coef = self.emb(torch.LongTensor(egf_idx).to(self.dev))
        # gemini-modified version below:
        coef = self.emb(egf_idx.to(device=self.dev, dtype=torch.long))
        coef = self.dropout(coef)
        return torch.sum(x * coef, dim=1)

    def initialize(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)
        nn.init.xavier_uniform_(self.emb.weight)


class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(MLPRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()

    def initialize(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)


class MultiMLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size, n_models, n_egf, dropout, device):
        super(MultiMLPRegression, self).__init__()
        self.dev = device

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, n_models)
        self.emb = torch.nn.Embedding(n_egf, n_models)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x, egf_idx):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        coef = self.emb(torch.LongTensor(egf_idx).to(self.dev))
        # coef = self.dropout(coef)
        return torch.sum(x * coef, dim=1)

    def initialize(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.xavier_uniform_(self.emb.weight)
