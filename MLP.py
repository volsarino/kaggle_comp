import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

class MLP(nn.Module):
    def __init__(self, num_hiddens_1, num_hiddens_2, dropout_1, dropout_2, num_outputs=1):
        super().__init__()
        self.net=nn.Sequential(nn.LazyLinear(num_hiddens_1),
                               nn.ReLU(),nn.Dropout(dropout_1),#過学習対策（ドロップアウト）
                               nn.BatchNorm1d(num_hiddens_1),
                               nn.LazyLinear(num_hiddens_2),nn.ReLU(),
                               nn.Dropout(dropout_2),
                               nn.BatchNorm1d(num_hiddens_2),
                               nn.LazyLinear(num_outputs))
    def forward(self,X):
        return self.net(X)
    