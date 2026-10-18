import torch
from torch import nn

class MLP(nn.Module):
    """
    Standard Multilayer Perceptron for Network Intrusion Detection.
    
    Architecture:
    - Input: input_dim (192 processed features from UNSW-NB15)
    - Hidden Layer 1: 128 units, ReLU, Dropout(0.20)
    - Hidden Layer 2: 64 units, ReLU, Dropout(0.20)
    - Hidden Layer 3: 32 units, ReLU
    - Output: 1 unit (logit for BCEWithLogitsLoss binary classification)
    """
    def __init__(self, input_dim=192):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)
