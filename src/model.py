"""DNN forward surrogate.
"""

import torch
import torch.nn as nn

from fet_data import X_DIM, Y_DIM


class FetDNNModel(nn.Module):
    # TODO tune this.
    dim_hidden = 32

    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(X_DIM, self.dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(self.dim_hidden, Y_DIM),
        )

    def forward(self, x):
        return self.mlp(x)
