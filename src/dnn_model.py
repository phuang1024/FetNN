"""DNN forward surrogate.
"""

import torch
import torch.nn as nn


class FetDNNModel(nn.Module):
    dim_hidden = 128

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim_in, self.dim_hidden),
            nn.SiLU(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.SiLU(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.SiLU(),
            nn.Linear(self.dim_hidden, dim_out),
        )

    def forward(self, x):
        return self.mlp(x)
