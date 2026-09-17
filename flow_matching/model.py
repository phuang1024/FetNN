"""Flow matching MLP model.
"""

import torch
import torch.nn as nn

# Dimensions.
DIM_LATENT = 11
DIM_COND = 4
DIM_HIDDEN = 128

# Sin embed with exponentially increasing freqs.
EMBED_DIM = 8
EMBED_FREQ_START = 1
EMBED_FREQ_MULT = 2


class FlowFetModel(nn.Module):
    """Residual MLP model that predicts velocity.
    """
    def __init__(self):
        super().__init__()

        dim_in = (DIM_LATENT + DIM_COND + 1) * EMBED_DIM
        self.input = nn.Linear(dim_in, DIM_HIDDEN)

        # Residual blocks.
        blocks = []
        for _ in range(6):
            blocks.append(nn.Sequential(
                nn.Linear(DIM_HIDDEN, DIM_HIDDEN),
                nn.SiLU(),
                nn.Dropout(0.1),
            ))
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Linear(DIM_HIDDEN, DIM_LATENT)

    def forward(self, xt, cond, time):
        """
        xt: (B, X) latent vector.
        cond: (B, Y) condition.
        time: (B, 1) time in [0, 1].
        """
        # Generate sin embeds for all input values.
        x = torch.cat([xt, cond, time], dim=-1)
        x = torch.cat([self.sin_embed(x[:, i]) for i in range(x.shape[1])], dim=1)

        x = self.input(x)
        for b in self.blocks:
            x = x + b(x)
        x = self.head(x)
        return x

    def sin_embed(self, x):
        """Generate exponentially increasing freq sin embed.
        x: (B,)
        return: (B, D)
        """
        ret = torch.zeros((x.shape[0], EMBED_DIM), device=x.device)
        freq = EMBED_FREQ_START
        for i in range(0, EMBED_DIM, 2):
            ret[:, i] = torch.sin(x * freq)
            ret[:, i + 1] = torch.cos(x * freq)
            freq *= EMBED_FREQ_MULT
        return ret

    def init_weights(self, init_std=1e-2):
        for param in self.parameters():
            if param.requires_grad:
                param.data = init_std * torch.randn_like(param)
