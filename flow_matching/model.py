"""Flow matching MLP model.
"""

import torch
import torch.nn as nn

# Latent and cond vector size.
# TODO test forward mode
DIM_LATENT = 4#11
DIM_COND = 11#4


class TimeEmbed(nn.Module):
    """T -> Gaussian Fourier features -> small MLP.
    """
    # 2 * number of Fourier features to use.
    feature_dim = 16
    # Fourier feature std dev.
    feature_scale = 1
    # Output dim of MLP.
    out_dim = 16

    def __init__(self):
        super().__init__()
        assert self.feature_dim % 2 == 0

        # (feature_dim / 2,)
        b_vector = torch.randn(self.feature_dim // 2, 1) * (self.feature_scale * 2 * torch.pi)
        self.register_buffer("b_vector", b_vector)

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.out_dim),
            nn.SiLU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

    def forward(self, t):
        """
        t: (B, 1) time in [0, 1]
        return: (B, 16)
        """
        # (B, feature_dim / 2)
        features = torch.matmul(t, self.b_vector.T)
        # (B, feature_dim)
        features = torch.cat((
            torch.cos(features),
            torch.sin(features),
        ), dim=-1)

        # (B, out_dim)
        embeds = self.mlp(features)
        return embeds


class ResBlock(nn.Module):
    """Residual MLP.
    """

    def __init__(self, dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        """
        x, return: (B, D)
        """
        return x + self.mlp(x)


class FlowFetModel(nn.Module):
    """Residual MLP model that predicts velocity.
    """

    hidden_dim = 64

    def __init__(self):
        super().__init__()

        self.time_embedder = TimeEmbed()

        # Projection of input values to hidden_dim.
        self.in_proj = nn.Linear(DIM_LATENT + DIM_COND + TimeEmbed.out_dim, self.hidden_dim)

        blocks = []
        for _ in range(3):
            blocks.append(ResBlock())
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Linear(self.hidden_dim, DIM_LATENT)

    def forward(self, xt, cond, time):
        """
        xt: (B, X) latent vector.
        cond: (B, Y) condition.
        time: (B, 1) time in [0, 1].
        """
        # Time embed.
        t_embeds = self.time_embedder(time)
        # Concat inputs. (B, Dx + Dy + Dt)
        x = torch.cat([xt, cond, t_embeds], dim=-1)
        # Input projection.
        x = self.in_proj(x)

        # Res blocks.
        for b in self.blocks:
            x = b(x)

        # Output proj.
        x = self.head(x)
        return x

    def init_weights(self, init_std=1e-2):
        for param in self.parameters():
            if param.requires_grad:
                param.data = init_std * torch.randn_like(param)
