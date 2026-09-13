"""Flow matching.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdiffeq import odeint
from tqdm import trange

from data import LdmosDegrData, split_train_val

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLOW_SIGMA = 0

EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-4

epoch = 0
global_step = 0


class FlowFetModel(nn.Module):
    # Dimensions.
    dim_latent = 10
    dim_cond = 4
    dim_hidden = 512

    # Exponentially increasing freqs for sin embed.
    embed_dim = 12
    embed_freq_start = 1
    embed_freq_mult = 2

    def __init__(self):
        super().__init__()

        dim_in = (self.dim_latent + self.dim_cond + 1) * self.embed_dim
        self.in_layer = nn.Linear(dim_in, self.dim_hidden)

        # Residual blocks.
        blocks = []
        for _ in range(6):
            blocks.append(nn.Sequential(
                nn.Linear(self.dim_hidden, self.dim_hidden),
                nn.LeakyReLU(),
            ))
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Linear(self.dim_hidden, self.dim_latent)

    def forward(self, xt, cond, time):
        """
        xt: (B, X) latent vector.
        cond: (B, Y) condition.
        time: (B, 1) time in [0, 1].
        """
        # Generate sin embeds for all input values.
        x = torch.cat([xt, cond, time], dim=-1)
        x = torch.cat([self.sin_embed(x[:, i]) for i in range(x.shape[1])], dim=1)

        x = self.in_layer(x)
        for b in self.blocks:
            x = x + b(x)
        x = self.head(x)
        return x

    def sin_embed(self, x):
        """Generate exponentially increasing freq sin embed.
        x: (B,)
        return: (B, D)
        """
        ret = torch.zeros((x.shape[0], self.embed_dim), device=x.device)
        freq = self.embed_freq_start
        for i in range(0, self.embed_dim, 2):
            ret[:, i] = torch.sin(x * freq)
            ret[:, i + 1] = torch.cos(x * freq)
            freq *= self.embed_freq_mult
        return ret


def train(flow_matcher, model, optim, train_loader, writer):
    global global_step
    for x, y in train_loader:
        # Starting Gaussian distr. Flow goes from z0 to x.
        z0 = torch.randn_like(x)
        # t: [0, 1]. xt: x at time. ut: GT velocity at time.
        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(z0, x)
        pred_ut = model(xt, y, t.unsqueeze(1))

        loss = torch.nn.functional.mse_loss(pred_ut, ut)
        loss.backward()
        optim.step()
        optim.zero_grad()

        writer.add_scalar("train/loss", loss.item(), global_step)
        global_step += 1


@torch.no_grad()
def val(flow_matcher, model, val_loader, writer):
    total_vel_loss = 0
    for x, y in val_loader:
        # Sample random time and flow.
        z0 = torch.randn_like(x)
        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(z0, x)
        pred_ut = model(xt, y, t.unsqueeze(1))

        vel_loss = torch.nn.functional.mse_loss(pred_ut, ut)
        total_vel_loss += vel_loss.item()

    # Generate result with ODE solver (using last iter of val_loader).
    def vel_func(t, xt):
        # Expand t to (B, 1)
        t = t.repeat(x.shape[0]).unsqueeze(1)
        # y comes from above for loop.
        return model(xt, y, t)

    ts = torch.linspace(0, 1, 100)
    try:
        trajectory = odeint(vel_func, z0, ts)
        pred_x = trajectory[-1]
        x_loss = torch.nn.functional.mse_loss(pred_x, x)
    except AssertionError as e:
        print(e)
        x_loss = None

    total_vel_loss /= len(val_loader)
    writer.add_scalar("val/vel_loss", total_vel_loss, global_step)
    writer.add_scalar("val/x_loss", x_loss, global_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()

    # Make datasets.
    dataset = LdmosDegrData(args.data, DEVICE)
    train_loader, val_loader = split_train_val(dataset, 0.8, BATCH_SIZE)

    # Make models.
    flow_matcher = ConditionalFlowMatcher(FLOW_SIGMA)

    model = FlowFetModel().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    writer = SummaryWriter(args.log_dir)

    with open(args.log_dir / "model.txt", "w") as f:
        print(model)
        print(model, file=f)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Num params:", num_params)
        print("Num params:", num_params, file=f)

    global epoch
    for epoch in trange(EPOCHS):
        train(flow_matcher, model, optim, train_loader, writer)
        val(flow_matcher, model, val_loader, writer)


if __name__ == "__main__":
    main()
