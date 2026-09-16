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

EPOCHS = 300
BATCH_SIZE = 64
LR = 7e-3

epoch = 0
global_step = 0


class FlowFetModel(nn.Module):
    """Residual MLP model that predicts velocity.
    """
    # Dimensions.
    dim_latent = 11
    dim_cond = 4
    dim_hidden = 128

    # Sin embed with exponentially increasing freqs.
    embed_dim = 8
    embed_freq_start = 1
    embed_freq_mult = 2

    def __init__(self):
        super().__init__()

        dim_in = (self.dim_latent + self.dim_cond + 1) * self.embed_dim
        self.input = nn.Linear(dim_in, self.dim_hidden)

        # Residual blocks.
        blocks = []
        for _ in range(6):
            blocks.append(nn.Sequential(
                nn.Linear(self.dim_hidden, self.dim_hidden),
                nn.SiLU(),
                nn.Dropout(0.1),
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
        ret = torch.zeros((x.shape[0], self.embed_dim), device=x.device)
        freq = self.embed_freq_start
        for i in range(0, self.embed_dim, 2):
            ret[:, i] = torch.sin(x * freq)
            ret[:, i + 1] = torch.cos(x * freq)
            freq *= self.embed_freq_mult
        return ret

    def init_weights(self, init_std=1e-2):
        for param in self.parameters():
            if param.requires_grad:
                param.data = init_std * torch.randn_like(param)


def train(flow_matcher, model, optim, train_loader, writer):
    global global_step
    model.train()
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
        writer.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)
        global_step += 1

    # Generate X sample.
    model.eval()
    pred_x = generate_samples(model, z0, y)
    if pred_x is not None:
        x_loss = torch.nn.functional.mse_loss(pred_x, x)
    else:
        x_loss = None
    writer.add_scalar("train/x_loss", x_loss, global_step)


@torch.no_grad()
def val(flow_matcher, model, val_loader, writer):
    model.eval()
    total_vel_loss = 0
    for x, y in val_loader:
        # Sample random time and flow.
        z0 = torch.randn_like(x)
        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(z0, x)
        pred_ut = model(xt, y, t.unsqueeze(1))

        vel_loss = torch.nn.functional.mse_loss(pred_ut, ut)
        total_vel_loss += vel_loss.item()

    # Generate result with ODE solver (using last iter of val_loader).
    pred_x = generate_samples(model, z0, y)
    if pred_x is not None:
        x_loss = torch.nn.functional.mse_loss(pred_x, x)
    else:
        x_loss = None

    total_vel_loss /= len(val_loader)
    writer.add_scalar("val/vel_loss", total_vel_loss, global_step)
    writer.add_scalar("val/x_loss", x_loss, global_step)


def generate_samples(model, z0, y):
    """Generate X samples given z0 (initial) and y (condition).
    z0: (B, Dz)
    y: (B, Dy)
    """
    def vel_func(t, xt):
        # Expand t to (B, 1)
        t = t.repeat(xt.shape[0]).unsqueeze(1)
        # y comes from above for loop.
        return model(xt, y, t)

    ts = torch.linspace(0, 1, 100)
    try:
        trajectory = odeint(vel_func, z0, ts)
        pred_x = trajectory[-1]
        return pred_x
    except AssertionError as e:
        print(e)
        return None


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
    model.init_weights()
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 75, 0.7)

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
        lr_scheduler.step()


if __name__ == "__main__":
    main()
