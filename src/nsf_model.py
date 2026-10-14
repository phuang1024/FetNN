"""Neural spline flow generative model.
Maps N(0, 1) -> P(x).
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.nn.nets import ResidualNet

from dataset import *

# Model the performance space.
NSF_DIM = X_DIM + Y_DIM
LAYERS = 6
HIDDEN_DIM = 128


def create_rq_nsf():
    transforms = []
    base_dist = StandardNormal(shape=[NSF_DIM])

    for i in range(LAYERS):
        mask = torch.arange(NSF_DIM) % 2
        if i % 2 == 1:
            mask = 1 - mask

        def create_net(in_features, out_features):
            return ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=HIDDEN_DIM,
                num_blocks=2,
                activation=torch.relu,
            )

        transforms.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=create_net,
                num_bins=8,
                tails="linear",
                tail_bound=5,
            )
        )
        transforms.append(RandomPermutation(features=NSF_DIM))

    return Flow(
        transform=CompositeTransform(transforms),
        distribution=base_dist,
    )


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO tune this.
EPOCHS = 100
BATCH_SIZE = 64
LR = 2e-3

epoch = 0
global_step = 0


@torch.no_grad()
def augment_data(x, y, noise=1e-3):
    # Random noise.
    x = x + torch.randn_like(x) * noise
    y = y + torch.randn_like(y) * noise
    return x, y


def train(dataset, model, optim, train_loader, writer):
    global global_step
    model.train()

    for x, y in train_loader:
        #x, y = augment_data(x, y)
        cat_xy = torch.cat([x, y], dim=1)
        log_p = model.log_prob(cat_xy)

        loss = -torch.mean(log_p)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()
        optim.zero_grad()

        writer.add_scalar("train/loss", loss.item(), global_step)
        writer.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)
        global_step += 1


@torch.no_grad()
def val(dataset, model, val_loader, writer):
    model.eval()

    total_loss = 0
    for x, y in val_loader:
        cat_xy = torch.cat([x, y], dim=1)
        log_p = model.log_prob(cat_xy)

        # MSE loss.
        loss = -torch.mean(log_p)
        total_loss += loss.item()

    total_loss /= len(val_loader)
    writer.add_scalar("val/loss", total_loss, global_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()

    # Make datasets.
    dataset = FetDataset(args.data, DEVICE)
    train_loader, val_loader = split_train_val(dataset, 0.8, BATCH_SIZE)

    # Make model.
    model = create_rq_nsf().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 25, 0.6)

    writer = SummaryWriter(args.log_dir)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num params:", num_params)

    global epoch
    for epoch in trange(EPOCHS):
        train(dataset, model, optim, train_loader, writer)
        val(dataset, model, val_loader, writer)
        lr_scheduler.step()

    torch.save(model.state_dict(), args.log_dir / f"model_e{epoch}.pt")


if __name__ == "__main__":
    main()
