"""Flow matching.
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm import trange

from data import LdmosDegrData, split_train_val

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3

epoch = 0
global_step = 0


def make_model(dx, dy, dh):
    """Make MLP model.
    dx, dy, dh: Dim of input, output, hidden.
    """
    return torch.nn.Sequential(
        nn.Linear(dx, dh),
        nn.LeakyReLU(),
        nn.Linear(dh, dh),
        nn.LeakyReLU(),
        nn.Linear(dh, dh),
        nn.LeakyReLU(),
        nn.Linear(dh, dy),
    )


def train(flow_matcher, model, optim, train_loader, writer):
    global epoch, global_step
    for epoch in trange(EPOCHS):
        for x, y in train_loader:
            # Starting distr Gaussian. Flow goes from z0 to x.
            z0 = torch.randn_like(x)

            # t: [0, 1]. xt: x at time. ut: velocity at time.
            t, xt, ut = flow_matcher.sample_location_and_conditional_flow(z0, x)

            # TODO check if conditioning done properly.
            inp = torch.cat([xt, y, t.unsqueeze(1)], dim=-1)
            pred_ut = model(inp)

            loss = torch.nn.functional.mse_loss(pred_ut, ut)
            loss.backward()
            optim.step()
            optim.zero_grad()

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("log_dir")
    args = parser.parse_args()

    # Make datasets.
    dataset = LdmosDegrData(args.data, DEVICE)
    train_loader, val_loader = split_train_val(dataset, 0.8, BATCH_SIZE)

    # Make models.
    # TODO param
    flow_matcher = ConditionalFlowMatcher(0.1)

    model = make_model(dataset.data.shape[1] + 1, dataset.x_size, 64)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    writer = SummaryWriter(args.log_dir)

    train(flow_matcher, model, optim, train_loader, writer)


if __name__ == "__main__":
    main()
