"""Train DNN.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from model import FetDNNModel
from fet_data import FetDataset, split_train_val

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO tune this.
EPOCHS = 3000
BATCH_SIZE = 64
LR = 1e-3

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
        x, y = augment_data(x, y)
        pred = model(x)

        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        optim.step()
        optim.zero_grad()

        writer.add_scalar("train/loss", loss.item(), global_step)
        writer.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)
        global_step += 1


@torch.no_grad()
def val(dataset, model, val_loader, writer):
    model.eval()

    total_mse = 0
    total_mre = 0
    for x, y in val_loader:
        pred = model(x)

        # MSE loss.
        loss = torch.nn.functional.mse_loss(pred, y)
        total_mse += loss.item()

        # MRE of unnormalized values.
        y_unnorm = dataset.unnormalize(y, (-4, -3, -2, -1))
        pred_unnorm = dataset.unnormalize(pred, (-4, -3, -2, -1))
        mre = torch.abs(pred_unnorm - y_unnorm) / y_unnorm
        total_mre += torch.mean(mre)

    total_mse /= len(val_loader)
    total_mre /= len(val_loader)
    writer.add_scalar("val/mse", total_mse, global_step)
    writer.add_scalar("val/mre", total_mre, global_step)

    # TODO generate samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()

    # Make datasets.
    dataset = FetDataset(args.data, DEVICE)
    train_loader, val_loader = split_train_val(dataset, 0.8, BATCH_SIZE)

    # Make model.
    model = FetDNNModel().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 600, 0.5)

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
