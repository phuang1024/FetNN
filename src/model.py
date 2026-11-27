"""DNN forward surrogate.
Run this file to train or test.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from dataset import *


class FetDNNModel(nn.Module):
    """DNN forward surrogate model.
    (B, X_DIM) -> (B, Y_DIM)
    """
    # TODO tune this.
    dim_hidden = 32

    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(X_DIM, self.dim_hidden),
            nn.SiLU(),
            #nn.Dropout(0.1),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.SiLU(),
            #nn.Dropout(0.1),
            nn.Linear(self.dim_hidden, Y_DIM),
        )

    def forward(self, x):
        return self.mlp(x)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LBFGS = True
EPOCHS = 1000


@torch.no_grad()
def augment_data(x, y, noise=1e-3):
    # Random noise.
    x = x + torch.randn_like(x) * noise
    y = y + torch.randn_like(y) * noise
    return x, y


def train_epoch_adam(dataset, train_loader, model, optim, writer, epoch):
    model.train()

    total_mse = 0
    for x, y in train_loader:
        #x, y = augment_data(x, y)
        pred = model(x)

        loss = mse_loss(pred, y)
        total_mse += loss.item()

        loss.backward()
        optim.step()
        optim.zero_grad()

    total_mse /= len(train_loader)
    writer.add_scalar("train/mse", total_mse, epoch)
    writer.add_scalar("train/lr", optim.param_groups[0]["lr"], epoch)


def train_epoch_lbfgs(dataset, train_loader, model, optim, writer, epoch):
    def loss_callback():
        # Weight decay.
        optim.zero_grad()
        pred = model(x)
        loss = mse_loss(pred, y)

        """
        l2_reg = sum(p.pow(2).sum() for p in model.parameters())
        loss += l2_reg * 1e-4
        """

        loss.backward()
        return loss

    model.train()
    x, y = next(iter(train_loader))
    #x, y = augment_data(x, y)
    total_mse = optim.step(loss_callback)

    writer.add_scalar("train/mse", total_mse.item(), epoch)


@torch.no_grad()
def val_epoch(dataset, val_loader, model, writer, epoch):
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
    writer.add_scalar("val/mse", total_mse, epoch)
    writer.add_scalar("val/mre", total_mre, epoch)

    # TODO generate samples


def train():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)

    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()

    # Make datasets.
    dataset = FetDataset(args.data, DEVICE)
    if LBFGS:
        train_loader, val_loader = split_train_val(dataset, 0.8, len(dataset))
    else:
        train_loader, val_loader = split_train_val(dataset, 0.8, 256)

    # Make model.
    model = FetDNNModel().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print("Num params:", num_params)

    if LBFGS:
        optim = torch.optim.LBFGS(model.parameters())
    else:
        optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 250, 0.5)

    writer = SummaryWriter(args.log_dir)

    for epoch in trange(EPOCHS):
        if LBFGS:
            train_epoch_lbfgs(dataset, train_loader, model, optim, writer, epoch)
        else:
            train_epoch_adam(dataset, train_loader, model, optim, writer, epoch)
            lr_scheduler.step()

        val_epoch(dataset, val_loader, model, writer, epoch)

    torch.save(model.state_dict(), args.log_dir / "model_latest.pt")


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", default="../data/new_data_5_filtered.csv")
    parser.add_argument("model", default="runs/32mlp/model_e2999.pt")
    args = parser.parse_args()

    dataset = FetDataset(args.dataset, "cpu")

    model = FetDNNModel()
    model.load_state_dict(torch.load(args.model))

    x = torch.load("x.pt").unsqueeze(0)
    y = model(x)
    print(y)


if __name__ == "__main__":
    train()
