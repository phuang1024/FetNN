"""DNN forward surrogate.
Run this file to train or test.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from dataset import *


class FetDNNModel(nn.Module):
    """DNN forward surrogate model.
    (B, X_DIM) -> (B, Y_DIM)
    """
    # TODO tune this.
    dim_hidden = 24

    def __init__(self):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(X_DIM, self.dim_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.dim_hidden, Y_DIM),
        )

    def forward(self, x):
        return self.mlp(x)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ADAM_BS = 256
ADAM_LR = 2e-3
WEIGHT_DECAY = 1e-4


@torch.no_grad()
def augment_data(x, y, noise=5e-3):
    # Random noise.
    x = x + torch.randn_like(x) * noise
    y = y + torch.randn_like(y) * noise
    return x, y


def train_epoch_adam(optim):
    """Train with Adam optimizer.
    """
    train_loader = DataLoader(train_data, batch_size=ADAM_BS, shuffle=True)
    model.train()

    total_mse = 0
    for x, y in train_loader:
        x, y = augment_data(x, y)
        pred = model(x)

        loss = F.mse_loss(pred, y)
        total_mse += loss.item()

        loss.backward()
        optim.step()
        optim.zero_grad()

    total_mse /= len(train_loader)
    writer.add_scalar("train/mse", total_mse, global_step)
    writer.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)


def train_epoch_lbfgs(optim):
    """Train with L-BFGS optimizer.
    Uses a single batch (per standard practice).
    """
    def loss_callback():
        optim.zero_grad()
        pred = model(x)
        loss = F.huber_loss(pred, y)

        # Weight decay.
        l2_reg = sum(p.pow(2).sum() for p in model.parameters())
        loss += l2_reg * WEIGHT_DECAY

        loss.backward()
        return loss

    train_loader = DataLoader(train_data, batch_size=len(train_data))
    # No Dropout for LBFGS.
    model.eval()

    x, y = next(iter(train_loader))
    #x, y = augment_data(x, y)
    total_mse = optim.step(loss_callback)

    writer.add_scalar("train/mse", total_mse.item(), global_step)


@torch.no_grad()
def val_epoch():
    """Compute per-feature MSE and MRE on val set.
    """
    train_loader = DataLoader(val_data, batch_size=len(val_data))
    model.eval()

    x, y = next(iter(train_loader))
    pred = model(x)

    # MSE loss.
    loss = F.mse_loss(pred, y)
    total_mse = loss.item()

    # MRE of unnormalized values.
    y_unnorm = dataset.unnormalize(y, (-4, -3, -2, -1))
    pred_unnorm = dataset.unnormalize(pred, (-4, -3, -2, -1))
    mre = torch.abs(pred_unnorm - y_unnorm) / y_unnorm
    total_mre = torch.mean(mre)

    writer.add_scalar("val/mse", total_mse, global_step)
    writer.add_scalar("val/mre", total_mre, global_step)


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)


if __name__ == "__main__":
    # For CPU utilization.
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)

    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()

    # Make datasets.
    dataset = FetDataset(args.data, DEVICE)
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_data, val_data = random_split(dataset, (train_len, val_len))

    # Make model.
    model = FetDNNModel().to(DEVICE)
    init_weights(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print("Num params:", num_params)

    # Make optimizers.
    lbfgs = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
    adam = torch.optim.AdamW(model.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)
    #adam_sched = torch.optim.lr_scheduler.StepLR(adam, 200, 0.6)

    # Logging.
    writer = SummaryWriter(args.log_dir)
    global_step = 0

    for epoch in trange(300, desc="Adam"):
        train_epoch_adam(adam)
        val_epoch()
        #adam_sched.step()
        global_step += 1

    for epoch in trange(100, desc="LBFGS"):
        train_epoch_lbfgs(lbfgs)
        val_epoch()
        global_step += 1

    torch.save(model.state_dict(), args.log_dir / "model_latest.pt")
