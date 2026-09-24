"""Train flow matching.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from dnn_model import FetDNNModel
from fet_data import FetDataset, split_train_val

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 500
BATCH_SIZE = 64
LR = 1e-3

epoch = 0
global_step = 0


def forward_dataset(model, loader):
    for x, y in loader:
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        yield loss


def train(model, optim, train_loader, writer):
    global global_step
    model.train()

    for loss in forward_dataset(model, train_loader):
        loss.backward()
        optim.step()
        optim.zero_grad()

        writer.add_scalar("train/loss", loss.item(), global_step)
        writer.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)
        global_step += 1


@torch.no_grad()
def val(model, val_loader, writer):
    model.eval()

    total_loss = 0
    for loss in forward_dataset(model, val_loader):
        total_loss += loss.item()

    total_loss /= len(val_loader)
    writer.add_scalar("val/loss", total_loss, global_step)

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
    model = FetDNNModel(11, 4).to(DEVICE)
    model.init_weights()

    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 100, 0.7)

    writer = SummaryWriter(args.log_dir)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num params:", num_params)

    global epoch
    for epoch in trange(EPOCHS):
        train(model, optim, train_loader, writer)
        val(model, val_loader, writer)
        lr_scheduler.step()

    torch.save(model.state_dict(), args.log_dir / f"model_e{epoch}.pt")


if __name__ == "__main__":
    main()
