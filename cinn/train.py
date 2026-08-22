"""cINN training and validation.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from tqdm import trange

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from constants import *
from data import LdmosDegrData, MosDataset
from model import make_model, init_weights

# Globals for progress tracking.
epoch = 0
global_step = 0


def bidir_loss(model, x, y):
    """Compute fwd and bwd loss for a single data sample.

    Forward:
        z = model(x | y)
        NLL(z, N(0, 1))

    Backward:
        z ~ N(0, 1)
        x_hat = model^-1(z | y)
        MSE(x_hat, x)

    x: (B, Dx) GT recipe.
    y: (B, Dy) condition.
    """
    # Forward.
    # fwd_z: (B, D) latent variable.
    # fwd_jac: (B,) jac magnitude.
    fwd_z, fwd_jac = model(x, [y])
    # NLL loss.
    fwd_z_mag = torch.mean(fwd_z ** 2)
    fwd_jac = torch.mean(fwd_jac) / fwd_z.shape[1]
    fwd_loss = fwd_z_mag - fwd_jac

    # Backward.
    bwd_z = BWD_Z_MAG * torch.randn_like(x)
    bwd_x, bwd_jac = model(bwd_z, [y], rev=True)
    bwd_loss = torch.nn.functional.mse_loss(bwd_x, x)

    return (
        fwd_z_mag, fwd_jac, fwd_loss,
        bwd_x, bwd_loss,
    )


def train_epoch(model, optim, train_loader, writer):
    global global_step
    model.train()
    for x, y in train_loader:
        fwd_z_mag, fwd_jac, fwd_loss, _, bwd_loss = bidir_loss(model, x, y)

        # Train step.
        fwd_loss.backward()
        bwd_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optim.step()
        optim.zero_grad()

        # Logging.
        writer.add_scalar("train/epoch", epoch, global_step)
        writer.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)

        writer.add_scalar("train/fwd_z_mag", fwd_z_mag.item(), global_step)
        writer.add_scalar("train/fwd_jac", fwd_jac.item(), global_step)
        writer.add_scalar("train/fwd_loss", fwd_loss.item(), global_step)
        writer.add_scalar("train/bwd_loss", bwd_loss.item(), global_step)
        global_step += 1


@torch.no_grad()
def val_epoch(model, val_loader, dataset, writer):
    """
    dataset (MosDataset) is used for the unnormalize method.
    """
    model.eval()
    total_fwd_loss = 0
    total_bwd_loss = 0
    for x, y in val_loader:
        _, _, fwd_loss, bwd_x, bwd_loss = bidir_loss(model, x, y)
        total_fwd_loss += fwd_loss.item()
        total_bwd_loss += bwd_loss.item()

    writer.add_scalar("val/fwd_loss", total_fwd_loss / len(val_loader), global_step)
    writer.add_scalar("val/bwd_loss", total_bwd_loss / len(val_loader), global_step)

    # Save some unnormalized generated samples (from last iter of val_loader).
    # Concat (bwd_x, y). Shape (B, Dx + Dy)
    logits_xy = torch.zeros([y.shape[0], x.shape[1] + y.shape[1]])
    for i in range(x.shape[0]):
        logits_xy[i, :x.shape[1]] = bwd_x[i]
        logits_xy[i, x.shape[1]:] = y[i]

    # Unnorm and save.
    unnorm_xy = dataset.unnormalize(logits_xy)
    writer.add_tensor("val/samples", unnorm_xy, global_step=global_step)
    return unnorm_xy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()

    # Make datasets.
    dataset = LdmosDegrData(args.data)
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_data, val_data = random_split(dataset, (train_len, val_len))

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Make model and stuff.
    model = make_model(dataset.x_size, dataset.data.shape[1] - dataset.x_size).to(DEVICE)
    init_weights(model, INIT_WEIGHT)

    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.7)

    writer = SummaryWriter(args.log_dir)

    with open(args.log_dir / "model.txt", "w") as f:
        print(model)
        print(model, file=f)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Num params:", num_params)
        print("Num params:", num_params, file=f)

    global epoch
    for epoch in trange(EPOCHS):
        train_epoch(model, optim, train_loader, writer)
        val_samples = val_epoch(model, val_loader, dataset, writer)
        lr_scheduler.step()

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            torch.save(val_samples, args.log_dir / f"samples_e{epoch}.pt")
            torch.save(model.state_dict(), args.log_dir / f"model_e{epoch}.pt")


if __name__ == "__main__":
    main()
