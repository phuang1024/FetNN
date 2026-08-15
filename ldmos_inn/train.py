"""cINN training and validation.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from constants import *
from data import LdmosDegrData, MosDataset
from model import make_model

# Globals for progress tracking.
epoch = 0
global_step = 0


def nll_loss(z, jac):
    """Negative log likelihood loss.
    Want |z| to be low and jac to be high.

    z: (B, D) latent variable.
    jac: (B,) jacobian magnitude.
    """
    z_mag = torch.mean(z**2) / 2
    jac = torch.mean(jac) / z.shape[1]
    nll = z_mag - jac
    return z_mag, jac, nll


def train_epoch(model, optim, train_loader, writer):
    global global_step
    model.train()
    for x, y in (pbar := tqdm(train_loader)):
        z, jac = model(x, [y])
        z_mag, jac, nll = nll_loss(z, jac)

        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optim.step()
        optim.zero_grad()

        writer.add_scalar("train/epoch", epoch, global_step)
        writer.add_scalar("train/z_mag", z_mag.item(), global_step)
        writer.add_scalar("train/jac", jac.item(), global_step)
        writer.add_scalar("train/nll", nll.item(), global_step)
        global_step += 1
    pbar.close()


@torch.no_grad()
def val_epoch(model, val_loader, dataset, writer):
    """
    dataset (MosDataset) is used for the unnormalize method.
    """
    model.eval()
    total_z_loss = 0
    total_x_loss = 0
    for x, y in (pbar := tqdm(val_loader)):
        # Forward NLL loss for Z.
        z, jac = model(x, [y])
        _, _, loss = nll_loss(z, jac)
        total_z_loss += loss.item()

        # Reverse direction MSE loss for X.
        zeros_z = torch.zeros_like(x, device=DEVICE)
        pred_x = model(zeros_z, [y], rev=True, jac=False)[0]
        total_x_loss += torch.nn.functional.mse_loss(pred_x, x)
    pbar.close()

    writer.add_scalar("val/nll", total_z_loss / len(val_loader), global_step)
    writer.add_scalar("val/x_mse_loss", total_x_loss / len(val_loader), global_step)

    # Save some unnormalized generated samples (from last iter of val_loader).
    # Concat (pred_x, y). Shape (B, Dx + Dy)
    logits_xy = torch.zeros([y.shape[0], x.shape[1] + y.shape[1]])
    for i in range(pred_x.shape[0]):
        logits_xy[i, :x.shape[1]] = pred_x[i]
        logits_xy[i, x.shape[1]:] = y[i]
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
    model = make_model(X_DIM, Y_DIM).to(DEVICE)
    print(model)
    print("Num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    writer = SummaryWriter(args.log_dir)

    global epoch
    for epoch in range(EPOCHS):
        train_epoch(model, optim, train_loader, writer)
        val_samples = val_epoch(model, val_loader, dataset, writer)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            torch.save(val_samples, writer.log_dir / f"samples_e{epoch}.pt")
            torch.save(model.state_dict(), writer.log_dir / f"model_e{epoch}.pt")


if __name__ == "__main__":
    main()
