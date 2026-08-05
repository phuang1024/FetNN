import argparse
from dataclasses import dataclass

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from constants import *
from data import load_ldmos_degr_data, ListDataset
from model import make_model


@dataclass
class Logging:
    writer: SummaryWriter
    step: int = 0
    epoch: int = 0

    @classmethod
    def from_dir(cls, dir):
        return Logging(SummaryWriter(dir))

    def log(self, prefix, pbar, inc_step=True, **kwargs):
        """Add scalars to both TB and pbar.
        """
        desc = prefix + " "
        for k, v in kwargs.items():
            if type(v) is float:
                desc += f"{k}={v:.3f} "
            else:
                desc += f"{k}={v}"
            self.writer.add_scalar(f"{prefix}/{k}", v, self.step)
        pbar.set_description(desc)

        if inc_step:
            self.step += 1


def nll_loss(z, jac):
    """
    z: (B, D) latent variable.
    jac: (B,) jacobian magnitude.
    """
    z_mse = torch.mean(z**2) / 2
    jac = torch.mean(jac) / z.shape[1]
    nll = z_mse - jac
    return z_mse, jac, nll


def train_epoch(model, optim, train_loader, log: Logging):
    model.train()
    for x, y in (pbar := tqdm(train_loader)):
        z, jac = model(x, [y])
        z_mse, jac, nll = nll_loss(z, jac)

        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optim.step()
        optim.zero_grad()

        log.log("train", pbar,
            z_mse=z_mse.item(),
            log_jac=jac.item(),
            nll=nll.item(),
            lr=optim.param_groups[0]["lr"],
        )


@torch.no_grad()
def val_epoch(model, optim, val_loader, log: Logging):
    model.eval()
    for x, y in (pbar := tqdm(val_loader)):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("log_dir")
    args = parser.parse_args()

    data_labels, x_data, y_data = load_ldmos_degr_data(args.data)
    dataset = ListDataset(x_data, y_data)
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_data, val_data = random_split(dataset, (train_len, val_len))

    loader_args = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
    }
    train_loader = DataLoader(train_data, **loader_args)
    val_loader = DataLoader(val_data, **loader_args)

    model = make_model(X_DIM, Y_DIM).to(DEVICE)
    print(model)
    print("Num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    log = Logging.from_dir(args.log_dir)
    for epoch in range(EPOCHS):
        log.epoch = epoch + 1
        train_epoch(model, optim, train_loader, log)
        val_epoch(model, optim, val_loader, log)


if __name__ == "__main__":
    main()
