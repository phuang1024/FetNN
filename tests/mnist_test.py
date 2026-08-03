"""Test cINN on MNIST.
"""

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms as T
from torchvision.utils import make_grid

import FrEIA.framework as FF
import FrEIA.modules as FM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LAYERS = 12

LR = 2e-3
LR_DECAY = 0.95
BATCH_SIZE = 512
EPOCHS = 50


def one_hot(y):
    """
    y: (B) int 0-9
    return: (B, 10) one hot.
    """
    return nn.functional.one_hot(torch.tensor(y), num_classes=10)


def load_data():
    transform = T.ToTensor()
    train_data = datasets.MNIST(
        root="mnist",
        train=True,
        download=True,
        transform=transform,
        target_transform=one_hot,
    )
    test_data = datasets.MNIST(
        root="mnist",
        train=False,
        download=True,
        transform=transform,
        target_transform=one_hot,
    )

    loader_args = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
    }
    train_loader = DataLoader(train_data, **loader_args)
    test_loader = DataLoader(test_data, **loader_args)
    return train_loader, test_loader


def make_model():
    def fc_subnet(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, dims_out),
        )

    # Create INN.
    model = FF.SequenceINN(784)
    for i in range(NUM_LAYERS):
        model.append(
            FM.AllInOneBlock,
            cond=0,
            cond_shape=[10],
            subnet_constructor=fc_subnet,
    )

    # Initialize weights.
    """
    for param in model.parameters():
        if param.requires_grad:
            param.data = torch.randn_like(param.data) * 1e-4
    """
    return model


def main():
    train_loader, test_loader = load_data()
    model = make_model().to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lr_decay = torch.optim.lr_scheduler.ExponentialLR(optim, LR_DECAY)

    logger = SummaryWriter("mnist_log_12layer")
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        for x, y in (pbar := tqdm(train_loader)):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            x = x.view(x.shape[0], -1)
            # x: (B, 784). y: (B, 10).

            # Forward pass.
            z, jac = model(x, [y])
            # z: (B, 784). jac: (B).
            loss = torch.mean(torch.sum(z**2, dim=1) / 2 - jac)
            loss.backward()

            optim.step()
            optim.zero_grad()

            logger.add_scalar("train/loss", loss.item(), global_step)
            logger.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)
            pbar.set_description(f"Train epoch {epoch}: loss={loss.item():.3f}")
            global_step += 1

        lr_decay.step()

        # Generate tests.
        print("Test epoch", epoch)
        model.eval()
        with torch.no_grad():
            test_bs = 16
            labels = one_hot(torch.randint(0, 9, [test_bs])).to(DEVICE)
            z = torch.randn((test_bs, 784)).to(DEVICE)
            x = model(z, [labels], jac=False, rev=True)[0]
            # x: (B, 784)

            x = x.view(x.shape[0], 28, 28).unsqueeze(1)
            # x: (B, 1, 28, 28)
            vis_img = make_grid(x, 4)
            logger.add_image("test/vis", vis_img, global_step)


if __name__ == "__main__":
    main()
