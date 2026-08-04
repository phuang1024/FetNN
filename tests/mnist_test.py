"""Test cINN on MNIST.
"""

import argparse
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

NUM_LAYERS = 16

#JAC_SCALE = 1#e-2

LR = 5e-4
BATCH_SIZE = 256
EPOCHS = 15

TEST_BS = 16


def one_hot(y):
    """
    y: Tensor int (B) 0-9.
    return: Tensor int (B, 10) one hot.
    """
    return nn.functional.one_hot(y, num_classes=10)


def load_data():
    transform = T.ToTensor()
    target_transform = lambda y: one_hot(torch.tensor(y))
    train_data = datasets.MNIST(
        root="mnist",
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )
    test_data = datasets.MNIST(
        root="mnist",
        train=False,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )

    loader_args = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
    }
    train_loader = DataLoader(train_data, **loader_args)
    test_loader = DataLoader(test_data, **loader_args)
    return train_loader, test_loader


class FcSubnet(nn.Module):
    def __init__(self, dims_in, dims_out):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dims_in, 512),
            nn.LeakyReLU(),
            nn.Linear(512, dims_out),
        )

    def forward(self, x):
        """
        cond = x[..., -10:]
        cond = self.cond_mlp(cond)
        data = x[..., :-10]
        data = self.data_mlp(data)
        x = torch.cat((data, cond), dim=-1)
        x = self.final_mlp(x)
        """
        return self.mlp(x)


def make_model():
    # Create INN.
    model = FF.SequenceINN(784)
    for i in range(NUM_LAYERS):
        model.append(
            FM.AllInOneBlock,
            cond=0,
            cond_shape=(10,),
            subnet_constructor=FcSubnet,
            affine_clamping=1,
        )

    # Initialize weights.
    """
    for param in model.parameters():
        if param.requires_grad:
            param.data = torch.randn_like(param.data) * 1e-4
    """
    return model


def generate_samples(model, z_scale):
    with torch.no_grad():
        labels = one_hot(torch.randint(0, 9, [TEST_BS])).to(DEVICE)
        z = torch.randn((TEST_BS, 784)).to(DEVICE) * z_scale

        x = model(z, [labels], jac=False, rev=True)[0]
        # x: (B, 784)
        x = x.view(x.shape[0], 28, 28).unsqueeze(1)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    args = parser.parse_args()

    train_loader, test_loader = load_data()
    model = make_model().to(DEVICE)
    print(model)

    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    lr_decay = torch.optim.lr_scheduler.StepLR(optim, 3, 0.5)

    logger = SummaryWriter(args.log_dir)
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
            z_mse = torch.mean(z**2) / 2
            jac = torch.mean(jac) / x.shape[1]

            loss = z_mse - jac
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optim.step()
            optim.zero_grad()

            # Logging.
            logger.add_scalar("train/z_mse", z_mse, global_step)
            logger.add_scalar("train/log_jac", jac, global_step)
            logger.add_scalar("train/loss", loss.item(), global_step)
            logger.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)
            pbar.set_description(f"Train epoch {epoch}: loss={loss.item():.3f}")
            global_step += 1

        lr_decay.step()
        torch.save(model.state_dict(), "model.pt")

        # Generate test samples.
        print("Test epoch", epoch)
        model.eval()

        x = generate_samples(model, 1)
        logger.add_image("test/vis_z1", make_grid(x, 4), global_step)

        x = generate_samples(model, 0)
        logger.add_image("test/vis_z0", make_grid(x, 4), global_step)


if __name__ == "__main__":
    main()
