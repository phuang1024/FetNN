import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from tqdm import tqdm

from data import load_data, one_hot
from model import make_model, init_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 5e-4
EPOCHS = 32

TEST_BS = 16


def nll_loss(z, jac):
    """
    z: (B, 784) latent variable.
    jac: (B,) jacobian magnitude.

    return: mean(mse(z)), mean(jac), nll_loss
    """
    z_mse = torch.mean(z**2) / 2
    jac = torch.mean(jac) / 784
    nll = z_mse - jac
    return z_mse, jac, nll


def generate_samples(model, z_scale):
    """Generate random testing samples.
    Labels (condition) random.
    Latent from a normal dist.
    """
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
    init_weights(model)
    print(model)
    print("Total num params:", sum(p.numel() for p in model.parameters()))
    print("Trainable num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optim = torch.optim.Adam(model.parameters(), LR, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 8, 0.4)

    logger = SummaryWriter(args.log_dir)
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        for x, y in (pbar := tqdm(train_loader)):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            x = x.view(x.shape[0], -1)
            # x: (B, 784). y: (B, 10).

            # Forward.
            z, jac = model(x, [y])
            z_mse, jac, nll = nll_loss(z, jac)

            nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optim.step()
            optim.zero_grad()

            logger.add_scalar("train/z_mse", z_mse, global_step)
            logger.add_scalar("train/log_jac", jac, global_step)
            logger.add_scalar("train/loss", nll.item(), global_step)
            logger.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)
            pbar.set_description(f"Train epoch {epoch}: loss={nll.item():.3f}")
            global_step += 1

        lr_scheduler.step()
        torch.save(model.state_dict(), os.path.join(args.log_dir, "model.pt"))

        # Test.
        print("Test epoch", epoch)
        model.eval()

        x = generate_samples(model, 1)
        logger.add_image("test/vis_z1", make_grid(x, 4), global_step)

        x = generate_samples(model, 0)
        logger.add_image("test/vis_z0", make_grid(x, 4), global_step)


if __name__ == "__main__":
    main()
