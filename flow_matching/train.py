"""Train flow matching.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchdiffeq import odeint
from tqdm import trange

from data import LdmosDegrData, split_train_val
from model import FlowFetModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLOW_SIGMA = 0

EPOCHS = 400
BATCH_SIZE = 64
LR = 7e-3

epoch = 0
global_step = 0


def sample_velocity_loss(flow_matcher, model, x, y):
    """Sample random latent starting vector Z
    and compute model velocity loss.
    x: (B, Dx) GT feature vector.
    y: (B, Dy) GT condition vector.
    """
    # t=0 latent vector from Gaussian distr. Flow path should go z0 to x.
    z0 = torch.randn_like(x)
    # t: [0, 1]. xt: x at time. ut: GT velocity at time.
    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(z0, x)

    pred_ut = model(xt, y, t.unsqueeze(1))
    loss = torch.nn.functional.mse_loss(pred_ut, ut)
    return loss


def generate_samples(model, z0, y):
    """Generate X samples given z0 (initial) and y (condition).
    Uses trained model and ODE solver.
    z0: (B, Dx)
    y: (B, Dy)
    return: (B, Dx)
    """
    def vel_func(t, xt):
        # Expand t to (B, 1)
        t = t.repeat(xt.shape[0]).unsqueeze(1)
        # Use external y value.
        return model(xt, y, t)

    t_span = torch.tensor([0, 1], dtype=z0.dtype, device=z0.device)
    trajectory = odeint(vel_func, z0, t_span, method="dopri5")
    pred_x = trajectory[-1]
    return pred_x


def train(flow_matcher, model, optim, train_loader, writer):
    global global_step
    model.train()
    # Minimize velocity loss.
    for x, y in train_loader:
        loss = sample_velocity_loss(flow_matcher, model, x, y)
        loss.backward()
        optim.step()
        optim.zero_grad()

        writer.add_scalar("train/vel_loss", loss.item(), global_step)
        writer.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)
        global_step += 1

    # Test X sample and X loss on train dataset.
    model.eval()
    z0 = torch.randn_like(x)
    pred_x = generate_samples(model, z0, y)

    x_loss = torch.nn.functional.mse_loss(pred_x, x)
    writer.add_scalar("train/x_loss", x_loss.item(), global_step)
    return pred_x


@torch.no_grad()
def val(flow_matcher, model, val_loader, writer):
    model.eval()
    # Average velocity loss.
    total_vel_loss = 0
    for x, y in val_loader:
        loss = sample_velocity_loss(flow_matcher, model, x, y)
        total_vel_loss += loss.item()

    total_vel_loss /= len(val_loader)
    writer.add_scalar("val/vel_loss", total_vel_loss, global_step)

    # X loss on val dataset.
    z0 = torch.randn_like(x)
    pred_x = generate_samples(model, z0, y)

    x_loss = torch.nn.functional.mse_loss(pred_x, x)
    writer.add_scalar("val/x_loss", x_loss.item(), global_step)
    return pred_x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()

    # Make datasets.
    dataset = LdmosDegrData(args.data, DEVICE)
    train_loader, val_loader = split_train_val(dataset, 0.8, BATCH_SIZE)

    # Make models.
    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(FLOW_SIGMA)

    model = FlowFetModel().to(DEVICE)
    model.init_weights()

    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 100, 0.7)

    writer = SummaryWriter(args.log_dir)

    with open(args.log_dir / "model.txt", "w") as f:
        print(model)
        print(model, file=f)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Num params:", num_params)
        print("Num params:", num_params, file=f)

    global epoch
    for epoch in trange(EPOCHS):
        train_samples = train(flow_matcher, model, optim, train_loader, writer)
        val_samples = val(flow_matcher, model, val_loader, writer)
        lr_scheduler.step()

        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            # Save results.
            torch.save(model.state_dict(), args.log_dir / f"model_e{epoch}.pt")
            torch.save(train_samples, args.log_dir / f"samples_trainData_e{epoch}.pt")
            torch.save(val_samples, args.log_dir / f"samples_valData_e{epoch}.pt")


if __name__ == "__main__":
    main()
