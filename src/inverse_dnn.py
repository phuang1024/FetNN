import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import FetDNNModel
from fet_data import FetDataset


def inverse_gd_design(criterion, model, steps=1000):
    x = torch.zeros([11], requires_grad=True)
    optim = torch.optim.Adam([x], lr=1e-2)

    traj_x = torch.zeros([steps, 11])
    traj_y = torch.zeros([steps, 4])
    losses = torch.zeros([steps])
    for i in range(steps):
        pred_y = model(x.unsqueeze(0)).squeeze(0)
        loss = criterion(x, pred_y)
        loss.backward()
        optim.step()
        optim.zero_grad()

        traj_x[i] = x
        traj_y[i] = pred_y
        losses[i] = loss

    return traj_x, traj_y, losses


def make_criterion():
    #y_target = torch.tensor((-0.25, 0, 0, 0))

    def criterion(x, y):
        #return torch.linalg.norm(y - y_target)
        wall_x = torch.mean(-torch.log(x + 1) - torch.log(-x + 1))
        fom = -y[0]
        return -fom + wall_x
    return criterion


def plot_results(dataset, traj_x, traj_y, losses):
    traj_x = traj_x.detach().numpy()
    traj_y = traj_y.detach().numpy()
    losses = losses.detach().numpy()

    # Loss.
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")

    # Step size.
    step_sizes = []
    for i in range(len(traj_x) - 1):
        size = np.linalg.norm(traj_x[i + 1] - traj_x[i])
        step_sizes.append(size)

    plt.figure()
    plt.plot(step_sizes)
    plt.xlabel("Step")
    plt.ylabel("X step magnitude")

    # BV-Rsp trajectory.
    plt.figure()
    plt.scatter(dataset.data[:, -3], dataset.data[:, -4], color="pink", alpha=0.4)
    plt.plot(traj_y[:, -3], traj_y[:, -4])

    plt.xlabel("BV")
    plt.ylabel("Rsp")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    # Some trajectories in X.
    plt.figure()
    plt.plot(traj_x[:, 0], label="Lsti")

    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Value")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("model")
    args = parser.parse_args()

    dataset = FetDataset(args.data, "cpu")

    model = FetDNNModel()
    model.load_state_dict(torch.load(args.model))

    criterion = make_criterion()
    traj_x, traj_y, losses = inverse_gd_design(criterion, model)
    plot_results(dataset, traj_x, traj_y, losses)


if __name__ == "__main__":
    main()
