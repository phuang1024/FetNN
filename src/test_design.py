"""Test inverse design via GD.
Plot results.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from fet_data import FetDataset
from inverse_gd import InverseGD
from model import FetDNNModel


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


def wall_x(dataset):
    maxes, _ = torch.max(dataset.data, dim=0)
    mins, _ = torch.min(dataset.data, dim=0)

    def criterion(x, y):
        loss = 0
        for i in range(len(x)):
            loss -= torch.log(x[i] - mins[i])
            loss -= torch.log(maxes[i] - x[i])
        return loss
    return criterion


def fom_crit(x, y):
    return y[0] - y[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("model")
    args = parser.parse_args()

    dataset = FetDataset(args.data, "cpu")

    model = FetDNNModel()
    model.load_state_dict(torch.load(args.model))

    designer = InverseGD(dataset, model)
    designer.add_criterion(wall_x(dataset), 1e-2)
    designer.add_criterion(fom_crit, 1)

    traj_x, traj_y, losses = designer.run_inverse_design(1000)
    plot_results(dataset, traj_x, traj_y, losses)


if __name__ == "__main__":
    main()
