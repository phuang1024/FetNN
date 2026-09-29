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


def plot_2d_trajs(dataset, trajectories, labels):
    """Plot BV-Rsp trajectories.
    Scatter plot BV Rsp dataset.

    trajs: (T, N, 2)
        Multiple GD runs (T),
        spanning N steps,
        (BV, Rsp) columns.
    """
    # 2D scatter of chosen features in dataset and trajectory.
    plt.figure()
    plt.scatter(dataset.data[:, -3], dataset.data[:, -4], color="pink", alpha=0.6)

    # Plot each traj.
    for i in range(len(trajectories)):
        traj = trajectories[i].detach().cpu().numpy()
        plt.plot(traj[:, 0], traj[:, 1], label=labels[i])

    plt.xlabel("BV")
    plt.ylabel("Rsp")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_losses(traj_x, losses):
    """
    Plot losses over time.

    losses: (N,) losses over time.
    """
    # Loss over time.
    plt.figure()
    plt.plot(losses, color="blue", label="Loss")
    plt.ylabel("Loss")

    # Step sizes.
    step_sizes = []
    for i in range(len(traj_x) - 1):
        size = np.linalg.norm(traj_x[i + 1] - traj_x[i])
        step_sizes.append(size)

    plt.twinx()
    plt.plot(step_sizes, color="orange", label="Step size")
    plt.ylabel("Step size")

    plt.xlabel("Time")
    plt.legend()

    plt.tight_layout()
    plt.show()


def wall_x(dataset):
    maxes, _ = torch.max(dataset.data, dim=0)
    mins, _ = torch.min(dataset.data, dim=0)

    # TODO hack to skip degenerate features.
    #skip = (maxes - mins) < 1e-3

    def criterion(x, y, raw_x, raw_y):
        loss = 0
        for i in range(len(x)):
            #if not skip[i]:
                loss -= torch.log(x[i] - mins[i])
                loss -= torch.log(maxes[i] - x[i])
        return loss
    return criterion


def fom_crit(x, y, raw_x, raw_y):
    return y[0]
    #return -1 * max(raw_y[-3], 0) / max(raw_y[-4], 0)


def test_wall_weights(dataset, model):
    """Test different weights for wall_x criterion.
    """
    # Run GD.
    weights = (1e-2, 1e-1, 1, 0)
    results = []
    for weight in weights:
        designer = InverseGD(dataset, model)
        designer.add_criterion(wall_x(dataset), weight)
        designer.add_criterion(fom_crit, 1)

        traj_x, traj_y, losses = designer.run_inverse_design(1000)
        results.append((traj_x, traj_y, losses))

    # Prepare data for plotting.
    plot_trajs = []
    for _, traj_y, _ in results:
        plot_trajs.append(traj_y[:, (-3, -4)])

    plot_2d_trajs(dataset, plot_trajs, weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("model")
    args = parser.parse_args()

    dataset = FetDataset(args.data, "cpu")

    model = FetDNNModel()
    model.load_state_dict(torch.load(args.model))

    test_wall_weights(dataset, model)


if __name__ == "__main__":
    main()
