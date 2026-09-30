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


def plot_2d_trajs(dataset, results, labels):
    """Plot BV-Rsp trajectories.
    Scatter plot BV Rsp dataset.

    results: List of (traj_x, traj_y, losses)
    """
    # 2D scatter of chosen features in dataset and trajectory.
    plt.figure()
    plt.scatter(dataset.data[:, -3], dataset.data[:, -4], color="pink", alpha=0.6)

    # Plot each traj.
    for i in range(len(results)):
        traj_y = results[i][1].detach().cpu().numpy()
        plt.plot(traj_y[:, -3], traj_y[:, -4], label=labels[i])

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
    """Create X barrier criterion based on dataset extrema.
    """
    maxes, _ = torch.max(dataset.data, dim=0)
    mins, _ = torch.min(dataset.data, dim=0)

    def criterion(x, y, raw_x, raw_y):
        loss = 0
        for i in range(len(x)):
            loss -= torch.log(x[i] - mins[i])
            loss -= torch.log(maxes[i] - x[i])
        return loss
    return criterion


def min_rsp(x, y, raw_x, raw_y):
    return y[0]


def bv_floor(x, y, raw_x, raw_y):
    return -torch.log(y[1] - -0.4)


def test_wall_weights(dataset, model):
    """Test different weights for wall_x criterion.
    """
    # Run GD.
    weights = (1e-2, 1e-1, 1, 0)
    results = []
    for weight in weights:
        designer = InverseGD(dataset, model)
        designer.add_criterion(wall_x(dataset), weight)
        designer.add_criterion(min_rsp, 1)

        traj_x, traj_y, losses = designer.run_inverse_design(1000)
        results.append((traj_x, traj_y, losses))

    plot_2d_trajs(dataset, results, weights)


def test_bv_floor(dataset, model):
    """Test BV barrier.
    """
    results = []
    # No BV.
    designer = InverseGD(dataset, model)
    designer.add_criterion(wall_x(dataset), 2e-2)
    designer.add_criterion(min_rsp, 1)
    results.append(designer.run_inverse_design(1000))

    # Yes BV.
    designer.add_criterion(bv_floor, 1e-3)
    results.append(designer.run_inverse_design(1000))

    plot_2d_trajs(dataset, results, ("No BV constraint", "BV > -0.4"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("model")
    args = parser.parse_args()

    dataset = FetDataset(args.data, "cpu")

    model = FetDNNModel()
    model.load_state_dict(torch.load(args.model))

    #test_wall_weights(dataset, model)
    test_bv_floor(dataset, model)


if __name__ == "__main__":
    main()
