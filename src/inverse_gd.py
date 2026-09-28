"""Framework for inverse design via GD.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from fet_data import FetDataset
from model import FetDNNModel


class InverseGD:
    """Inverse design implementation.

    Each criterion is a function (x, y, x_raw, y_raw) -> loss.
        x, y: Normalized recipe and electrical performance vectors, respectively.
        x_raw, y_raw: Unnormalized.
        loss: Will be minimized.

    Final loss is weighted sum of all criteria.
    """

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

        self.criteria = []
        self.criteria_weights = []

    def add_criterion(self, criterion, weight):
        self.criteria.append(criterion)
        self.criteria_weights.append(weight)

    def total_loss(self, x, y):
        """Compute weighted sum of criteria.
        Unnormalizes x and y.
        x, y: Normalized values.
        """
        # Unnormalize x and y.
        values = torch.cat((x, y))
        raw_values = self.dataset.unnormalize(values.unsqueeze(0)).squeeze(0)
        raw_x = raw_values[:len(x)]
        raw_y = raw_values[len(x):]

        # Apply all criteria.
        loss = 0
        for criterion, weight in zip(self.criteria, self.criteria_weights):
            loss += criterion(x, y, raw_x, raw_y) * weight
        return loss

    def run_inverse_design(self, steps):
        """
        """
        # TODO starting point customization
        x = torch.zeros([11], requires_grad=True)
        optim = torch.optim.Adam([x], lr=1e-2)

        traj_x = torch.zeros([steps, 11])
        traj_y = torch.zeros([steps, 4])
        losses = torch.zeros([steps])
        for i in range(steps):
            pred_y = self.model(x.unsqueeze(0)).squeeze(0)
            loss = self.total_loss(x, pred_y)
            loss.backward()
            optim.step()
            optim.zero_grad()

            traj_x[i] = x
            traj_y[i] = pred_y
            losses[i] = loss

        return traj_x, traj_y, losses


def plot_trajectory(dataset, traj_x, traj_y, losses, x_index, y_index, x_label, y_label):
    """Plot a few plots regarding design trajectory:

    Scatter plot two features of GT dataset,
    and plot those features from design traj.

    Plot losses over time.

    traj_x, traj_y: (N, Dx), (N, Dy).
    losses: (N,) losses over time.
    x_index, y_index: X and Y axes data feature index in dataset.
    """
    traj_x = traj_x.detach().cpu().numpy()
    traj_y = traj_y.detach().cpu().numpy()
    traj = np.concat((traj_x, traj_y), axis=1)
    losses = losses.detach().cpu().numpy()

    # 2D scatter of chosen features in dataset and trajectory.
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.scatter(dataset.data[:, x_index], dataset.data[:, y_index], color="pink", alpha=0.7)
    plt.plot(traj[:, x_index], traj[:, y_index], color="blue")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Loss over time.
    step_sizes = []
    for i in range(len(traj_x) - 1):
        size = np.linalg.norm(traj_x[i + 1] - traj_x[i])
        step_sizes.append(size)

    plt.subplot(2, 1, 2)
    plt.plot(losses, color="blue", label="Loss")
    plt.ylabel("Loss")

    plt.twinx()
    plt.plot(step_sizes, color="orange", label="Step size")
    plt.ylabel("Step size")

    plt.xlabel("Time")
    plt.legend()

    plt.tight_layout()
    plt.show()
