"""Framework for inverse design via GD.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import *
from dnn_model import FetDNNModel


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
        """
        weight: Either a number or a function (t,) -> weight.
            t is in [0, 1]. I.e. step / total_steps.
        """
        self.criteria.append(criterion)
        self.criteria_weights.append(weight)

    def total_loss(self, x, y, t):
        """Compute weighted sum of criteria.
        x, y: Normalized values.
        t: Progress from 0 to 1.
        """
        # Unnormalize x and y.
        values = torch.cat((x, y))
        raw_values = self.dataset.unnormalize(values.unsqueeze(0)).squeeze(0)
        raw_x = raw_values[:len(x)]
        raw_y = raw_values[len(x):]

        # Apply all criteria.
        loss = 0
        for criterion, weight in zip(self.criteria, self.criteria_weights):
            if callable(weight):
                weight = weight(t)
            loss += criterion(x, y, raw_x, raw_y) * weight
        return loss

    def run_inverse_design(self, steps, lr=1e-2, start=None):
        """
        start: If None, is set to zero.
        """
        if start is None:
            x = torch.zeros([X_DIM], requires_grad=True)
        else:
            x = torch.tensor(start)

        optim = torch.optim.Adam([x], lr=lr)

        traj_x = torch.zeros([steps, X_DIM])
        traj_y = torch.zeros([steps, Y_DIM])
        losses = torch.zeros([steps])
        for i in range(steps):
            pred_y = self.model(x.unsqueeze(0)).squeeze(0)
            t = i / (steps - 1)
            loss = self.total_loss(x, pred_y, t)

            loss.backward()
            optim.step()
            optim.zero_grad()

            traj_x[i] = x
            traj_y[i] = pred_y
            losses[i] = loss

        return traj_x, traj_y, losses
