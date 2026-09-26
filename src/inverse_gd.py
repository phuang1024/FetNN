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

    Criteria:
        Each criterion is a function (x, y) -> loss.
            x, y is raw values of recipe and electrical performance.
            loss will be minimized.
        Final loss is weighted sum of all criteria.

    Can manually add criterion.
    Also can call utils for commonly used criteria. E.g. wall function.
    """

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

        self.criteria = []
        self.criteria_weights = []

    def total_loss(self, x, y):
        """Compute weighted sum of criteria.
        Unnormalizes x and y.
        x, y: Normalized values.
        """
        # TODO unnormalize.
        loss = 0
        for criterion, weight in zip(self.criteria, self.criteria_weights):
            loss += criterion(x, y) * weight
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
