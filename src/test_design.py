"""Test inverse design via GD.
Plot results.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from fet_data import FetDataset
from inverse_gd import InverseGD, plot_trajectory
from model import FetDNNModel


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
    #return y[0]
    return -1 * max(raw_y[-3], 0) / max(raw_y[-4], 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("model")
    args = parser.parse_args()

    dataset = FetDataset(args.data, "cpu")

    model = FetDNNModel()
    model.load_state_dict(torch.load(args.model))

    designer = InverseGD(dataset, model)
    designer.add_criterion(wall_x(dataset), 1e2)
    designer.add_criterion(fom_crit, 1)

    traj_x, traj_y, losses = designer.run_inverse_design(1000)
    plot_trajectory(dataset, traj_x, traj_y, losses, -3, -4, "BV", "Rsp")


if __name__ == "__main__":
    main()
