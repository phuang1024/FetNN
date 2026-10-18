"""Plot various things regarding NSF model.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import FetDataset
from nsf_model import create_rq_nsf


@torch.no_grad()
def plot_2d_nll(dataset, model, grid_size=64, bv_min=-2, bv_max=2, rsp_min=-2, rsp_max=2):
    """Scatter BV-Rsp data.
    Plot model's NLL across the 2D space, with Vth=Rout=0

    grid_size: Query model across NxN grid.
    """
    # (Y, X, (rsp, bv))
    inputs = torch.zeros((grid_size, grid_size, 2))
    for x_ind in range(grid_size):
        bv = np.interp(x_ind, (0, grid_size), (bv_min, bv_max))
        for y_ind in range(grid_size):
            rsp = np.interp(y_ind, (0, grid_size), (rsp_min, rsp_max))
            inputs[y_ind, x_ind, 0] = rsp
            inputs[y_ind, x_ind, 1] = bv

    # Convert to (B, (rsp, bv, ...))
    inputs = inputs.reshape(-1, 2)
    # For remaining columns.
    inputs = torch.cat((
        #torch.zeros((inputs.shape[0], 11)),
        inputs,
        torch.zeros((inputs.shape[0], 2)),
    ), dim=1)

    # Query model.
    log_prob = model.log_prob(inputs)
    log_prob = log_prob.view(grid_size, grid_size)
    log_prob = log_prob.cpu().numpy()

    plt.figure()
    plt.scatter(dataset.data[:, -3], dataset.data[:, -4])

    plt.imshow(log_prob, extent=(bv_min, bv_max, rsp_min, rsp_max))

    plt.xlabel("BV")
    plt.ylabel("Rsp")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("model")
    args = parser.parse_args()

    dataset = FetDataset(args.dataset, "cpu")

    model = create_rq_nsf()
    model.load_state_dict(torch.load(args.model))

    plot_2d_nll(dataset, model)


if __name__ == "__main__":
    main()
