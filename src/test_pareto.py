"""Find BV-Rsp Pareto front.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import FetDataset, X_DIM
from inverse_gd import *
from model import FetDNNModel


def plot_pareto(dataset, pareto_pts):
    # Plot dataset.
    xs = dataset.unnormalize(dataset.data[:, -3][:, None], [-3])
    ys = dataset.unnormalize(dataset.data[:, -4][:, None], [-4])
    plt.scatter(xs, ys, color="plum", alpha=0.8, label="TCAD dataset")

    # Plot pareto pts.
    pareto_pts = np.array(pareto_pts)
    pareto_xs = dataset.unnormalize(pareto_pts[:, 0][:, None], [-3])
    pareto_ys = dataset.unnormalize(pareto_pts[:, 1][:, None], [-4])
    plt.scatter(pareto_xs, pareto_ys, label="Pareto front points")

    plt.xlabel("BV (Volts)")
    plt.ylabel("$R_{sp}$ (Ohms)")
    plt.title("BV-Rsp Pareto front")
    plt.legend()
    plt.tight_layout()
    plt.show()


def weight_sweep(dataset, model):
    """Sweep BV and Rsp relative weight.
    """
    # Weight of Rsp criterion.
    weights = np.linspace(0, 1, 50)

    results = []
    for weight in weights:
        designer = InverseGD(dataset, model)
        designer.add_criterion(wall_x(dataset), 2e-3)
        # Minimize Rsp.
        designer.add_criterion(y_minimize(0), weight)
        # Maximize BV.
        designer.add_criterion(y_maximize(1), 1 - weight)

        res = designer.run_inverse_design(300)
        if res is not None:
            results.append(res)

    return results


def bv_sweep(dataset, model):
    """Sweep BV target.
    """
    bvs = np.linspace(10, 35, 50)

    results = []
    for bv in bvs:
        bv_logit = dataset.normalize([[bv]], [-3])

        designer = InverseGD(dataset, model)
        designer.add_criterion(wall_x(dataset), 2e-3)
        # Minimize Rsp.
        designer.add_criterion(y_minimize(0), 1)
        # BV equality.
        designer.add_criterion(y_target(1, bv_logit), 1)

        res = designer.run_inverse_design(300)
        if res is not None:
            results.append(res)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", default="../data/new_data_5_filtered.csv")
    parser.add_argument("model", default="runs/32mlp/model_e2999.pt")
    args = parser.parse_args()

    dataset = FetDataset(args.data, "cpu")

    model = FetDNNModel()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # Design trajs from each sweep point.
    results = bv_sweep(dataset, model)
    # List of (BV, Rsp) final point in each trajectory.
    pareto_pts = []
    for res in results:
        pareto_pts.append((res[1][-1][1].item(), res[1][-1][0].item()))

    plot_pareto(dataset, pareto_pts)


if __name__ == "__main__":
    plt.rc("axes", titlesize=16)
    plt.rc("axes", labelsize=14)
    plt.rc("legend", fontsize=15)

    main()
