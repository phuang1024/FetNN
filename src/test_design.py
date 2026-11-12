"""Test inverse design via GD.
Plot results.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import FetDataset, X_DIM
from inverse_gd import *
from model import FetDNNModel

BV_TARGET = 15
VTH_TARGET = 1.5


def plot_2d_trajs(dataset, results, labels):
    """Plot trajectories and scatter dataset.

    results: List of (traj_x, traj_y, losses)
    """
    def plot_fig(ind_x, ind_y, ax):
        # Plot dataset.
        xs = dataset.unnormalize(dataset.data[:, ind_x][:, None], [ind_x])
        ys = dataset.unnormalize(dataset.data[:, ind_y][:, None], [ind_y])
        ax.scatter(xs, ys, color="plum", alpha=0.7, label="TCAD dataset")

        # Plot each traj.
        for i in range(len(results)):
            traj_y = results[i][1].detach().cpu().numpy()
            xs = dataset.unnormalize(traj_y[:, ind_x][:, None], [ind_x])
            ys = dataset.unnormalize(traj_y[:, ind_y][:, None], [ind_y])

            linewidth = 3 if i == 0 else 2
            ax.plot(xs, ys, label=labels[i], linewidth=linewidth)

            # Plot direction arrows.
            for j in range(10, 260, 100):
                x1 = xs[j]
                x2 = xs[j + 1]
                y1 = ys[j]
                y2 = ys[j + 1]
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={
                    "arrowstyle": "-|>",
                    "color": f"C{i}",
                    "lw": 0,
                    "mutation_scale": 30,
                    "alpha": 0.8,
                })

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    # BV-Rsp.
    plot_fig(-3, -4, ax1)

    x_min, x_max = ax1.get_xlim()
    ax1.axvspan(x_min, BV_TARGET, color="red", alpha=0.2, hatch="/")
    ax1.set_xlim(x_min, x_max)

    ax1.set_xlabel("BV (Volts)")
    ax1.set_ylabel("$R_{sp}$ (Ohms)")
    #ax1.legend()

    # Vth-Rsp
    plot_fig(-1, -4, ax2)

    x_min, x_max = ax2.get_xlim()
    ax2.axvspan(VTH_TARGET - 0.1, VTH_TARGET + 0.1, color="green", alpha=0.2)
    ax2.set_xlim(x_min, x_max)

    ax2.set_xlabel("$V_{th}$ (Volts)")
    ax2.legend()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def write_recipes(recipes, file):
    """Write generated recipes to CSV file.
    """
    with open(file, "w") as fp:
        fp.write("Lsti,tox,nwellD1,nwellD2,nwellD3,dose_energy1,dose_energy2,dose_energy3,aneal_temp,Lextra,high_k,Rsp,BV,Rout,vth\n")
        for x in recipes:
            x = x[:11].detach().cpu().tolist()
            x = map(str, x)
            fp.write(",".join(x))
            fp.write("\n")


def test_std_design(dataset, model, bv_floor, vth_target):
    """Criteria: wall X, minimize Rsp, BV > x, Vth = x
    """
    # Test different Wall X weights.
    weights = (1e-2, 0)
    labels = ("With 2-sided X barrier", "No barrier")
    # weights = (1e-2,)
    # labels = ("Design trajectory",)

    results = []
    for weight in weights:
        designer = InverseGD(dataset, model)
        designer.add_criterion(wall_x(dataset), weight)
        # Minimize Rsp.
        designer.add_criterion(y_minimize(0), 1)
        # BV > x.
        designer.add_criterion(y_floor(1, bv_floor), 1e-2)
        # Vth = x.
        designer.add_criterion(y_target(3, vth_target), 1)

        results.append(designer.run_inverse_design(300))

    # Print recipe normalized btwn min/max.
    for i in range(len(results)):
        norm_x = results[i][0][-1]
        unnormed_x = dataset.unnormalize(norm_x.unsqueeze(0), range(X_DIM))

        norm_x = (norm_x - dataset.mins[:X_DIM]) / (dataset.maxes[:X_DIM] - dataset.mins[:X_DIM])
        norm_x = norm_x[:X_DIM]
        print(f"Label: {labels[i]}")
        print(f"    Min-max normed recipe: {norm_x}")
        print(f"    Unnormed: {unnormed_x}")

    plot_2d_trajs(dataset, results, labels)
    return results[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("model")
    args = parser.parse_args()

    dataset = FetDataset(args.data, "cpu")

    model = FetDNNModel()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # Extract target BV and Vth normalized values.
    bv_logit, vth_logit = dataset.normalize([[BV_TARGET, VTH_TARGET]], [-3, -1]).squeeze(0)
    print(f"BV={BV_TARGET}, logit={bv_logit}")
    print(f"Vth={VTH_TARGET}, logit={vth_logit}")

    # Design.
    result = test_std_design(dataset, model, bv_logit, vth_logit)

    # Save results.
    result = torch.cat((result[0][-1], result[1][-1]))
    result = dataset.unnormalize(result.unsqueeze(0), range(len(result)))
    print(result)
    write_recipes(result, "nodes.csv")


if __name__ == "__main__":
    plt.rc("axes", titlesize=16)
    plt.rc("axes", labelsize=14)
    plt.rc("legend", fontsize=15)

    main()
