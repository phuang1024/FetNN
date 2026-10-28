"""Test inverse design via GD.
Plot results.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import FetDataset, X_DIM
from inverse_gd import InverseGD
from dnn_model import FetDNNModel

BV_TARGET = 15
VTH_TARGET = 1.5


# Plotting funcs.

def plot_2d_trajs(dataset, results, labels):
    """Plot trajectories and scatter dataset.

    results: List of (traj_x, traj_y, losses)
    """
    def plot_fig(ind_x, ind_y, ax):
        # Plot dataset.
        xs = dataset.unnormalize(dataset.data[:, ind_x], ind_x)
        ys = dataset.unnormalize(dataset.data[:, ind_y], ind_y)
        ax.scatter(xs, ys, color="plum", alpha=0.7)

        # Plot each traj.
        for i in range(len(results)):
            traj_y = results[i][1].detach().cpu().numpy()
            xs = dataset.unnormalize(traj_y[:, ind_x], ind_x)
            ys = dataset.unnormalize(traj_y[:, ind_y], ind_y)

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
                    "mutation_scale": 20,
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


def plot_design_losses(traj_x, losses):
    """Plot losses and step sizes over time.
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
    plt.ylabel("Step size (normalized)")

    plt.xlabel("Time")
    plt.legend()

    plt.tight_layout()
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


@torch.no_grad()
def plot_loss_map(dataset, model, designer, points=10000):
    """Randomly sample X points and compute loss fn at each.
    Then plot over 2D map of variables of choice.
    """
    # Sample and map X values.
    xs = torch.rand([points, X_DIM])
    maxes, _ = torch.max(dataset.data, dim=0)
    mins, _ = torch.min(dataset.data, dim=0)
    for i in range(X_DIM):
        xs[:, i] = xs[:, i] * (maxes[i] - mins[i]) + mins[i]

    # Get corr Y from model.
    ys = model(xs)

    # Or, use dataset points.
    """
    xs = dataset.data[:, :X_DIM]
    ys = dataset.data[:, X_DIM:]
    """

    # Get loss values per point.
    losses = []
    for i in range(len(xs)):
        loss = designer.total_loss(xs[i], ys[i], 0)
        #print(loss)
        losses.append(loss)

    xs = xs.cpu().numpy()
    ys = ys.cpu().numpy()
    losses = np.array(losses)
    #print(xs.shape, losses.dtype)

    # Scatter over BV-Rsp space.
    plt.figure()
    plt.scatter(ys[:, 1], ys[:, 0], c=losses, vmin=-5, vmax=5)

    plt.xlabel("BV")
    plt.ylabel("Rsp")
    plt.title("Recipe negative-log barrier loss across dataset.")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# Example constraint funcs.

def wall_x(dataset):
    """Create X barrier criterion based on dataset extrema.
    """
    # Column indices to skip.
    #skip = (3, 4, 10,)

    def func(x, y, raw_x, raw_y):
        loss = 0
        for i in range(len(x)):
            #if i not in skip:
                loss -= torch.log(x[i] - dataset.mins[i])
                loss -= torch.log(dataset.maxes[i] - x[i])
        return loss
    return func


def y_minimize(ind):
    def func(x, y, raw_x, raw_y):
        return y[ind]
    return func

def y_floor(ind, floor):
    def func(x, y, raw_x, raw_y):
        return -torch.log(y[ind] - floor)
    return func

def y_target(ind, target):
    def func(x, y, raw_x, raw_y):
        return torch.mean((y[ind] - target) ** 2)
    return func


# Design tests.

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

        results.append(designer.run_inverse_design(1000))

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
    bv_logit = (BV_TARGET - dataset.means[-3]) / dataset.stds[-3]
    print(f"BV={BV_TARGET}, logit={bv_logit}")

    vth_logit = (VTH_TARGET - dataset.means[-1]) / dataset.stds[-1]
    print(f"Vth={VTH_TARGET}, logit={vth_logit}")

    # Design.
    result = test_std_design(dataset, model, bv_logit, vth_logit)

    # Save results.
    result = torch.cat((result[0][-1], result[1][-1]))
    result = dataset.unnormalize(result.unsqueeze(0))
    print(result)
    write_recipes(result, "nodes.csv")


if __name__ == "__main__":
    plt.rc("axes", titlesize=16)
    plt.rc("axes", labelsize=14)
    plt.rc("legend", fontsize=15)

    main()
