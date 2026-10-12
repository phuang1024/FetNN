"""Test inverse design via GD.
Plot results.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from fet_data import FetDataset, X_DIM
from inverse_gd import InverseGD
from model import FetDNNModel

BV_FLOOR = -0.3
VTH_TARGET = 0.25


# Plotting funcs.

def plot_2d_trajs(dataset, results, labels):
    """Plot trajectories and scatter dataset.

    results: List of (traj_x, traj_y, losses)
    """
    def plot_fig(ind_x, ind_y):
        # Plot dataset.
        plt.scatter(dataset.data[:, ind_x], dataset.data[:, ind_y], color="pink", alpha=0.6)

        # Plot each traj.
        for i in range(len(results)):
            traj_y = results[i][1].detach().cpu().numpy()
            plt.plot(traj_y[:, ind_x], traj_y[:, ind_y], label=labels[i])

    # BV-Rsp.
    plt.figure()
    plot_fig(-3, -4)
    plt.xlabel("BV")
    plt.ylabel("Rsp")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Vth-Rsp
    plt.figure()
    plot_fig(-1, -4)
    plt.xlabel("Vth")
    plt.ylabel("Rsp")
    plt.legend()
    plt.tight_layout()
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

    # Get loss values per point.
    losses = []
    for i in range(len(xs)):
        losses.append(designer.total_loss(xs[i], ys[i]))

    xs = xs.cpu().numpy()
    ys = ys.cpu().numpy()
    losses = np.array(losses)

    # Scatter over BV-Rsp space.
    plt.figure()
    plt.scatter(ys[:, 1], ys[:, 0], c=losses)
    plt.xlabel("BV")
    plt.ylabel("Rsp")

    plt.tight_layout()
    plt.show()


# Example constraint funcs.

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
    return -torch.log(y[1] - BV_FLOOR)


def target_vth(x, y, raw_x, raw_y):
    return torch.mean((y[3] - VTH_TARGET) ** 2)


def impossible_design(dataset, model):
    """Test infeasible design.
    """
    def target_bv_rsp(x, y, raw_x, raw_y):
        return (y[0] + 0.5) ** 2 + (y[1] - 0.5) ** 2

    results = []
    designer = InverseGD(dataset, model)
    designer.add_criterion(target_bv_rsp, 1)
    results.append(designer.run_inverse_design(1000))

    designer.add_criterion(wall_x(dataset), 2e-3)
    results.append(designer.run_inverse_design(1000))

    plot_2d_trajs(dataset, results, ("Without X barrier", "With X barrier"))


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
    # Print BV value.
    bv_value = dataset.unnormalize(torch.tensor([[BV_FLOOR]]), [-3])
    print(f"BV corr. to logit {BV_FLOOR}:", bv_value)

    results = []
    # No BV.
    designer = InverseGD(dataset, model)
    designer.add_criterion(wall_x(dataset), 2e-2)
    designer.add_criterion(min_rsp, 1)
    results.append(designer.run_inverse_design(1000))

    # Yes BV.
    designer.add_criterion(bv_floor, 1e-2)
    results.append(designer.run_inverse_design(1000))

    plot_2d_trajs(dataset, results, ("No BV constraint", f"BV > {BV_FLOOR}"))
    return results[-1]


def test_target_vth(dataset, model):
    # Print Vth value.
    vth_value = dataset.unnormalize(torch.tensor([[VTH_TARGET]]), [-1])
    print(f"Vth corr. to logit {VTH_TARGET}:", vth_value)

    results = []
    # No Vth.
    designer = InverseGD(dataset, model)
    designer.add_criterion(wall_x(dataset), 2e-2)
    designer.add_criterion(min_rsp, 1)
    designer.add_criterion(bv_floor, 1e-2)
    results.append(designer.run_inverse_design(1000))

    # Yes Vth.
    designer.add_criterion(target_vth, 5)
    results.append(designer.run_inverse_design(1000))

    plot_2d_trajs(dataset, results, ("No Vth constraint", f"Vth = "))
    return results[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("model")
    args = parser.parse_args()

    dataset = FetDataset(args.data, "cpu")

    model = FetDNNModel()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    designer = InverseGD(dataset, model)
    designer.add_criterion(wall_x(dataset), 1)
    plot_loss_map(dataset, model, designer)
    stop

    # Design.
    #impossible_design(dataset, model)
    #test_wall_weights(dataset, model)
    #result = test_bv_floor(dataset, model)
    #result = test_target_vth(dataset, model)

    # Save results.
    result = torch.cat((result[0][-1], result[1][-1]))
    result = dataset.unnormalize(result.unsqueeze(0))
    print(result)
    write_recipes(result, "nodes.csv")


if __name__ == "__main__":
    main()
