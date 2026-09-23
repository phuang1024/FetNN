import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from data import load_data
from designer import Designer


def plot_design(data, trajectory, scores):
    for i in range(len(trajectory)):
        print(f"Step {i}\t  Score {scores[i]}\t  Design {trajectory[i]}")

    # Step abs dists.
    step_dists = []
    for i in range(len(trajectory) - 1):
        dist = np.linalg.norm(trajectory[i + 1] - trajectory[i])
        step_dists.append(dist)

    plt.figure()
    plt.plot(step_dists)
    plt.show()

    # Plot BV-Rsp data and trajectory.
    data = data[:, -4:-2]
    bvs = data[:, 1]
    rsps = data[:, 0]
    plt.figure()
    plt.scatter(bvs, rsps, c="pink", alpha=0.5)

    plt.plot(trajectory[:, 1], trajectory[:, 0])
    plt.show()


def design_criterion(y):
    """How good given design is. Higher is better.
    y: (4,) design Y values.
    """
    vth_score = -1 * (y[3] - 0.2) ** 2
    fom_score = 3 * y[1] - y[0]  # TODO
    return vth_score + fom_score * 0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    args = parser.parse_args()

    data, _, _ = load_data(args.data)

    data = data[:, -4:]
    designer = Designer(data, design_criterion)

    trajectory, scores = designer.design(100)
    plot_design(data, trajectory, scores)


if __name__ == "__main__":
    main()
