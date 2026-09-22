import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def load_data(path):
    """Load data from CSV, and normalize.
    """
    data = []
    with open(path) as fp:
        reader = csv.reader(fp)
        for i, line in enumerate(reader):
            # First line is labels.
            if i == 0:
                labels = line
            else:
                data.append(list(map(float, line)))
    data = torch.tensor(data, dtype=torch.float)

    # Log and normalize.
    means = []
    stds = []
    for i in range(data.shape[1]):
        # Log this feature.
        if False:
            data[:, i] = torch.log1p(data[:, i])

        # Compute Z score.
        mean = torch.mean(data[:, i]).item()
        std = torch.std(data[:, i]).item() + 1e-3
        data[:, i] = (data[:, i] - mean) / std

        means.append(mean)
        stds.append(std)

    return data, means, stds


def vis_knn_dist(data):
    """Visualize KNN distance over BV-Rsp space.
    """
    # Take Rsp, BV cols.
    data = data[:, -4:-2]
    bvs = data[:, 1]
    rsps = data[:, 0]
    plt.figure()
    plt.scatter(bvs, rsps)

    # Image of coords from -3 to 3.
    coords = np.zeros((50, 50, 2), dtype=float)
    for x_index in range(50):
        for y_index in range(50):
            x = np.interp(x_index, [0, 49], [-3, 3])
            y = np.interp(y_index, [0, 49], [3, -3])
            coords[y_index][x_index] = y, x

    knn = NearestNeighbors().fit(data)
    dists, _ = knn.kneighbors(coords.reshape(-1, 2))
    dists = np.mean(dists, axis=1)
    dists = dists.reshape(50, 50)
    plt.figure()
    plt.imshow(dists)
    plt.show()


def constrained_design(data):
    """Constrained design main function.
    """
    # Take Y cols.
    data = data[:, -4:]
    knn = NearestNeighbors().fit(data)

    def design_target_score(y):
        """How good given design is. Higher is better.
        y: (4,) design Y values.
        """
        vth_score = -1 * (y[3] - 0.2) ** 2
        fom_score = y[1] - y[0]  # TODO
        return vth_score + fom_score * 0.1

    def overall_score(y):
        """Target score and KNN score.
        """
        target_score = design_target_score(y)
        dist, _ = knn.kneighbors(y[None, :])
        dist = np.mean(dist)
        return target_score - dist * 0.1

    design = np.zeros([4], dtype=float)
    trajectory = []
    scores = []

    # Gradient ascent.
    for step in range(200):
        # Numerically compute gradient.
        curr_score = overall_score(design)
        grad = np.zeros_like(design)
        for i in range(len(design)):
            design[i] += 1e-2
            score = overall_score(design)
            grad[i] = (score - curr_score) / 1e-2
            design[i] -= 1e-2

        # Step.
        design += grad * 0.1

        trajectory.append(design.copy())
        scores.append(curr_score)

    trajectory = np.array(trajectory)
    scores = np.array(scores)
    return trajectory, scores


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
    plt.scatter(bvs, rsps, c="pink", alpha=0.3)

    plt.plot(trajectory[:, 1], trajectory[:, 0])
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    args = parser.parse_args()

    data, _, _ = load_data(args.data)

    trajectory, scores = constrained_design(data)
    plot_design(data, trajectory, scores)


if __name__ == "__main__":
    main()
