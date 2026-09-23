import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from data import load_data


class Designer:
    """Constrained optimization design on FET dataset.
    """

    def __init__(self, data, design_criterion):
        """
        data: (N, 4) Y data.
        design_criterion: Function that returns score of a design.
            Higher is better.
            Args: (4,). Return: Scalar.
        """
        self.knn = NearestNeighbors().fit(data)
        self.design_criterion = design_criterion

    def dist_criterion(self, design):
        """
        design: (4,)
        """
        dist, _ = self.knn.kneighbors(design[None, :])
        dist = np.mean(dist)

        # Hard increase at 0.2.
        score = max(dist - 0.2, 0)
        return score

    def overall_score(self, design):
        """Combine target score and KNN score.
        """
        target_score = self.design_criterion(design)
        dist_score = self.dist_criterion(design)
        return target_score - dist_score

    def design(self, steps=100):
        """Constrained design main function.
        """
        design = np.zeros([4], dtype=float)
        trajectory = []
        scores = []
    
        # Gradient ascent.
        for step in range(steps):
            # Numerically compute gradient.
            curr_score = self.overall_score(design)
            grad = np.zeros_like(design)
            for i in range(len(design)):
                design[i] += 1e-2
                score = self.overall_score(design)
                grad[i] = (score - curr_score) / 1e-2
                design[i] -= 1e-2
    
            # Step.
            design += grad * 0.1
    
            trajectory.append(design.copy())
            scores.append(curr_score)
    
        trajectory = np.array(trajectory)
        scores = np.array(scores)
        return trajectory, scores


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
    plt.colorbar()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    args = parser.parse_args()

    data, _, _ = load_data(args.data)
    vis_knn_dist(data)


if __name__ == "__main__":
    main()
