import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from dnn_model import FetDNNModel
from fet_data import FetDataset


def inverse_gd_design(criterion, model, steps=100):
    x = torch.zeros([11], requires_grad=True)
    optim = torch.optim.Adam([x], lr=1e-2)

    traj_x = torch.zeros([steps, 11])
    traj_y = torch.zeros([steps, 4])
    for i in range(steps):
        pred_y = model(x.unsqueeze(0)).squeeze(0)
        loss = criterion(x, pred_y)
        loss.backward()
        optim.step()

        traj_x[i] = x
        traj_y[i] = pred_y

    return traj_x, traj_y


def make_criterion(y_target=(0, 0, 0, 0)):
    y_target = torch.tensor(y_target)

    def criterion(x, y):
        return torch.linalg.norm(y - y_target)
    return criterion


def plot_results(dataset, traj_x, traj_y):
    traj_x = traj_x.detach().numpy()
    traj_y = traj_y.detach().numpy()

    # Step size.
    step_sizes = []
    for i in range(len(traj_x) - 1):
        size = np.linalg.norm(traj_x[i + 1] - traj_x[i])
        step_sizes.append(size)

    plt.figure()
    plt.plot(step_sizes)
    plt.xlabel("Step")
    plt.ylabel("X step magnitude")

    # BV-Rsp trajectory.
    plt.figure()
    plt.plot(traj_y[:, -3], traj_y[:, -4])
    plt.xlabel("BV")
    plt.ylabel("Rsp")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("model")
    args = parser.parse_args()

    dataset = FetDataset(args.data, "cpu")

    model = FetDNNModel()
    model.load_state_dict(torch.load(args.model))

    criterion = make_criterion()
    traj_x, traj_y = inverse_gd_design(criterion, model)
    plot_results(dataset, traj_x, traj_y)


if __name__ == "__main__":
    main()
