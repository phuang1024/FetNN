"""Benchmark DNN+GD against INN."""

import argparse

import numpy as np
import torch

from dataset import FetDataset
from inverse_gd import *
from model import FetDNNModel
from test_design import plot_2d_trajs
from weiman_code import build_inn, inn_inverse

# Typical values for testing.
RSP = 4e-3
BV = 15
ROUT = 1e4
VTH = 1.5


def load_models():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", default="../data/new_data_5_filtered.csv")
    parser.add_argument("model", default="runs/32mlp/model_e2999.pt")
    args = parser.parse_args()

    dataset = FetDataset(args.data, "cpu")

    dnn_model = FetDNNModel()
    dnn_model.load_state_dict(torch.load(args.model))
    dnn_model.eval()

    inn_model = build_inn(1)
    model_path = "../../WeimanCode/inn2/inn_unified.pt"
    inn_model.load_state_dict(torch.load(model_path))
    inn_model.eval()

    return dataset, dnn_model, inn_model


def constrain_x(dataset, x):
    """Modify X so each value is inside it's min and max.
    x: (Dx,)
    """
    for i in range(len(x)):
        x[i] = min(max(x[i], dataset.mins[i] + 1e-3), dataset.maxes[i] - 1e-3)


def designs_std(data):
    """Compute design variability metric.
    data: (N, Dx)
    """
    data = np.array(data)
    cov = np.cov(data, rowvar=False)

    variance = np.trace(cov)
    std = np.sqrt(variance)
    return std


def compare_diversity(dataset, dnn_model, inn_model, logits):
    """Compare recipe diversity between INN and GD.
    """
    # INN.
    inn_designs = inn_inverse(inn_model, logits.unsqueeze(0), 100).squeeze(1)

    # DNN GD.
    designer = InverseGD(dataset, dnn_model)
    designer.add_criterion(wall_x(dataset), 1e-2)
    for i in range(4):
        designer.add_criterion(y_target(i, logits[i]), 1)

    dnn_designs = []
    total_runs = 0
    while len(dnn_designs) < 100:
        total_runs += 1
        start = torch.randn(X_DIM) * 1
        result = designer.run_inverse_design(500, start=start)

        if result is None:
            # Violated one of the ineq constraints.
            continue
        dnn_designs.append(result[0][-1])

    print(f"Total DNN runs: {total_runs}")

    inn_std = designs_std(inn_designs)
    dnn_std = designs_std(dnn_designs)
    print(f"INN std: {inn_std}")
    print(f"DNN std: {dnn_std}")


def inn_seeds_gd(dataset, dnn_model, inn_model, logits):
    """Use INN result to seed GD run.
    """
    inn_design = inn_inverse(inn_model, logits.unsqueeze(0), 1).squeeze(0).squeeze(0)
    constrain_x(dataset, inn_design)

    # DNN GD.
    designer = InverseGD(dataset, dnn_model)
    designer.add_criterion(wall_x(dataset), 1e-2)
    for i in range(4):
        designer.add_criterion(y_target(i, logits[i]), 1)

    results = []
    results.append(designer.run_inverse_design(500))
    results.append(designer.run_inverse_design(500, start=inn_design))
    labels = ("Start at zero", "Start at INN design")
    plot_2d_trajs(dataset, results, labels)


def main():
    dataset, dnn_model, inn_model = load_models()

    # Normalized target Y.
    logits = dataset.normalize(
        [[RSP, BV, ROUT, VTH]],
        [-4, -3, -2, -1],
    ).squeeze(0)

    #compare_diversity(dataset, dnn_model, inn_model, logits)
    inn_seeds_gd(dataset, dnn_model, inn_model, logits)


if __name__ == "__main__":
    main()
