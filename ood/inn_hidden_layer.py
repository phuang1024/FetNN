"""Out of distribution detection by analyzing distr of INN hidden activations.
"""

import matplotlib.pyplot as plt
import torch

from weiman_code import (
    build_inn,
    y_train, y_val,
    ndim_tot, ndim_x, ndim_y, ndim_z,
)

MODEL = build_inn(1)
model_path = "../../WeimanCode/inn2/inn_unified.pt"
MODEL.load_state_dict(torch.load(model_path))
MODEL.eval()
print("Load INN model from", model_path)

# Shift BV and Rsp by const.
y_fake_added = y_val.clone()
y_fake_added[:, 0] -= 2
y_fake_added[:, 1] += 2


def inverse_hidden_stats(y_data):
    """Capture mean and std of each hidden layer activations, per data sample.
    y_data: (N, 4) dataset to use.
    return: means, stds, layer_names.
        means, stds: (M, N). means[m][n] is mean of activations of m'th layer using y_data[n]
    """
    # Prepare input data. Concat (z, zero, y)
    zs = torch.randn(y_data.shape[0], ndim_z)
    zeros = torch.zeros(y_data.shape[0], ndim_tot - ndim_y - ndim_z)
    in_data = torch.cat([zs, zeros, y_data], dim=1)

    # Run model.
    out_data = MODEL(in_data, rev=True, intermediate_outputs=True)[0]

    means = []
    stds = []
    layer_names = []
    for layer, logits in out_data.items():
        # Only want coupling layers.
        if layer[0].name[0] != "c":
            continue

        # logits: (N, ndim_tot) activations for current layer.
        means.append(torch.mean(logits, dim=1))
        stds.append(torch.std(logits, dim=1))
        layer_names.append(layer[0].name)

    means = torch.stack(means)
    stds = torch.stack(stds)
    print("Layer names:", layer_names)
    return means, stds, layer_names


def ood_z_score(y_base, y_test):
    """Compute z score of y_test mean/std metric wrt y_base metric distrs.
    y_base: (Nbase, 4).
    y_test: (N, 4).
    return: mean_zs, std_zs.
        Both are (M, N). mean_zs[n][m] is z-score of the mean statistic of the m'th layer,
            passing in y_test[n]; compared to the overall m'th layer distr for y_base.
    """
    base_means, base_stds, _ = inverse_hidden_stats(y_base)
    # Take mean and std of EACH of activation means and stds, across y_base data.
    # These are (M,).
    base_mean_means = torch.mean(base_means, dim=1)
    base_mean_stds = torch.std(base_means, dim=1)
    base_std_means = torch.mean(base_stds, dim=1)
    base_std_stds = torch.std(base_stds, dim=1)

    test_means, test_stds, _ = inverse_hidden_stats(y_test)
    # Compute z scores. (M, N).
    mean_zs = torch.zeros_like(test_means)
    std_zs = torch.zeros_like(test_means)
    # Iterate through layers (M).
    for i in range(len(test_means)):
        mean_zs[i] = (test_means[i] - base_mean_means[i]) / base_mean_stds[i]
        std_zs[i] = (test_stds[i] - base_std_means[i]) / base_std_stds[i]
    return mean_zs, std_zs


def rms_mean(data, dim=None):
    """return sqrt(mean(square(data)))
    """
    return torch.sqrt(torch.mean(data ** 2, dim=dim))


def main_z_score():
    """Print average z scores."""
    def print_score(y_test, name):
        mean_zs, std_zs = ood_z_score(y_train, y_test)
        print(f"{name}: avg mean z: {rms_mean(mean_zs)}, avg std z: {rms_mean(std_zs)}")

    print_score(y_train, "y_train")
    print_score(y_val, "y_val")
    print_score(y_fake_added, "y_fake_added")


def main_2d_sweep():
    """Sweep BV and Rsp and scatter plot OOD metric."""
    X_MAX = 4

    # BV and Rsp go [-X, X].
    samples = torch.linspace(-X_MAX, X_MAX, 50)
    bvs, rsps = torch.meshgrid(samples, samples)
    # Shape (50*50,)
    bvs = bvs.reshape(-1)
    rsps = rsps.reshape(-1)

    # (50*50, 4)
    y_fake_sweep = torch.stack((
        rsps,
        bvs,
        torch.zeros_like(rsps),
        torch.zeros_like(rsps),
    ), dim=1)

    # Get per-layer, per-sample z scores.
    mean_zs, std_zs = ood_z_score(y_train, y_fake_sweep)

    def plot_score_heatmap(scores):
        plt.scatter(bvs, rsps, c=scores, vmin=0, vmax=5)

        plt.xlim(-X_MAX, X_MAX)
        plt.ylim(-X_MAX, X_MAX)

        plt.colorbar()
        plt.xlabel("BV")
        plt.ylabel("Rsp")


    # Plot average score across layers, and overlay train dataset.
    plt.figure()
    # Two plots: One with heatmap, one with heatmap and train data.
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plot_score_heatmap(rms_mean(std_zs, dim=0))

    # Scatter train data on last plot.
    plt.scatter(y_train[:, 1], y_train[:, 0], color="pink", alpha=0.5)

    plt.suptitle("OOD $\sigma$ z-score, and y_train dataset")
    plt.tight_layout()
    plt.show()

    # Plot heatmap per layer.
    plt.figure()
    for i in range(len(std_zs)):
        plt.subplot(4, 4, i + 1)
        plot_score_heatmap(std_zs[i])
        #plt.title()

    plt.suptitle("OOD $\sigma$ z-score per layer")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    #main_z_score()
    main_2d_sweep()
