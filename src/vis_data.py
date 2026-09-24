import matplotlib.pyplot as plt

from fet_data import FetDataset


def plot_feature_dists(dataset: FetDataset):
    """Plot hist of each feature.
    After logging and normalizing.
    """
    plt.figure(figsize=(20, 20))

    for i in range(dataset.data.shape[1]):
        plt.subplot(5, 3, i + 1)
        plt.hist(dataset.data[:, i], bins=50)
        plt.title(f"{dataset.labels[i]}: log={dataset.log[i]}, mean={dataset.means[i]:.3f}, std={dataset.stds[i]:.3f}")

    plt.tight_layout()
    plt.show()


def plot_bv_rsp(dataset):
    """Plot BV-Rsp 2D scatter.
    """
    plt.figure()
    plt.scatter(dataset.orig_data[:, -3], dataset.orig_data[:, -4])

    plt.xlabel("BV (V)")
    plt.ylabel("Rsp (Ohm)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    args = parser.parse_args()

    # Print dataset stats.
    dataset = FetDataset(args.data, "cpu")
    print("Dataset length:", len(dataset))
    x, y = dataset[0]
    print("  x:", x.shape, x.dtype, x)
    print("  y:", y.shape, y.dtype, y)

    #plot_feature_dists(dataset)
    plot_bv_rsp(dataset)
