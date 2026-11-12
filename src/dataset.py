"""Load and process dataset from CSV.
Run this file to vis data.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

X_DIM = 11
Y_DIM = 4
"""Number of X, Y features in dataset."""

"""
# For new_data_4.csv
LOG_FEATURE = (
    False, False,
    True, True, True,
    False, False, False,
    True, False,

    False, False, True, False,
)
"""
# For new_data_5_filtered.csv
LOG_FEATURE = (
    False, False,
    True, True, True,
    False, False, False,
    True, False, False,

    False, False, True, False,
)
"""Whether to log each feature."""


class FetDataset(Dataset):
    raw_data: torch.Tensor
    """(N, D) raw data."""
    data: torch.Tensor
    """(N, D) data after normalization and log."""
    labels: list[str]

    means: list[float]
    stds: list[float]
    """Mean and std after logging."""

    def __init__(self, path, device):
        """Init from CSV file.
        """
        self.device = device

        self.raw_data, self.labels = read_csv(path)
        self.preprocess_data()

    def preprocess_data(self):
        """Log and normalize features.
        """
        self.data = self.raw_data.clone()

        self.means = []
        self.stds = []
        for i in range(self.raw_data.shape[1]):
            # Log this feature.
            if LOG_FEATURE[i]:
                self.data[:, i] = torch.log1p(self.raw_data[:, i])

            # Z score normalization.
            mean = torch.mean(self.data[:, i]).item()
            std = torch.std(self.data[:, i]).item() + 1e-3
            self.data[:, i] = (self.data[:, i] - mean) / std

            self.means.append(mean)
            self.stds.append(std)

        # Get per-dim maxes and mins.
        self.maxes, _ = torch.max(self.data, dim=0)
        self.mins, _ = torch.min(self.data, dim=0)

    def normalize(self, data, col_indices=None):
        """Normalize and log (if applicable) arbitrary data according to
        this dataset's means and stds.
        data: (B, D).
        col_indices:
            None: Use indices 0 to D-1.
            List: data[:, i] is column col_indices[i]
        """
        # Copy data.
        data = torch.tensor(data)

        for i in range(data.shape[1]):
            # Get column index.
            col = i if col_indices is None else col_indices[i]

            if LOG_FEATURE[col]:
                data[:, i] = torch.log1p(data[:, i])
            data[:, i] = (data[:, i] - self.means[col]) / self.stds[col]

        return data

    def unnormalize(self, data, col_indices=None):
        """Counterpart of normalize().
        """
        data = torch.tensor(data)

        for i in range(data.shape[1]):
            # Get column index.
            col = i if col_indices is None else col_indices[i]

            data[:, i] = data[:, i] * self.stds[col] + self.means[col]
            if LOG_FEATURE[col]:
                data[:, i] = torch.expm1(data[:, i])

        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index, :X_DIM]
        y = self.data[index, X_DIM:]
        return x, y


def read_csv(path):
    """Load raw data from CSV.
    return: raw_data, labels
    """
    raw_data = []
    with open(path) as fp:
        reader = csv.reader(fp)
        for i, line in enumerate(reader):
            # First line is labels.
            if i == 0:
                labels = line
            else:
                raw_data.append(list(map(float, line)))

    raw_data = torch.tensor(raw_data, dtype=torch.float)
    return raw_data, labels


def split_train_val(dataset, ratio, batch_size):
    """Helper func to split dataset.
    """
    train_len = int(len(dataset) * ratio)
    val_len = len(dataset) - train_len
    train_data, val_data = random_split(dataset, (train_len, val_len))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def plot_feature_dists(dataset: FetDataset):
    """Plot hist of each feature.
    After logging and normalizing.
    """
    plt.figure(figsize=(20, 20))

    for i in range(dataset.data.shape[1]):
        plt.subplot(5, 3, i + 1)
        plt.hist(dataset.data[:, i], bins=50)
        plt.title(f"{dataset.labels[i]}: log={LOG_FEATURE[i]}, mean={dataset.means[i]:.3f}, std={dataset.stds[i]:.3f}")

    plt.tight_layout()
    plt.show()


def plot_2d(dataset, col1, col2, label1="", label2=""):
    """Scatter plot 2 features.
    """
    plt.figure()
    plt.scatter(dataset.raw_data[:, col1], dataset.raw_data[:, col2])

    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data", default="../data/new_data_5_filtered.csv")
    args = parser.parse_args()

    # Print dataset stats.
    dataset = FetDataset(args.data, "cpu")
    print("Dataset length:", len(dataset))
    x, y = dataset[0]
    print("  x:", x.shape, x.dtype, x)
    print("  y:", y.shape, y.dtype, y)

    plot_feature_dists(dataset)
    #plot_2d(dataset, 1, -4, "tox", "Rsp")
