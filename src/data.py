"""Load data from CSV.
Run this file to visualize data.
"""

import csv

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

DATA_NOISE = 1e-4
"""Additive Gaussian noise augmentation magnitude."""


class MosDataset(Dataset):
    """Dataset base class for MOS data generated with TCAD.
    X is list of scalar recipe params. Y is list of scalar electrical features.

    Usage:
        Create a subclass and define params below.
        On init, pass in the CSV data file.
        For inference, use unnormalize() to convert from logits to recipe.
    """
    x_size: int
    """First N features (in csv) are X."""
    log: list[bool]
    """Whether to log each feature."""

    data: torch.Tensor
    means: list[float]
    stds: list[float]

    def __init__(self, path, device):
        """Initialize from CSV file.
        """
        self.load_data(path)
        self.preprocess_data()
        self.device = device

    def load_data(self, path):
        """Sets ``self.data`` and ``self.labels``.
        """
        self.data = []
        with open(path) as fp:
            reader = csv.reader(fp)
            for i, line in enumerate(reader):
                if i == 0:
                    self.labels = line
                else:
                    self.data.append(list(map(float, line)))
        self.data = torch.tensor(self.data, dtype=torch.float)

    def preprocess_data(self):
        """Logs and normalizes features.
        Sets ``self.means`` and ``self.stds``.
        """
        self.means = []
        self.stds = []
        for i in range(self.data.shape[1]):
            if self.log[i]:
                self.data[:, i] = torch.log1p(self.data[:, i])

            mean = torch.mean(self.data[:, i]).item()
            std = torch.std(self.data[:, i]).item()
            self.data[:, i] = (self.data[:, i] - mean) / std
            self.means.append(mean)
            self.stds.append(std)

        self.data = self.data.to(self.device)

    def unnormalize(self, data):
        """Undo the normalize and log (given logits).
        data: Tensor (B, D).
            If D dimension is smaller than original CSV data,
            data is assumed to be the first D columns.
        """
        for i in range(data.shape[1]):
            data[:, i] = data[:, i] * self.stds[i] + self.means[i]
            if self.log[i]:
                data[:, i] = torch.expm1(data[:, i])
        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index, :self.x_size]
        y = self.data[index, self.x_size:]
        if True:
            self.augment(x, y)
        return x, y

    def augment(self, x, y):
        """In place.
        """
        # Random noise.
        x += torch.randn_like(x) * DATA_NOISE
        y += torch.randn_like(y) * DATA_NOISE


class LdmosDegrData(MosDataset):
    x_size = 10
    log = (
        False, False,
        True, True, True,
        False, False, False,
        False, False,

        False, False, True, False
    )
    """
    # TODO testing all log.
    log = (
        True, True,
        True, True, True,
        True, True, True,
        True, True,

        True, True, True, True
    )
    """


def split_train_val(dataset, ratio, batch_size):
    """Helper func to split dataset.
    """
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_data, val_data = random_split(dataset, (train_len, val_len))

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def vis_data(dataset: MosDataset):
    """Plot hist of each feature.
    After logging and normalizing.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 20))

    for i in range(dataset.data.shape[1]):
        plt.subplot(5, 3, i + 1)
        plt.hist(dataset.data[:, i], bins=50)
        plt.title(f"{dataset.labels[i]}: log={dataset.log[i]}, mean={dataset.means[i]:.3f}, std={dataset.stds[i]:.3f}")

    plt.tight_layout()
    plt.savefig("data.jpg")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    args = parser.parse_args()

    dataset = LdmosDegrData(args.data)
    print("Dataset length:", len(dataset))
    x, y = dataset[0]
    print("  x:", x.shape, x.dtype, x)
    print("  y:", y.shape, y.dtype, y)

    vis_data(dataset)
