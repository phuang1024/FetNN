"""Load and process dataset from CSV.
Run this file to visualize data.
"""

import csv

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

DATA_NOISE = 1e-4
"""Additive Gaussian noise augmentation magnitude."""


class FetDataset(Dataset):
    x_size = 11
    """First N features (in csv) are X."""

    # For new_data_5_filtered.csv
    log = (
        False, False,
        True, True, True,
        False, False, False,
        True, False, False,

        False, False, True, False
    )
    """Whether to log each feature."""

    orig_data: torch.Tensor
    """(N, D) raw data."""
    data: torch.Tensor
    """(N, D) data after normalization and log."""

    means: list[float]
    stds: list[float]
    labels: list[str]

    def __init__(self, path, device):
        """Init from CSV file.
        """
        self.device = device

        self.orig_data, self.labels = load_data(path)
        self.data, self.means, self.stds = preprocess_data(self.orig_data, self.log)

    def unnormalize(self, data):
        """Undo the normalize and log. Convert from logits to original units.
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
        #x, y = self.augment_data(x, y)
        return x, y


def augment_data(x, y):
    # Random noise.
    x = x + torch.randn_like(x) * DATA_NOISE
    y = y + torch.randn_like(y) * DATA_NOISE
    return x, y


def load_data(path):
    """Load raw data from CSV.
    return: data, labels
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
    return data, labels


def preprocess_data(data, log):
    """Log and normalize features.
    data: (N, D) raw data.
    log: (N,) bool. Whether to log each feature.
    return: normed_data, means, stds
    """
    means = []
    stds = []
    for i in range(data.shape[1]):
        # Log this feature.
        if log[i]:
            data[:, i] = torch.log1p(data[:, i])

        # Compute Z score.
        mean = torch.mean(data[:, i]).item()
        std = torch.std(data[:, i]).item() + 1e-3
        data[:, i] = (data[:, i] - mean) / std

        means.append(mean)
        stds.append(std)

    return data, means, stds
