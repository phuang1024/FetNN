"""Load and process dataset from CSV.
Run this file to visualize data.
"""

import csv

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class FetDataset(Dataset):
    x_dim = 11
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

    raw_data: torch.Tensor
    """(N, D) raw data."""
    data: torch.Tensor
    """(N, D) data after normalization and log."""
    labels: list[str]

    # Per feature stats.
    means: list[float]
    stds: list[float]
    """Mean and std after logging."""
    mins: list[float]
    maxes: list[float]
    """Min and max after logging and Z score norm."""

    def __init__(self, path, device):
        """Init from CSV file.
        """
        self.device = device

        self.raw_data, self.labels = self.load_data(path)
        self.data, self.means, self.stds = self.preprocess_data()

    def load_data(self, path):
        """Load raw data from CSV.
        Sets self.raw_data, self.labels.
        """
        self.raw_data = []
        with open(path) as fp:
            reader = csv.reader(fp)
            for i, line in enumerate(reader):
                # First line is labels.
                if i == 0:
                    self.labels = line
                else:
                    self.raw_data.append(list(map(float, line)))

        self.raw_data = torch.tensor(self.raw_data, dtype=torch.float)

    def preprocess_data(self):
        """Log and normalize features.
        """
        self.data = self.raw_data.clone()

        self.means = []
        self.stds = []
        self.mins = []
        self.maxes = []
        for i in range(self.raw_data.shape[1]):
            # Log this feature.
            if self.log[i]:
                self.data[:, i] = torch.log1p(self.raw_data[:, i])

            # Z score normalization.
            mean = torch.mean(self.data[:, i]).item()
            std = torch.std(self.data[:, i]).item() + 1e-3
            self.data[:, i] = (self.data[:, i] - mean) / std

            self.means.append(mean)
            self.stds.append(std)

            # Min and max stats.
            self.mins.append(torch.min(self.data[:, i]).item())
            self.maxes.append(torch.max(self.data[:, i]).item())

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
        x = self.data[index, :self.x_dim]
        y = self.data[index, self.x_dim:]
        #x, y = augment_data(x, y)
        return x, y


def augment_data(x, y, noise=1e-4):
    # Random noise.
    x = x + torch.randn_like(x) * noise
    y = y + torch.randn_like(y) * noise
    return x, y


def split_train_val(dataset, ratio, batch_size):
    """Helper func to split dataset.
    """
    train_len = int(len(dataset) * ratio)
    val_len = len(dataset) - train_len
    train_data, val_data = random_split(dataset, (train_len, val_len))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
