import csv

import torch


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
