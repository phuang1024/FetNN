"""Load data from CSV.
Run this file to visualize data.
"""

import csv

import numpy as np

# Whether to log features for the Degradation dataset.
DEGR_X_LOG = (
    False,
    False,
    True,
    True,
    True,
    False,
    False,
    False,
    False,
    False,
)
DEGR_Y_LOG = (
    False,
    False,
    True,
    False
)


def process_data(labels, x, y, x_log, y_log):
    """Normalize all feature axes.
    Log selected features.

    x, y: ndarray (N, D) data.
    x_log, y_log: ndarray bool (D,), whether to log each feature.
    """
    def process_array(data, do_log, index_offset=0):
        for i in range(data.shape[1]):
            if do_log[i]:
                data[:, i] = np.log(data[:, i])
                labels[i + index_offset] += " (log)"
            mean = np.mean(data[:, i])
            std = np.std(data[:, i])
            data[:, i] = (data[:, i] - mean) / std

    process_array(x, x_log)
    process_array(y, y_log, x.shape[1])


def load_ldmos_degr_data(file):
    """Load csv data from LDMOS Degradation template.
    10X, 4Y.
    """
    with open(file) as fp:
        reader = csv.reader(fp)
        data = []
        for i, line in enumerate(reader):
            if i == 0:
                labels = line
            else:
                data.append(list(map(float, line)))
    data = np.array(data)

    x = data[:, :10]
    y = data[:, 10:]
    process_data(labels, x, y, DEGR_X_LOG, DEGR_Y_LOG)
    return labels, x, y


def vis_data(labels, x, y):
    def add_plot(data, index):
        plt.subplot(5, 3, index + 1)
        plt.hist(data, bins=50)
        plt.title(labels[index])

    plt.figure(figsize=(20, 20))
    index = 0
    for i in range(x.shape[1]):
        add_plot(x[:, i], index)
        index += 1
    for i in range(y.shape[1]):
        add_plot(y[:, i], index)
        index += 1

    plt.tight_layout()
    plt.savefig("data.jpg")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    labels, x, y = load_ldmos_degr_data("new_data_4.csv")
    print("x:", x.shape, x.dtype)
    print("y:", y.shape, y.dtype)
    vis_data(labels, x, y)
