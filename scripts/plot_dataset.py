"""Plot Baliga's FOM for power transistors across a dataset.
BV^2 / Rsp
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_hist(data, title):
    plt.hist(data, bins=100)
    plt.xlabel(title)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def read_dataset(file):
    """
    Columns should be new_data_4.csv format.
    """
    with open(file, "r") as f:
        bvs = []
        rsps = []
        foms = []
        for i, line in enumerate(f.read().strip().split("\n")):
            if i == 0:
                continue
            values = line.strip().split(",")
            bvs.append(float(values[-3]))
            rsps.append(float(values[-4]))
            foms.append(bvs[-1] ** 2 / rsps[-1])

    bvs = np.array(bvs)
    rsps = np.array(rsps)
    foms = np.array(foms)
    return bvs, rsps, foms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="+")
    args = parser.parse_args()

    datas = []
    for file in args.data:
        datas.append(read_dataset(file))
        """
        for i, fom in enumerate(datas[-1][2]):
            if fom > 1.4e5:
                print(f"file={file}, dataIdx={i}, fom={fom}")
        """

    feat_labels = ["BV", "Rsp", "FOM"]
    for y in range(len(datas)):
        for x in range(3):
            plt.subplot(len(datas), 3, 3*y + x + 1)
            plt.hist(datas[y][x], bins=100)
            plt.xlabel(feat_labels[x])
            if x == 0:
                plt.ylabel(os.path.basename(args.data[y]))

    plt.tight_layout()
    plt.show()

    # Scatter BV vs Rsp.
    """
    plt.scatter(bvs, rsps)
    plt.xlabel("BV")
    plt.ylabel("Rsp")
    plt.show()
    """


if __name__ == "__main__":
    main()
