"""Display results of LDMOS Degradation sim.
First:
- Generate the sims: degr_test_nn.py
- Run the sims.
- Collect results with Weiman's script: collect_results.py
Then run this.
"""

import argparse
import csv

import numpy as np
import torch

LEGEND = ("Rsp", "BV", "Rout", "Vth")

# TCAD csv columns.
TCAD_COLS = (-5, -4, -3, -1)
# NN tensor columns.
NN_COLS = (-4, -3, -2, -1)


def read_tcad_results(file):
    data = []
    with open(file) as fp:
        reader = csv.reader(fp)
        for i, line in enumerate(reader):
            if i == 0:
                labels = line
            else:
                data.append(line)

    # Take hardcoded columns.
    col_names = [labels[i] for i in TCAD_COLS]
    print(f"TCAD: Taking indices {TCAD_COLS}: {col_names}")
    ret = []
    for d in data:
        ret.append([d[i] for i in TCAD_COLS])
    return np.array(ret)


def read_nn_results(file):
    data = torch.load(file)
    print(f"NN: Taking indices {NN_COLS}")
    ret = []
    for d in data:
        ret.append([d[i] for i in NN_COLS])
    return np.array(ret)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tcad", help="Path to all_simulation_results.csv")
    parser.add_argument("nn", help="Path to NN ground truth and recipes.")
    args = parser.parse_args()

    print(f"TCAD file: {args.tcad}")
    print(f"NN file: {args.nn}")

    tcad_data = read_tcad_results(args.tcad)
    nn_data = read_nn_results(args.nn)

    # Print results.
    for i in range(min(len(tcad_data), len(nn_data))):
        print(f"Sim {i+1} | GT   |")
        for j in range(len(LEGEND)):
            print(f"{LEGEND[j]:5} | {nn_data[i][j]:5} | {tcad_data[i][j]:5}")


if __name__ == "__main__":
    main()
