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
    return np.array(ret, dtype=float)


def read_nn_results(file):
    data = torch.load(file)
    print(f"NN: Taking indices {NN_COLS}")
    ret = []
    for d in data:
        ret.append([d[i] for i in NN_COLS])
    return np.array(ret, dtype=float)


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
        rel_errors = (tcad_data[i] - nn_data[i]) / nn_data[i] * 100

        print(f"Sim {i+1} | GT        | TCAD      | Rel error")
        # Specific formatting.
        print(f"{LEGEND[0]:5} | {nn_data[i][0] * 1e3:.3f}e-3  | {tcad_data[i][0] * 1e3:.3f}e-3  | {rel_errors[0]:.1f}%")
        print(f"{LEGEND[1]:5} | {nn_data[i][1]:.3f}    | {tcad_data[i][1]:.3f}    | {rel_errors[1]:.1f}%")
        print(f"{LEGEND[2]:5} | {nn_data[i][2] / 1e3:.3f}e3  | {tcad_data[i][2] / 1e3:.3f}e3  | {rel_errors[2]:.1f}%")
        print(f"{LEGEND[3]:5} | {nn_data[i][3]:.3f}     | {tcad_data[i][3]:.3f}     | {rel_errors[3]:.1f}%")
        # Mean rel error.
        mre = np.mean(np.abs(rel_errors))
        print(f"Abs mean rel error: {mre:.1f}%")


if __name__ == "__main__":
    main()
