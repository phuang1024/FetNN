"""Discard categorical data in training dataset.
"""

import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    with open(args.input, "r") as in_file, open(args.output, "w") as out_file:
        for i, line in enumerate(in_file):
            write = True
            if i > 0:
                values = list(map(float, line.strip().split(",")))
                # Check nwell 2 and 3.
                if values[3] < 1.1e10 or values[4] < 1.1e10:
                    write = False
                # Check high_k.
                if values[10] > 0.5:
                    write = False

            if write:
                out_file.write(line)


if __name__ == "__main__":
    main()
