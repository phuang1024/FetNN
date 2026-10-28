import argparse
import torch

from dataset import FetDataset
from dnn_model import FetDNNModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("model")
    args = parser.parse_args()

    dataset = FetDataset(args.dataset, "cpu")

    model = FetDNNModel()
    model.load_state_dict(torch.load(args.model))

    x = torch.load("x.pt").unsqueeze(0)
    y = model(x)
    print(y)


if __name__ == "__main__":
    main()
