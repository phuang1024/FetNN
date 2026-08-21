import argparse

import matplotlib.pyplot as plt
import torch


parser = argparse.ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()

params = torch.load(args.file, map_location="cpu")
params = torch.cat([torch.abs(p.flatten()) for p in params.values()])
params = params.numpy()

plt.hist(params, 20)
plt.yscale("log")
plt.savefig("weight_dist.jpg")
