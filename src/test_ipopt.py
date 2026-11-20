"""Use Ipopt IPM optimizer.
"""

import argparse

import cyipopt
import numpy as np
import torch

from dataset import FetDataset, X_DIM
from model import FetDNNModel
from test_design import plot_2d_trajs

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="../data/new_data_5_filtered.csv")
parser.add_argument("--model", default="runs/32mlp/model_e2999.pt")
args = parser.parse_args()

dataset = FetDataset(args.data, "cpu")

model = FetDNNModel()
model.load_state_dict(torch.load(args.model))
model.eval()

traj_ipopt = []


class FetProblem(cyipopt.Problem):
    """Design objective gradient,
    according to cyipopt format.
    """

    @torch.no_grad()
    def objective(self, x):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = model(x).squeeze(0)

        loss = y[0]
        return loss.item()

    def gradient(self, x):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        x.requires_grad_(True)
        y = model(x).squeeze(0)

        loss = y[0]
        loss.backward()

        grad = x.grad.squeeze(0).numpy()
        return grad

    def constraints(self, x):
        pass

    def intermediate(self, *args, **kwargs):
        curr = self.get_current_iterate()
        traj_ipopt.append(curr["x"])

        return True


@torch.no_grad()
def plot_results():
    traj_x = torch.tensor(traj_ipopt, dtype=torch.float32)
    traj_y = model(traj_x)
    losses = torch.zeros([len(traj_x)])

    plot_2d_trajs(dataset, [(traj_x, traj_y, losses)], "Ipopt")


def main():
    nlp = FetProblem(
        n=X_DIM,
        m=0,
        #lb=lb,
        #ub=ub,
        #cl=cl,
        #cu=cu,
    )
    nlp.add_option("max_iter", 300)

    x0 = np.zeros([X_DIM], dtype=np.float32)
    x, info = nlp.solve(x0)

    plot_results()


if __name__ == "__main__":
    main()
