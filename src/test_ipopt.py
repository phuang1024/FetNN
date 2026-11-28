"""Use Ipopt IPM optimizer.
"""

import argparse

import cyipopt
import numpy as np
import torch

from dataset import FetDataset, X_DIM
from inverse_gd import *
from model import FetDNNModel
from test_design import plot_2d_trajs

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="../data/new_data_4.csv")
parser.add_argument("--model", default="runs/adam_3layer/model_latest.pt")
args = parser.parse_args()

dataset = FetDataset(args.data, "cpu")

BV_TARGET = 18
VTH_TARGET = 1.5
bv_logit, vth_logit = dataset.normalize([[BV_TARGET, VTH_TARGET]], [-3, -1]).squeeze(0)

model = FetDNNModel()
model.load_state_dict(torch.load(args.model))
model.eval()

traj_ipopt = []


class ComputeIpmValues:
    """Compute objective, grad, etc. for Ipopt.
    Caches most recent result.
    """

    def __init__(self):
        self.last_x = None
        self.results = None

    def compute(self, x_array):
        if self.results is not None and np.allclose(x_array, self.last_x):
            return self.results
        self.last_x = x_array.copy()

        # Loss and grad.
        x, y = self.run_model(x_array)
        rsp_loss = y[0]
        rsp_loss.backward()

        loss = rsp_loss.item()
        loss_grad = x.grad.numpy()

        # Constraints.
        x, y = self.run_model(x_array)
        bv_loss = y[1]
        bv_loss.backward()
        bv_grad = x.grad.numpy()

        x, y = self.run_model(x_array)
        vth_loss = y[3]
        vth_loss.backward()
        vth_grad = x.grad.numpy()

        constrs = np.array([bv_loss.item(), vth_loss.item()])
        constrs_grad = np.array([bv_grad, vth_grad]).flatten()

        self.results = {
            "loss": loss,
            "loss_grad": loss_grad,
            "constrs": constrs,
            "constrs_grad": constrs_grad,
        }
        return self.results

    def run_model(self, x):
        """Run model to get a fresh graph.
        """
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        y = model(x.unsqueeze(0)).squeeze(0)
        return x, y


class FetProblem(cyipopt.Problem):
    """Design objective gradient,
    according to cyipopt format.
    """
    ipm_values = ComputeIpmValues()

    @classmethod
    def construct_problem(cls, dataset):
        """Create FetProblem instance using dataset bounds for min-max constraints.
        """
        x_mins = dataset.mins[:X_DIM]
        x_maxes = dataset.maxes[:X_DIM]
        nlp = cls(
            n=X_DIM,
            m=2,
            lb=x_mins,
            ub=x_maxes,
            cl=[bv_logit, vth_logit - 0.2],
            cu=[1e9, vth_logit + 0.2],
        )

        nlp.add_option("mu_strategy", "monotone")
        nlp.add_option("mu_init", 3e-2)

        nlp.add_option("max_iter", 50)
        return nlp

    def objective(self, x):
        return self.ipm_values.compute(x)["loss"]

    def gradient(self, x):
        return self.ipm_values.compute(x)["loss_grad"]

    def constraints(self, x):
        return self.ipm_values.compute(x)["constrs"]

    def jacobian(self, x):
        return self.ipm_values.compute(x)["constrs_grad"]

    def intermediate(self, *args, **kwargs):
        curr = self.get_current_iterate()
        traj_ipopt.append(curr["x"])
        return True


@torch.no_grad()
def plot_results():
    traj_x = torch.tensor(traj_ipopt, dtype=torch.float32)
    traj_y = model(traj_x)
    losses = torch.zeros([len(traj_x)])

    plot_2d_trajs(dataset, [(traj_x, traj_y, losses)], "Ipopt", 1)


def test_gd():
    """Test Adam vs SGD
    """
    designer = InverseGD(dataset, model)
    designer.add_criterion(wall_x(dataset), 1e-3)
    designer.add_criterion(y_minimize(0), 1)
    designer.add_criterion(y_floor(1, bv_logit), 1e-3)
    designer.add_criterion(y_target(3, vth_logit), 1)

    results = []
    results.append(designer.run_inverse_design(
        steps=30,
        lr=1e-1,
    ))

    plot_2d_trajs(dataset, results, ["GD method"], 1)
    return results[0]


def main():
    # test_gd()

    # Test this.
    ipm_values = ComputeIpmValues()
    result = ipm_values.compute(np.zeros([X_DIM]))
    #print(result)

    nlp = FetProblem.construct_problem(dataset)

    x0 = np.zeros([X_DIM], dtype=np.float32)
    x, info = nlp.solve(x0)

    plot_results()


if __name__ == "__main__":
    main()
