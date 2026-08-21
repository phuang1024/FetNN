"""cINN model.
"""

import torch
import torch.nn as nn

import FrEIA.framework as FF
import FrEIA.modules as FM

from constants import *


def fc_subnet(dims_in, dims_out):
    """Fully connected subnet for each INN block.
    """
    return nn.Sequential(
        nn.Linear(dims_in, 1024),
        nn.LeakyReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, dims_out),
    )


def make_model(dim, cond_dim):
    """
    dim: Dimension of input and output.
    cond_dim: Dimension of condition.
    """
    model = FF.SequenceINN(dim)
    for _ in range(NUM_BLOCKS):
        model.append(FM.PermuteRandom)
        model.append(
            FM.GLOWCouplingBlock,
            cond=0,
            cond_shape=[cond_dim],
            subnet_constructor=fc_subnet,
        )
    return model


def init_weights(model, std):
    """Gaussian weight initialization.
    """
    for param in model.parameters():
        if param.requires_grad:
            param.data = std * torch.randn_like(param)


if __name__ == "__main__":
    model = make_model(10, 4)
    print(model)
