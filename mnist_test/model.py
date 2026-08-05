import torch
import torch.nn as nn

import FrEIA.framework as FF
import FrEIA.modules as FM

NUM_LAYERS = 20


def fc_subnet(dims_in, dims_out):
    """Fully connected subnet for each INN layer.
    Input size: 784/2 + 10
    """
    return nn.Sequential(
        nn.Linear(dims_in, 512),
        nn.ReLU(),
        nn.Linear(512, dims_out),
    )


def make_model():
    """Use Graph API to create cINN model.
    """
    model = FF.SequenceINN(784)
    for i in range(NUM_LAYERS):
        model.append(FM.PermuteRandom)
        model.append(
            FM.GLOWCouplingBlock,
            cond=0,
            cond_shape=[10],
            subnet_constructor=fc_subnet,
        )
    return model


def init_weights(model):
    for param in model.parameters():
        if param.requires_grad:
            param.data = 0.01 * torch.randn_like(param)
