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
    cond = FF.ConditionNode(10)

    nodes = []
    nodes.append(FF.InputNode(784))
    for i in range(NUM_LAYERS):
        nodes.append(FF.Node(
            nodes[-1],
            FM.PermuteRandom,
            {"seed": i},
        ))
        nodes.append(FF.Node(
            nodes[-1],
            FM.GLOWCouplingBlock,
            {
                "subnet_constructor": fc_subnet,
                "clamp": 1,
            },
            conditions=cond,
        ))

    nodes.append(FF.OutputNode(nodes[-1]))
    nodes.append(cond)
    model = FF.ReversibleGraphNet(nodes, verbose=False)
    return model


def init_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            """
            if name.endswith(("2.weight", "2.bias")):
                # Last layer.
                param.data = torch.zeros_like(param.data)
            else:
                param.data = 1e-3 * torch.randn_like(param)
            """
            param.data = 0.01 * torch.randn_like(param)
