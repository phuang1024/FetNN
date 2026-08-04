import torch.nn as nn

import FrEIA.framework as FF
import FrEIA.modules as FM

NUM_LAYERS = 20


def fc_subnet(dims_in, dims_out):
    return nn.Sequential(
        nn.Linear(dims_in, 512),
        nn.LeakyReLU(),
        nn.Linear(512, dims_out),
    )


"""
def make_model():
    model = FF.SequenceINN(28 * 28)
    for i in range(NUM_LAYERS):
        model.append(
            FM.AllInOneBlock,
            cond=0,
            cond_shape=(10,),
            subnet_constructor=fc_subnet,
            affine_clamping=1,
        )
    return model
"""

def make_model():
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
    model = FF.ReversibleGraphNet(nodes)
    return model
