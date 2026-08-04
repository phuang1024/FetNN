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
