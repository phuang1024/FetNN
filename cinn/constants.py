import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_NOISE = 1e-4
"""Additive Gaussian noise augmentation magnitude."""

NUM_BLOCKS = 16
"""Number of cINN coupling blocks."""
INIT_WEIGHT = 1e-2
"""Weight init magnitude."""

BWD_Z_MAG = 0.1

BATCH_SIZE = 128
LR = 2e-5
WEIGHT_DECAY = 2e-5
EPOCHS = 500
