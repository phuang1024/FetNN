import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_DIM = 10
Y_DIM = 4

NUM_BLOCKS = 8
"""Number of cINN coupling blocks."""

INIT_WEIGHT = 1e-2
"""Weight init magnitude."""

BATCH_SIZE = 128
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 500
