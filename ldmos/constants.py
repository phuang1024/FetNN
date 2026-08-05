import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_DIM = 10
Y_DIM = 4

NUM_BLOCKS = 4
"""Number of cINN coupling blocks."""

BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 0
EPOCHS = 10
