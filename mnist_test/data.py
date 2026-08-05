import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms.v2 as T

BATCH_SIZE = 256


def load_data():
    """Create dataloaders for MNIST.
    """
    x_trans = T.Compose([
        T.ToTensor(),
        # Noise is necessary to prevent large jacobian.
        T.GaussianNoise(0, 0.1, clip=False),
        # RRCrop seems to work fine. Might need regularization.
        #T.RandomResizedCrop((28, 28), (0.6, 1), (0.9, 1 / 0.9)),
    ])
    def y_trans(y):
        return one_hot(torch.tensor(y), num_classes=10)

    train_data = datasets.MNIST(
        root="mnist",
        train=True,
        download=True,
        transform=x_trans,
        target_transform=y_trans,
    )
    test_data = datasets.MNIST(
        root="mnist",
        train=False,
        download=True,
        transform=x_trans,
        target_transform=y_trans,
    )

    loader_args = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
    }
    train_loader = DataLoader(train_data, **loader_args)
    test_loader = DataLoader(test_data, **loader_args)
    return train_loader, test_loader
