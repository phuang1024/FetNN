import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms.v2 as T

X_MEAN = 0.128
X_STD = 0.305

BATCH_SIZE = 256


def one_hot(y):
    """Convert int labels into one hot.
    y: Tensor int (B) 0-9.
    return: Tensor int (B, 10) one hot.
    """
    return torch.nn.functional.one_hot(y, num_classes=10)


def load_data():
    """Create dataloaders for MNIST.
    """
    x_trans = T.Compose([
        T.ToTensor(),
        T.Normalize([X_MEAN], [X_STD]),

        # Augs.
        T.GaussianNoise(0, 0.08, clip=False),
        #T.RandomResizedCrop((28, 28), (0.6, 1), (0.9, 1 / 0.9)),
    ])
    def y_trans(y):
        return one_hot(torch.tensor(y))

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
