# sws/data.py
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_loaders(
    dataset: str, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, int]:
    if dataset == "mnist":
        # Paper's Keras example scales to [0,1] (no mean/std normalization).
        tf_train = transforms.ToTensor()
        tf_test = transforms.ToTensor()
        train = datasets.MNIST(
            root="./data", train=True, download=True, transform=tf_train
        )
        test = datasets.MNIST(
            root="./data", train=False, download=True, transform=tf_test
        )
        return (
            DataLoader(
                train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            ),
            DataLoader(
                test,
                batch_size=2 * batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
            10,
        )

    if dataset in ("cifar10", "cifar100"):
        if dataset == "cifar10":
            mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            DS, num_classes = datasets.CIFAR10, 10
        else:
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            DS, num_classes = datasets.CIFAR100, 100
        tf_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        tf_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        train = DS(root="./data", train=True, download=True, transform=tf_train)
        test = DS(root="./data", train=False, download=True, transform=tf_test)
        return (
            DataLoader(
                train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            ),
            DataLoader(
                test,
                batch_size=2 * batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
            num_classes,
        )

    raise ValueError(f"Unknown dataset: {dataset}")
