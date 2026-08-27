import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(train=True):
    """Returns torchvision image transformation pipeline.

    Args:
        train (bool): If True, applies training augmentations. If False,
          applies evaluation transforms.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )


def get_dataloaders(
    data_dir="./data", batch_size=64, dataset_name="FashionMNIST"
):
    """Downloads dataset and constructs train/val PyTorch DataLoaders."""
    dataset_cls = getattr(datasets, dataset_name, datasets.FashionMNIST)

    train_dataset = dataset_cls(
        root=data_dir, train=True, download=True, transform=get_transforms(train=True)
    )

    val_dataset = dataset_cls(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader