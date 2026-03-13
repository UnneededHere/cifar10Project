"""
CIFAR-10 data loading with three augmentation pipelines and configurable
training-set fraction for data-efficiency experiments.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ----- CIFAR-10 channel statistics -----
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _baseline_transform():
    """Baseline: random crop + horizontal flip + normalise."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _standard_transform():
    """Standard: baseline + RandAugment."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _aggressive_transform():
    """Aggressive: same image-level transforms as standard.
    Mixup/CutMix are batch-level and applied inside the training loop.
    """
    return _standard_transform()


def _test_transform():
    """Deterministic transform for validation/test."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


_AUG_MAP = {
    "baseline": _baseline_transform,
    "standard": _standard_transform,
    "aggressive": _aggressive_transform,
}


def _stratified_subset(dataset, fraction, seed=42):
    """Return a Subset with `fraction` of the data, preserving class balance."""
    if fraction >= 1.0:
        return dataset

    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    selected_indices = []

    for cls in np.unique(targets):
        cls_indices = np.where(targets == cls)[0]
        n_keep = max(1, int(len(cls_indices) * fraction))
        selected_indices.extend(rng.choice(cls_indices, size=n_keep, replace=False))

    return Subset(dataset, sorted(selected_indices))


def get_cifar10_loaders(
    aug_type="baseline",
    batch_size=128,
    num_workers=2,
    data_fraction=1.0,
    data_dir="./data",
):
    """Create CIFAR-10 train and validation DataLoaders.

    Parameters
    ----------
    aug_type : str
        One of 'baseline', 'standard', 'aggressive'.
    batch_size : int
        Mini-batch size.
    num_workers : int
        DataLoader worker count.
    data_fraction : float
        Fraction of training data to use (0, 1]. Stratified sampling.
    data_dir : str
        Root directory for CIFAR-10 download.

    Returns
    -------
    train_loader, val_loader : DataLoader pair
    """
    if aug_type not in _AUG_MAP:
        raise ValueError(f"Unknown aug_type '{aug_type}'. Choose from {list(_AUG_MAP)}")

    train_transform = _AUG_MAP[aug_type]()
    test_transform = _test_transform()

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Sub-sample training data if requested
    if data_fraction < 1.0:
        train_dataset = _stratified_subset(train_dataset, data_fraction)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
