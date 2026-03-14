"""
EuroSAT data loading with three augmentation pipelines and configurable
training-set fraction for data-efficiency experiments.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ----- EuroSAT RGB channel statistics (computed over full dataset) -----
EUROSAT_MEAN = (0.3444, 0.3803, 0.4078)
EUROSAT_STD = (0.2032, 0.1365, 0.1152)


def _baseline_transform():
    """Baseline: random crop + horizontal flip + normalise."""
    return transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD),
    ])


def _standard_transform():
    """Standard: baseline + RandAugment."""
    return transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD),
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
        transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD),
    ])


_AUG_MAP = {
    "baseline": _baseline_transform,
    "standard": _standard_transform,
    "aggressive": _aggressive_transform,
}


def _stratified_subset(targets, fraction, seed=42):
    """Return indices for a stratified subset with `fraction` of the data."""
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    selected_indices = []

    for cls in np.unique(targets):
        cls_indices = np.where(targets == cls)[0]
        n_keep = max(1, int(len(cls_indices) * fraction))
        selected_indices.extend(rng.choice(cls_indices, size=n_keep, replace=False))

    return sorted(selected_indices)


def _stratified_train_val_split(targets, val_fraction=0.2, seed=42):
    """Split indices into stratified train/val sets."""
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    train_indices = []
    val_indices = []

    for cls in np.unique(targets):
        cls_indices = np.where(targets == cls)[0]
        rng.shuffle(cls_indices)
        n_val = max(1, int(len(cls_indices) * val_fraction))
        val_indices.extend(cls_indices[:n_val])
        train_indices.extend(cls_indices[n_val:])

    return sorted(train_indices), sorted(val_indices)


def get_eurosat_loaders(
    aug_type="baseline",
    batch_size=128,
    num_workers=2,
    data_fraction=1.0,
    data_dir="./data",
):
    """Create EuroSAT train and validation DataLoaders.

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
        Root directory for EuroSAT download.

    Returns
    -------
    train_loader, val_loader : DataLoader pair
    """
    if aug_type not in _AUG_MAP:
        raise ValueError(f"Unknown aug_type '{aug_type}'. Choose from {list(_AUG_MAP)}")

    # Download full dataset (no built-in split)
    full_dataset = datasets.EuroSAT(
        root=data_dir, download=True, transform=None
    )

    # Stratified 80/20 train/val split
    all_targets = [s[1] for s in full_dataset.samples]
    train_indices, val_indices = _stratified_train_val_split(all_targets)

    # Build train and val datasets with appropriate transforms
    train_transform = _AUG_MAP[aug_type]()
    test_transform = _test_transform()

    train_dataset = datasets.EuroSAT(
        root=data_dir, download=False, transform=train_transform
    )
    val_dataset = datasets.EuroSAT(
        root=data_dir, download=False, transform=test_transform
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Sub-sample training data if requested
    if data_fraction < 1.0:
        train_targets = [all_targets[i] for i in train_indices]
        sub_indices = _stratified_subset(train_targets, data_fraction, seed=123)
        # Map back to original dataset indices
        sub_original_indices = [train_indices[i] for i in sub_indices]
        train_subset = Subset(train_dataset, sub_original_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
