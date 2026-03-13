"""
Training loop for CNN vs ViT CIFAR-10 experiments.

Usage
-----
    python src/train.py --model resnet --aug baseline --epochs 200
    python src/train.py --model vit --aug aggressive --data-fraction 0.25 --epochs 200
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from data import get_cifar10_loaders
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN/ViT on CIFAR-10")
    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet", "vit"],
                        help="Model architecture")
    parser.add_argument("--aug", type=str, required=True,
                        choices=["baseline", "standard", "aggressive"],
                        help="Augmentation regimen")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=20,
                        help="Linear warmup epochs before cosine decay")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--data-fraction", type=float, default=1.0,
                        help="Fraction of training data to use (0, 1]")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for CSV logs")
    parser.add_argument("--weight-dir", type=str, default="weights",
                        help="Directory for model checkpoints")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimiser, device, mixup_fn=None,
                    max_grad_norm=1.0):
    """Run one training epoch; return average loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        # Batch-level Mixup/CutMix for aggressive augmentation
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimiser.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute validation loss (CE) and accuracy."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * targets.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def run_name(model_name, aug_type, data_fraction):
    """Generate a consistent run identifier for file naming."""
    frac_str = f"frac{data_fraction:.2f}".replace(".", "")
    return f"{model_name}_{aug_type}_{frac_str}"


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  Model: {args.model}  |  Aug: {args.aug}  |  "
          f"Data: {args.data_fraction*100:.0f}%  |  Device: {device}")
    print(f"{'='*60}\n")

    # ---- Data ----
    train_loader, val_loader = get_cifar10_loaders(
        aug_type=args.aug,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_fraction=args.data_fraction,
    )
    print(f"Training samples: {len(train_loader.dataset):,}  |  "
          f"Validation samples: {len(val_loader.dataset):,}")

    # ---- Model ----
    model = get_model(args.model).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,}\n")

    # ---- Loss / Optimiser / Scheduler ----
    mixup_fn = None
    if args.aug == "aggressive":
        mixup_fn = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=1.0,
            switch_prob=0.5,
            num_classes=10,
        )
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # Linear warmup + cosine decay
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=1e-3, total_iters=args.warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs - args.warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimiser,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    # ---- Logging setup ----
    name = run_name(args.model, args.aug, args.data_fraction)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.weight_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, f"{name}.csv")
    weight_path = os.path.join(args.weight_dir, f"{name}_best.pt")

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])

    # ---- Training loop ----
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimiser, device, mixup_fn
        )
        val_loss, val_acc = evaluate(model, loader=val_loader, device=device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  |  "
              f"train_loss: {train_loss:.4f}  |  "
              f"val_loss: {val_loss:.4f}  |  "
              f"val_acc: {val_acc:.4f}  |  "
              f"{elapsed:.1f}s")

        csv_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}"])
        csv_file.flush()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), weight_path)

    csv_file.close()
    print(f"\nBest val accuracy: {best_acc:.4f}")
    print(f"Logs saved to:    {csv_path}")
    print(f"Weights saved to: {weight_path}\n")


if __name__ == "__main__":
    main()
