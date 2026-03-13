"""
Generate all experiment plots from CSV logs.

Produces 6 figures across two phases:
  Phase 1 (data efficiency): data_efficiency.png, data_efficiency_gap.png
  Phase 2 (augmentation):    val_accuracy_curves.png, final_accuracy_bars.png,
                              augmentation_gap.png, train_loss_curves.png
"""

import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

# ---- Style ----
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})

COLOURS = {
    "baseline": "#2196F3",
    "standard": "#FF9800",
    "aggressive": "#4CAF50",
}
MODEL_LABELS = {"resnet": "ResNet-18", "vit": "Tiny ViT"}


def load_all_logs(log_dir="logs"):
    """Load every CSV in log_dir and return a dict keyed by run name."""
    logs = {}
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".csv"):
            continue
        name = fname.replace(".csv", "")
        df = pd.read_csv(os.path.join(log_dir, fname))
        logs[name] = df
    return logs


def _parse_run_name(name):
    """Parse 'resnet_baseline_frac100' → (model, aug, fraction)."""
    m = re.match(r"(resnet|vit)_(baseline|standard|aggressive)_frac(\d+)", name)
    if not m:
        return None, None, None
    frac = int(m.group(3)) / 100.0
    return m.group(1), m.group(2), frac


# ---------- Phase 1 plots ----------

def plot_data_efficiency(logs, plot_dir):
    """Line plot: best val_acc vs data fraction for each model."""
    fractions = []
    for name, df in logs.items():
        model, aug, frac = _parse_run_name(name)
        if model is None or aug != "baseline":
            continue
        fractions.append((model, frac, df["val_acc"].max()))

    if not fractions:
        print("  [skip] No Phase 1 logs found for data_efficiency plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for model_key, label in MODEL_LABELS.items():
        pts = sorted([(f, a) for m, f, a in fractions if m == model_key])
        if not pts:
            continue
        xs, ys = zip(*pts)
        style = "-o" if model_key == "resnet" else "--s"
        ax.plot([x * 100 for x in xs], [y * 100 for y in ys], style,
                label=label, linewidth=2, markersize=7)

    ax.set_xlabel("Training Data (%)")
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("Data Efficiency: ResNet-18 vs Tiny ViT")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "data_efficiency.png"))
    plt.close(fig)
    print("  ✓ data_efficiency.png")


def plot_data_efficiency_gap(logs, plot_dir):
    """Bar plot: accuracy gap (ResNet − ViT) at each data fraction."""
    accs = {}
    for name, df in logs.items():
        model, aug, frac = _parse_run_name(name)
        if model is None or aug != "baseline":
            continue
        accs[(model, frac)] = df["val_acc"].max()

    fracs = sorted({f for (_, f) in accs})
    gaps = []
    for f in fracs:
        r = accs.get(("resnet", f))
        v = accs.get(("vit", f))
        if r is not None and v is not None:
            gaps.append((f, (r - v) * 100))

    if not gaps:
        print("  [skip] No Phase 1 logs found for data_efficiency_gap plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    xs, ys = zip(*gaps)
    bars = ax.bar([f"{x*100:.0f}%" for x in xs], ys, color="#E53935", width=0.5)
    for bar, val in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_xlabel("Training Data (%)")
    ax.set_ylabel("Accuracy Gap (ResNet − ViT) in pp")
    ax.set_title("Inductive Bias Gap vs Dataset Size")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "data_efficiency_gap.png"))
    plt.close(fig)
    print("  ✓ data_efficiency_gap.png")


# ---------- Phase 2 plots ----------

def _phase2_logs(logs):
    """Filter to Phase 2 runs: full dataset (frac100), all aug types."""
    phase2 = {}
    for name, df in logs.items():
        model, aug, frac = _parse_run_name(name)
        if model is None:
            continue
        if abs(frac - 1.0) < 1e-6:
            phase2[(model, aug)] = df
    return phase2


def plot_val_accuracy_curves(logs, plot_dir):
    """2×3 subplot grid of val_acc vs epoch."""
    phase2 = _phase2_logs(logs)
    if not phase2:
        print("  [skip] No Phase 2 logs found for val_accuracy_curves plot.")
        return

    aug_order = ["baseline", "standard", "aggressive"]
    model_order = ["resnet", "vit"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for row, model_key in enumerate(model_order):
        for col, aug in enumerate(aug_order):
            ax = axes[row][col]
            df = phase2.get((model_key, aug))
            if df is not None:
                ax.plot(df["epoch"], df["val_acc"] * 100,
                        color=COLOURS[aug], linewidth=1.5)
            ax.set_title(f"{MODEL_LABELS[model_key]} — {aug.title()}")
            ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel("Val Accuracy (%)")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Validation Accuracy Curves", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(plot_dir, "val_accuracy_curves.png"))
    plt.close(fig)
    print("  ✓ val_accuracy_curves.png")


def plot_final_accuracy_bars(logs, plot_dir):
    """Grouped bar chart: final accuracy per aug level, ResNet vs ViT."""
    phase2 = _phase2_logs(logs)
    if not phase2:
        print("  [skip] No Phase 2 logs found for final_accuracy_bars plot.")
        return

    aug_order = ["baseline", "standard", "aggressive"]
    import numpy as np
    x = np.arange(len(aug_order))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (model_key, label) in enumerate(MODEL_LABELS.items()):
        vals = []
        for aug in aug_order:
            df = phase2.get((model_key, aug))
            vals.append(df["val_acc"].max() * 100 if df is not None else 0)
        bars = ax.bar(x + i * width - width / 2, vals, width, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([a.title() for a in aug_order])
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("Final Accuracy: ResNet-18 vs Tiny ViT by Augmentation")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "final_accuracy_bars.png"))
    plt.close(fig)
    print("  ✓ final_accuracy_bars.png")


def plot_augmentation_gap(logs, plot_dir):
    """Accuracy gap (ResNet − ViT) across augmentation levels."""
    phase2 = _phase2_logs(logs)
    aug_order = ["baseline", "standard", "aggressive"]
    gaps = []
    for aug in aug_order:
        r = phase2.get(("resnet", aug))
        v = phase2.get(("vit", aug))
        if r is not None and v is not None:
            gaps.append((aug, (r["val_acc"].max() - v["val_acc"].max()) * 100))

    if not gaps:
        print("  [skip] No Phase 2 logs found for augmentation_gap plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    labels, vals = zip(*gaps)
    bars = ax.bar([l.title() for l in labels], vals, color="#7B1FA2", width=0.45)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}pp", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Accuracy Gap (ResNet − ViT) in pp")
    ax.set_title("Inductive Bias Gap vs Augmentation Level")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "augmentation_gap.png"))
    plt.close(fig)
    print("  ✓ augmentation_gap.png")


def plot_train_loss_curves(logs, plot_dir):
    """Overlay training loss for all Phase 2 runs."""
    phase2 = _phase2_logs(logs)
    if not phase2:
        print("  [skip] No Phase 2 logs found for train_loss_curves plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for (model_key, aug), df in sorted(phase2.items()):
        style = "-" if model_key == "resnet" else "--"
        ax.plot(df["epoch"], df["train_loss"], style,
                color=COLOURS[aug], linewidth=1.5,
                label=f"{MODEL_LABELS[model_key]} — {aug.title()}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Curves (solid=ResNet, dashed=ViT)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "train_loss_curves.png"))
    plt.close(fig)
    print("  ✓ train_loss_curves.png")


def main():
    log_dir = "logs"
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.isdir(log_dir):
        print(f"Error: log directory '{log_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    logs = load_all_logs(log_dir)
    if not logs:
        print(f"Error: no CSV files found in '{log_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(logs)} log file(s). Generating plots...\n")

    # Phase 1
    plot_data_efficiency(logs, plot_dir)
    plot_data_efficiency_gap(logs, plot_dir)

    # Phase 2
    plot_val_accuracy_curves(logs, plot_dir)
    plot_final_accuracy_bars(logs, plot_dir)
    plot_augmentation_gap(logs, plot_dir)
    plot_train_loss_curves(logs, plot_dir)

    print(f"\nAll plots saved to {plot_dir}/\n")


if __name__ == "__main__":
    main()
