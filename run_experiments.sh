#!/usr/bin/env bash
# ==============================================================
#  Run all CNN vs ViT CIFAR-10 experiments
#  Phase 1: Data-efficiency sweep  (8 runs)
#  Phase 2: Augmentation comparison (6 runs)
# ==============================================================
set -e

EPOCHS=${1:-50}           # pass epoch count as first arg, default 50
SRC_DIR="$(dirname "$0")/src"

echo "============================================================"
echo "  CIFAR-10 CNN vs ViT Experiment Suite"
echo "  Epochs per run: ${EPOCHS}"
echo "============================================================"

# ---- Create output directories ----
mkdir -p weights logs plots

# ==================================================================
#  PHASE 1 — Data Efficiency (baseline aug, varying data fractions)
# ==================================================================
echo ""
echo "############################################################"
echo "  PHASE 1: Data Efficiency Sweep"
echo "############################################################"

MODELS="resnet vit"
FRACTIONS="0.1 0.25 0.5 1.0"

for model in $MODELS; do
    for frac in $FRACTIONS; do
        echo ""
        echo "------------------------------------------------------------"
        echo "  Running: model=${model}  aug=baseline  data=${frac}"
        echo "------------------------------------------------------------"
        python "${SRC_DIR}/train.py" \
            --model "$model" \
            --aug baseline \
            --data-fraction "$frac" \
            --epochs "$EPOCHS"
    done
done

# ==================================================================
#  PHASE 2 — Augmentation Comparison (100% data, varying aug)
# ==================================================================
echo ""
echo "############################################################"
echo "  PHASE 2: Augmentation Comparison"
echo "############################################################"

AUGS="baseline standard aggressive"

for model in $MODELS; do
    for aug in $AUGS; do
        # Skip baseline@100% — already trained in Phase 1
        if [ "$aug" = "baseline" ]; then
            echo ""
            echo "  [skip] ${model} baseline frac1.0 — already trained in Phase 1"
            continue
        fi

        echo ""
        echo "------------------------------------------------------------"
        echo "  Running: model=${model}  aug=${aug}  data=1.0"
        echo "------------------------------------------------------------"
        python "${SRC_DIR}/train.py" \
            --model "$model" \
            --aug "$aug" \
            --data-fraction 1.0 \
            --epochs "$EPOCHS"
    done
done

# ==================================================================
#  PLOTTING
# ==================================================================
echo ""
echo "############################################################"
echo "  Generating plots"
echo "############################################################"
python "${SRC_DIR}/plot.py"

echo ""
echo "============================================================"
echo "  All experiments complete!"
echo "  Logs:    logs/"
echo "  Weights: weights/"
echo "  Plots:   plots/"
echo "============================================================"
