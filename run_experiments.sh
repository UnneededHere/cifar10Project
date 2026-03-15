#!/usr/bin/env bash
# ==============================================================
#  Run all CNN vs ViT EuroSAT experiments
#  Phase 2: Augmentation comparison at 10% data (4 runs)
# ==============================================================
set -e

EPOCHS=${1:-200}          # pass epoch count as first arg, default 200
SRC_DIR="$(dirname "$0")/src"

echo "============================================================"
echo "  EuroSAT CNN vs ViT Experiment Suite"
echo "  Epochs per run: ${EPOCHS}"
echo "============================================================"

# ---- Create output directories ----
mkdir -p weights logs plots

# ==================================================================
#  PHASE 2 — Augmentation Comparison (10% data, varying aug)
# ==================================================================
echo ""
echo "############################################################"
echo "  PHASE 2: Augmentation Comparison (10% Data)"
echo "############################################################"

MODELS="resnet vit"
AUGS="baseline standard aggressive"

for model in $MODELS; do
    for aug in $AUGS; do
        # Skip baseline@10% — already trained in Phase 1
        if [ "$aug" = "baseline" ]; then
            echo ""
            echo "  [skip] ${model} baseline frac0.10 — already trained in Phase 1"
            continue
        fi

        echo ""
        echo "------------------------------------------------------------"
        echo "  Running: model=${model}  aug=${aug}  data=0.1"
        echo "------------------------------------------------------------"
        python "${SRC_DIR}/train.py" \
            --model "$model" \
            --aug "$aug" \
            --data-fraction 0.1 \
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

sudo shutdown -h now