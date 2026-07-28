#!/usr/bin/env bash
# One-command training pipeline
# Usage: ./scripts/train.sh [model_variant] [config]
# Example: ./scripts/train.sh yolov8m config/config.yaml
set -euo pipefail

MODEL="${1:-yolov8m}"
CONFIG="${2:-config/config.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "================================================"
echo " Ice Cream Shelf Detector — Training"
echo " Model  : $MODEL"
echo " Config : $CONFIG"
echo "================================================"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated."
fi

# Validate dataset first
echo ""
echo "Step 1/3 — Validating dataset…"
python -m src.data.validator \
    --images-dir data/splits/images/train \
    --labels-dir data/splits/labels/train \
    --config "$CONFIG" \
    --report reports/pre_train_validation.json || {
    echo "WARNING: Validation found issues. Check reports/pre_train_validation.json"
}

# Train
echo ""
echo "Step 2/3 — Training $MODEL…"
python -m src.models.trainer \
    --model "$MODEL" \
    --config "$CONFIG"

# Evaluate
echo ""
echo "Step 3/3 — Evaluating best weights…"
BEST_WEIGHTS=$(find runs -name "best.pt" | sort -r | head -1)
if [ -n "$BEST_WEIGHTS" ]; then
    python -m src.models.evaluator \
        --weights "$BEST_WEIGHTS" \
        --config "$CONFIG"
    echo "Evaluation complete. Report in reports/evaluation_report.json"
else
    echo "WARNING: No best.pt found — skipping evaluation."
fi

echo ""
echo "Training pipeline complete."
