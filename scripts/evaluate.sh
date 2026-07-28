#!/usr/bin/env bash
# One-command evaluation on the latest trained model
# Usage: ./scripts/evaluate.sh [weights_path] [split]
set -euo pipefail

WEIGHTS="${1:-}"
SPLIT="${2:-test}"
CONFIG="config/config.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Auto-find latest best.pt if not provided
if [ -z "$WEIGHTS" ]; then
    WEIGHTS=$(find runs -name "best.pt" | sort -r | head -1)
    if [ -z "$WEIGHTS" ]; then
        echo "ERROR: No trained weights found. Run scripts/train.sh first."
        exit 1
    fi
    echo "Using latest weights: $WEIGHTS"
fi

echo "================================================"
echo " Ice Cream Shelf Detector — Evaluation"
echo " Weights : $WEIGHTS"
echo " Split   : $SPLIT"
echo "================================================"

python -m src.models.evaluator \
    --weights "$WEIGHTS" \
    --config "$CONFIG" \
    --split "$SPLIT" \
    --output-dir reports/

REPORT="reports/evaluation_report.json"
if [ -f "$REPORT" ]; then
    echo ""
    echo "Generating HTML report…"
    python -m src.analysis.report_generator "$REPORT" \
        --output reports/evaluation_report.html
    echo "HTML report saved → reports/evaluation_report.html"
fi

echo ""
echo "Evaluation complete."
