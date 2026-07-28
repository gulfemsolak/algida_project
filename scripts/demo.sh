#!/usr/bin/env bash
# One-command Streamlit dashboard launch
# Usage: ./scripts/demo.sh [port]
set -euo pipefail

PORT="${1:-8501}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

for VENV_DIR in venv.nosync .venv.nosync venv .venv; do
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR"/bin/activate
        break
    fi
done

echo "================================================"
echo " Ice Cream Shelf Analyzer Dashboard"
echo " URL: http://localhost:$PORT"
echo " Press Ctrl+C to stop."
echo "================================================"

streamlit run dashboard/app.py \
    --server.port "$PORT" \
    --server.headless false \
    --browser.gatherUsageStats false
