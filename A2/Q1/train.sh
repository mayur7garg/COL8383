#!/usr/bin/env bash
# COL8383 / ELL8299 / COL873 Assignment 2 - Task 1 Training Shell Script
set -e

if [ -z "$1" ]; then
    echo "Usage: ./train.sh <absolute_path_to_output_dir>"
    exit 1
fi

OUTPUT_DIR="$1"
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Determine Python execution command (prioritizing uv environment if available)
if command -v uv &>/dev/null && ([ -f "$REPO_ROOT/pyproject.toml" ] || [ -f "$REPO_ROOT/../pyproject.toml" ]); then
    PYTHON_EXEC=("uv" "run" "python")
elif [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_EXEC=("$REPO_ROOT/.venv/bin/python")
elif [ -f "$REPO_ROOT/.venv/Scripts/python.exe" ]; then
    PYTHON_EXEC=("$REPO_ROOT/.venv/Scripts/python.exe")
elif [ -f "$REPO_ROOT/../.venv/bin/python" ]; then
    PYTHON_EXEC=("$REPO_ROOT/../.venv/bin/python")
elif [ -f "$REPO_ROOT/../.venv/Scripts/python.exe" ]; then
    PYTHON_EXEC=("$REPO_ROOT/../.venv/Scripts/python.exe")
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC=("python3")
else
    PYTHON_EXEC=("python")
fi

echo "=========================================="
echo "Running Task 1 Training & Setup"
echo "Output Directory: $OUTPUT_DIR"
echo "Execution Command: ${PYTHON_EXEC[*]}"
echo "=========================================="

"${PYTHON_EXEC[@]}" "$SCRIPT_DIR/train.py" "$OUTPUT_DIR" --data_dir "$REPO_ROOT/data/public"

echo "Task 1 training completed successfully."
