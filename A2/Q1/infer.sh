#!/usr/bin/env bash
# COL8383 / ELL8299 / COL873 Assignment 2 - Task 1 Inference Shell Script
set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: ./infer.sh <mode: bias|qa> <absolute_path_to_output_dir> <input_jsonl>"
    exit 1
fi

MODE="$1"
OUTPUT_DIR="$2"
INPUT_JSONL="$3"

if [ "$MODE" != "bias" ] && [ "$MODE" != "qa" ]; then
    echo "Error: mode must be either 'bias' or 'qa', got: $MODE"
    exit 1
fi

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
echo "Running Task 1 Inference"
echo "Mode: $MODE"
echo "Output Directory: $OUTPUT_DIR"
echo "Input JSONL: $INPUT_JSONL"
echo "Execution Command: ${PYTHON_EXEC[*]}"
echo "=========================================="

"${PYTHON_EXEC[@]}" "$SCRIPT_DIR/infer.py" "$MODE" "$OUTPUT_DIR" "$INPUT_JSONL"

echo "Task 1 inference completed successfully."
