#!/usr/bin/env bash
# Usage: ./infer.sh <input_jsonl> <output_dir>

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$DIR/infer.py" "$1" "$2"
