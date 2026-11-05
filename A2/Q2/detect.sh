#!/usr/bin/env bash
# Usage: ./detect.sh <input_jsonl> <output_dir>

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$DIR/detect.py" "$1" "$2"
