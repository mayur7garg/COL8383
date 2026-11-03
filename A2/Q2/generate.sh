#!/usr/bin/env bash
# Usage: ./generate.sh <input_jsonl> <output_dir> <seed>

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$DIR/generate.py" "$1" "$2" "$3"
