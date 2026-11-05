#!/usr/bin/env bash
# Usage: ./perturb.sh <input_jsonl> <output_dir> <seed> [condition]

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$DIR/perturb.py" "$1" "$2" "$3" "${@:4}"
