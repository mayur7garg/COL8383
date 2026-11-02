#!/usr/bin/env bash
# Usage: ./infer.sh <bias|qa> <absolute_path_to_output_dir> <input_jsonl> [--baseline]

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$DIR/infer.py" "$1" "$2" "$3" "${@:4}"