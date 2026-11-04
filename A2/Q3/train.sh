#!/usr/bin/env bash
# Usage: ./train.sh <absolute_path_to_output_dir>

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$DIR/train.py" "$1" "${@:2}"
