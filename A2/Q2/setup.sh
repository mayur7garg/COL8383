#!/usr/bin/env bash
# Usage: ./setup.sh <absolute_path_to_output_dir>

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$DIR/setup.py" "$1"
