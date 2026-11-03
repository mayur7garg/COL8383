#!/usr/bin/env python3
"""
Task 2 Setup: Watermarking Generated Text
Generates and writes watermark configuration, secret key, and hash tables.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 Watermarking Setup")
    parser.add_argument("output_dir", type=str, help="Output directory to save watermark configuration")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B", help="Base model name")
    parser.add_argument("--secret_key", type=int, default=20261103, help="Secret integer key for PRNG hashing")
    parser.add_argument("--gamma_bi", type=float, default=0.5, help="Green list fraction for bigram hashing")
    parser.add_argument("--gamma_uni", type=float, default=0.5, help="Green list fraction for unigram hashing")
    parser.add_argument("--delta_bi", type=float, default=1.4, help="Logit bias for bigram green tokens")
    parser.add_argument("--delta_uni", type=float, default=0.6, help="Logit bias for unigram green tokens")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Task 2 Setup: Initializing in {output_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    vocab_size = len(tokenizer)
    print(f"Tokenizer loaded. Vocabulary size: {vocab_size}")

    # Identify special tokens to exclude from substitution perturbations
    all_special_ids = set(tokenizer.all_special_ids)
    non_special_tokens = [i for i in range(vocab_size) if i not in all_special_ids]
    print(f"Found {len(non_special_tokens)} non-special tokens for substitution.")

    # Generate pseudo-random token hashes using secret key
    # Using Mersenne Prime M31 = 2^31 - 1 = 2147483647
    rng = np.random.RandomState(args.secret_key)
    token_hashes = rng.randint(0, 2147483647, size=vocab_size, dtype=np.int64).tolist()

    config: Dict[str, Any] = {
        "task": "Task 2: Watermarking Generated Text",
        "model_name": args.model_name,
        "vocab_size": vocab_size,
        "secret_key": args.secret_key,
        "modulus": 2147483647,
        "hash_multiplier": 1000003,
        "gamma_bi": args.gamma_bi,
        "gamma_uni": args.gamma_uni,
        "delta_bi": args.delta_bi,
        "delta_uni": args.delta_uni,
        "token_hashes": token_hashes,
        "non_special_tokens": non_special_tokens,
    }

    config_path = output_dir / "watermark_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f)

    print(f"Watermark configuration written to {config_path}")
    print("Task 2 setup complete.")


if __name__ == "__main__":
    main()
