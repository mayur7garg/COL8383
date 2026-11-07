import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from transformers import AutoConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type = str)
    parser.add_argument("--model_name", type = str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--secret_key", type = int, default=20261103)
    parser.add_argument("--gamma_bi", type = float, default=0.5)
    parser.add_argument("--gamma_uni", type = float, default=0.5)
    parser.add_argument("--delta_bi", type = float, default = 0.0)
    parser.add_argument("--delta_uni", type = float, default = 1.5)
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Task 2 Setup: Initializing in {output_dir}")

    try:
        vocab_size = AutoConfig.from_pretrained(args.model_name, local_files_only = True).vocab_size
    except Exception:
        vocab_size = AutoConfig.from_pretrained(args.model_name).vocab_size
    print(f"Loaded model {args.model_name} with vocab size: {vocab_size}")

    # Mersenne Prime M31
    rng = np.random.RandomState(args.secret_key)
    token_hashes = rng.randint(0, 2147483647, size = vocab_size, dtype = np.int64).tolist()

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
        "token_hashes": token_hashes
    }

    config_path = output_dir / "watermark_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f)

    print(f"Watermark configuration written to {config_path}")
    print("Task 2 setup complete.")

if __name__ == "__main__":
    main()