import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Any, List

from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type = str)
    parser.add_argument("output_dir", type = str)
    return parser.parse_args()

def extract_condition_name(input_path: Path) -> str:
    # generated_clean_42 -> clean, generated_short_123 -> short
    m = re.fullmatch(r"generated_([a-zA-Z0-9_]+)_[0-9]+", input_path.stem)
    return m.group(1) if m else input_path.stem

def compute_z_score(tokens: List[int], config: Dict[str, Any]) -> float:
    T = len(tokens)
    if T <= 1:
        return 0.0

    token_hashes = config["token_hashes"]
    secret_key = config["secret_key"]
    modulus = config["modulus"]
    hash_multiplier = config["hash_multiplier"]
    gamma_bi = config["gamma_bi"]
    gamma_uni = config["gamma_uni"]

    threshold_bi = int(gamma_bi * modulus)
    threshold_uni = int(gamma_uni * modulus)
    vocab_len = len(token_hashes)

    N_bi = T - 1
    G_bi = 0
    for i in range(1, T):
        curr_t = tokens[i]
        prev_t = tokens[i - 1]
        if curr_t < vocab_len and prev_t < vocab_len:
            h_bi = (token_hashes[curr_t] + token_hashes[prev_t] * hash_multiplier + secret_key) % modulus
            if h_bi < threshold_bi:
                G_bi += 1

    expected_bi = gamma_bi * N_bi
    std_bi = math.sqrt(N_bi * gamma_bi * (1.0 - gamma_bi))
    z_bi = (G_bi - expected_bi) / std_bi if std_bi > 0 else 0.0

    N_uni = T
    G_uni = 0
    for i in range(T):
        curr_t = tokens[i]
        if curr_t < vocab_len:
            h_uni = (token_hashes[curr_t] + secret_key) % modulus
            if h_uni < threshold_uni:
                G_uni += 1

    expected_uni = gamma_uni * N_uni
    std_uni = math.sqrt(N_uni * gamma_uni * (1.0 - gamma_uni))
    z_uni = (G_uni - expected_uni) / std_uni if std_uni > 0 else 0.0

    return float(config["delta_bi"] * z_bi + config["delta_uni"] * z_uni)


def main():
    args = parse_args()
    input_path = Path(args.input_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_path = output_dir / "watermark_config.json"

    if not config_path.exists():
        print(f"Error: watermark_config.json not found in {output_dir}. Run setup.sh first.")
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    items = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Detecting watermark on {len(items)} items from {input_path.name}...")

    scored_records = []
    for it in items:
        tokens = tokenizer.encode(it.get("text", ""), add_special_tokens = False)
        scored_records.append({
            "id": it["id"],
            "score": compute_z_score(tokens, config)
        })

    condition = extract_condition_name(input_path)
    out_path = output_dir / f"scores_{condition}.jsonl"
    with out_path.open("w", encoding = "utf-8") as f:
        for rec in scored_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(scored_records)} detection scores to {out_path}")

if __name__ == "__main__":
    main()