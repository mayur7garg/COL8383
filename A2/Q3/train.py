import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from transformers import set_seed

from infer import answer_with_features, confidence, contains, load_model, read_jsonl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type = str)
    parser.add_argument("--model_name", type = str, default = "Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_train_samples", type = int, default = 0)
    parser.add_argument("--num_samples", type = int, default = 0)
    parser.add_argument("--temperature", type = float, default = 1.0)
    parser.add_argument("--max_new_tokens", type = int, default = 32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--target_risk", type = float, default = 0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def find_threshold(conf: np.ndarray, labels: np.ndarray, target_risk: float) -> float:
    for theta in np.sort(np.unique(conf)):
        answered = conf >= theta
        if 1.0 - labels[answered].mean() <= target_risk:
            return float(theta)
    return 1.01

def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents = True, exist_ok = True)
    print(f"Task 3 Setup & Calibration: Output dir = {output_dir}")

    data_dir = Path(__file__).resolve().parent.parent / "data" / "public"
    train_file = data_dir / "q3_qa_train.jsonl"

    if not train_file.exists():
        print(f"Error: Training file not found at {train_file}")
        sys.exit(1)

    items = read_jsonl(train_file)
    if args.max_train_samples:
        items = items[: args.max_train_samples]
    print(f"Loaded {len(items)} training items for calibration.")

    config = {
        "model_name": args.model_name,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
    }

    tokenizer, model = load_model(args.model_name)
    answers, features = answer_with_features(items, tokenizer, model, config)
    labels = np.array([int(any(contains(a, g) for g in item["answers"])) for a, item in zip(answers, items)])
    print(f"Training generation accuracy: {labels.mean():.2%} ({labels.sum()}/{len(labels)})")

    clf = LogisticRegression().fit(features, labels)
    config["coef"] = clf.coef_[0].tolist()
    config["intercept"] = float(clf.intercept_[0])
    conf = confidence(features, config)
    threshold = find_threshold(conf, labels, args.target_risk)
    config["abstention_threshold"] = threshold
    answered = conf >= threshold
    print(f"Logistic coef={config['coef']}, intercept={config['intercept']:.4f}")
    print(f"Threshold {threshold:.4f}: train coverage = {answered.mean():.2%}, train risk = {1.0 - labels[answered].mean() if answered.any() else 0.0:.2%}")

    config_path = output_dir / "calibration_config.json"
    with config_path.open("w", encoding = "utf-8") as f:
        json.dump(config, f, indent = 2)

    print(f"Calibration configuration saved successfully to {config_path}")
    print("Task 3 training complete.")

if __name__ == "__main__":
    main()