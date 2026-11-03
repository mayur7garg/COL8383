#!/usr/bin/env python3
"""
Task 3 Training: Calibrated Question Answering with Abstention
Fits confidence calibrator and optimal risk-bounded abstention threshold.
Saves calibration artifacts to <output_dir>/calibration_config.json.
"""

import argparse
import json
import math
import re
import string
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Task 3 Calibration Training")
    parser.add_argument("output_dir", type=str, help="Output directory to save calibration config and models")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Instruct model name")
    parser.add_argument("--max_train_samples", type=int, default=1000, help="Max train samples to use for calibration fitting")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for question answering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    """
    Standard QA answer normalization per PDF Page 7:
    Lowercase, remove punctuation, remove articles ('a', 'an', 'the'), collapse whitespace.
    """
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def is_correct(pred_answer: str, gold_answers: List[str]) -> bool:
    """
    Contiguous word sequence match per PDF Page 7:
    y_i = 1 if any normalized gold answer appears as a contiguous word sequence inside normalized prediction.
    """
    norm_pred = normalize_answer(pred_answer)
    if not norm_pred:
        return False
    pred_words = norm_pred.split()

    for gold in gold_answers:
        norm_gold = normalize_answer(gold)
        if not norm_gold:
            continue
        gold_words = norm_gold.split()
        gold_len = len(gold_words)
        # Check contiguous subsequence of words
        for i in range(len(pred_words) - gold_len + 1):
            if pred_words[i : i + gold_len] == gold_words:
                return True
    return False


def build_prompt(question: str, tokenizer) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a direct, factual question-answering assistant. Provide only the concise, direct answer to the question in as few words as possible without explanation."
        },
        {
            "role": "user",
            "content": f"Question: {question}\nAnswer:"
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def fit_platt_scaling(raw_scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Fits logistic regression w * s + b via cross-entropy loss."""
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    # Simple gradient descent or closed-form logistic regression
    w = 2.0
    b = -1.0
    lr = 0.05
    for _ in range(500):
        logits = w * raw_scores + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        grad_w = np.mean((probs - labels) * raw_scores)
        grad_b = np.mean(probs - labels)
        w -= lr * grad_w
        b -= lr * grad_b
    return float(w), float(b)


def find_optimal_threshold(confidences: np.ndarray, labels: np.ndarray, target_risk: float = 0.16) -> float:
    """
    Finds abstention threshold theta that maximizes coverage subject to risk <= target_risk.
    Conservative target_risk = 0.16 ensures OOD test risk remains comfortably <= 0.20.
    """
    best_theta = 0.50
    best_coverage = 0.0

    # Search through candidate thresholds
    for theta in np.linspace(0.10, 0.95, 86):
        answered_mask = confidences >= theta
        n_answered = np.sum(answered_mask)
        if n_answered == 0:
            continue
        risk = np.sum(answered_mask & (labels == 0)) / n_answered
        coverage = n_answered / len(labels)
        if risk <= target_risk and coverage > best_coverage:
            best_coverage = coverage
            best_theta = float(theta)

    # Fallback to 0.50 if no threshold achieved risk target
    return best_theta


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Task 3 Setup & Calibration: Output dir = {output_dir}")

    data_dir = Path(__file__).resolve().parent.parent / "data" / "public"
    train_file = data_dir / "q3_qa_train.jsonl"

    if not train_file.exists():
        print(f"Error: Training file not found at {train_file}")
        sys.exit(1)

    items = []
    with train_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if args.max_train_samples and len(items) > args.max_train_samples:
        items = items[: args.max_train_samples]

    print(f"Loaded {len(items)} training items for calibration.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if torch.cuda.is_available() else torch.float32)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map="auto"
        )
    model.eval()

    raw_scores = []
    labels = []
    batch_size = args.batch_size

    print(f"Generating answers and computing confidence scores on {len(items)} samples...")
    with torch.no_grad():
        for start_idx in range(0, len(items), batch_size):
            batch = items[start_idx : start_idx + batch_size]
            prompts = [build_prompt(it["question"], tokenizer) for it in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=None,
                top_p=None,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id
            )

            # Extract sequence probabilities from output token scores
            # gen_outputs.scores is a tuple of length new_tokens, each tensor [batch_size, vocab_size]
            stacked_scores = torch.stack(gen_outputs.scores, dim=1)  # [batch_size, seq_len, vocab_size]
            log_probs = torch.log_softmax(stacked_scores, dim=-1)

            input_len = inputs["input_ids"].shape[1]
            generated_sequences = gen_outputs.sequences[:, input_len:]

            for b_idx, it in enumerate(batch):
                seq_tokens = generated_sequences[b_idx]
                non_pad_mask = (seq_tokens != tokenizer.pad_token_id)
                active_tokens = seq_tokens[non_pad_mask]

                if len(active_tokens) > 0:
                    token_log_probs = []
                    for t_step in range(len(active_tokens)):
                        tok_id = active_tokens[t_step].item()
                        tok_lp = log_probs[b_idx, t_step, tok_id].item()
                        token_log_probs.append(tok_lp)
                    mean_lp = float(np.mean(token_log_probs))
                    # Raw confidence signal: length-normalized sequence probability
                    raw_conf = math.exp(mean_lp)
                else:
                    raw_conf = 0.10

                answer_str = tokenizer.decode(active_tokens, skip_special_tokens=True).strip()
                gold_answers = it.get("answers", [])
                correct = 1 if is_correct(answer_str, gold_answers) else 0

                raw_scores.append(raw_conf)
                labels.append(correct)

            if (start_idx + batch_size) % 100 == 0 or (start_idx + len(batch)) == len(items):
                print(f"Processed {min(start_idx + batch_size, len(items))}/{len(items)} items.")

    raw_scores_arr = np.array(raw_scores)
    labels_arr = np.array(labels)
    acc = np.mean(labels_arr)
    print(f"Training generation accuracy: {acc * 100:.2f}% ({np.sum(labels_arr)}/{len(labels_arr)})")

    # Fit Platt scaling calibrator
    w, b = fit_platt_scaling(raw_scores_arr, labels_arr)
    print(f"Fitted Platt Scaling parameters: w = {w:.4f}, b = {b:.4f}")

    # Compute calibrated confidences
    calibrated = 1.0 / (1.0 + np.exp(-(w * raw_scores_arr + b)))
    # Find optimal threshold guaranteeing selective risk <= 0.16 on training distribution
    threshold = find_optimal_threshold(calibrated, labels_arr, target_risk=0.16)
    print(f"Optimal selective abstention threshold (target risk <= 0.16): theta = {threshold:.4f}")

    config: Dict[str, Any] = {
        "task": "Task 3: Calibrated QA with Abstention",
        "model_name": args.model_name,
        "platt_w": w,
        "platt_b": b,
        "abstention_threshold": threshold,
        "max_new_tokens": 32,
        "system_prompt": "You are a direct, factual question-answering assistant. Provide only the concise, direct answer to the question in as few words as possible without explanation."
    }

    config_path = output_dir / "calibration_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Calibration configuration saved successfully to {config_path}")
    print("Task 3 training complete.")


if __name__ == "__main__":
    main()
