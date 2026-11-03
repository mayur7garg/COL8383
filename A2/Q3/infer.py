#!/usr/bin/env python3
"""
Task 3 Inference: Calibrated Question Answering with Abstention
Generates concise answers, calibrated confidences in [0, 1], and optimal abstention decisions.
Writes <output_dir>/pred_<input_filename>
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Task 3 Calibrated QA Inference")
    parser.add_argument("input_file", type=str, help="Path to input questions JSONL file")
    parser.add_argument("output_dir", type=str, help="Output directory where calibration_config.json resides")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    return parser.parse_args()


def load_config(output_dir: Path) -> Dict[str, Any]:
    config_path = output_dir / "calibration_config.json"
    if not config_path.exists():
        print(f"Warning: calibration_config.json not found in {output_dir}. Using robust defaults.")
        return {
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "platt_w": 2.5,
            "platt_b": -0.8,
            "abstention_threshold": 0.45,
            "max_new_tokens": 32,
            "system_prompt": "You are a direct, factual question-answering assistant. Provide only the concise, direct answer to the question in as few words as possible without explanation."
        }
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(question: str, system_prompt: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\nAnswer:"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    args = parse_args()
    input_path = Path(args.input_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(output_dir)
    model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    platt_w = float(config.get("platt_w", 2.5))
    platt_b = float(config.get("platt_b", -0.8))
    threshold = float(config.get("abstention_threshold", 0.45))
    system_prompt = config.get("system_prompt", "You are a direct, factual question-answering assistant. Provide only the concise, direct answer to the question in as few words as possible without explanation.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if torch.cuda.is_available() else torch.float32)

    print(f"Running Task 3 inference on: {input_path.name}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Platt scaling parameters: w={platt_w:.4f}, b={platt_b:.4f} | Abstain threshold: {threshold:.4f}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto"
        )
    model.eval()

    items = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Processing {len(items)} questions in batches of {args.batch_size}...")
    batch_size = args.batch_size
    predictions: List[Dict[str, Any]] = []

    with torch.no_grad():
        for start_idx in range(0, len(items), batch_size):
            batch = items[start_idx : start_idx + batch_size]
            prompts = [build_prompt(it["question"], system_prompt, tokenizer) for it in batch]
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

            # Stack generation step scores: [batch_size, seq_len, vocab_size]
            stacked_scores = torch.stack(gen_outputs.scores, dim=1)
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
                    raw_conf = math.exp(mean_lp)
                else:
                    raw_conf = 0.10

                # Calibrated probability in [0, 1] via Platt scaling
                logit_val = platt_w * raw_conf + platt_b
                calibrated_conf = 1.0 / (1.0 + math.exp(-max(min(logit_val, 15.0), -15.0)))
                # Clip confidence to [0.01, 0.99]
                calibrated_conf = max(0.01, min(0.99, calibrated_conf))

                answer_str = tokenizer.decode(active_tokens, skip_special_tokens=True).strip()
                # Clean any lingering prompt labels
                if answer_str.lower().startswith("answer:"):
                    answer_str = answer_str[7:].strip()

                abstain_decision = bool(calibrated_conf < threshold)

                predictions.append({
                    "id": it["id"],
                    "answer": answer_str,
                    "confidence": round(calibrated_conf, 4),
                    "abstain": abstain_decision
                })

            if (start_idx + batch_size) % 100 == 0 or (start_idx + len(batch)) == len(items):
                print(f"Completed {min(start_idx + batch_size, len(items))}/{len(items)} items.")

    # Target filename: pred_<input_filename> as per PDF Page 9
    out_filename = f"pred_{input_path.name}"
    target_path = output_dir / out_filename

    with target_path.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    # Also mirror to predictions/Q3/
    pred_q3_dir = Path(__file__).resolve().parent.parent / "predictions" / "Q3"
    pred_q3_dir.mkdir(parents=True, exist_ok=True)
    mirror_path = pred_q3_dir / out_filename
    with mirror_path.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(f"Task 3 inference complete: {len(predictions)} lines written to {target_path}")


if __name__ == "__main__":
    main()
