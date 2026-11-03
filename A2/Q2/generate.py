#!/usr/bin/env python3
"""
Task 2 Watermarked Text Generation
Generates watermarked continuations on M-base (Qwen/Qwen2.5-1.5B) under strict multinomial sampling.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 Watermark Generation")
    parser.add_argument("input_file", type=str, help="Path to input prompts JSONL file")
    parser.add_argument("output_dir", type=str, help="Output directory where watermark_config.json was saved")
    parser.add_argument("seed", type=int, help="Random seed for generation")
    return parser.parse_args()


def load_config(output_dir: Path) -> Dict[str, Any]:
    config_path = output_dir / "watermark_config.json"
    if not config_path.exists():
        print(f"Error: watermark_config.json not found in {output_dir}. Run setup.sh first.")
        sys.exit(1)
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    input_file = Path(args.input_file).resolve()
    config = load_config(output_dir)

    model_name = config["model_name"]
    secret_key = config["secret_key"]
    modulus = config["modulus"]
    hash_multiplier = config["hash_multiplier"]
    gamma_bi = config["gamma_bi"]
    gamma_uni = config["gamma_uni"]
    delta_bi = config["delta_bi"]
    delta_uni = config["delta_uni"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if torch.cuda.is_available() else torch.float32)

    print(f"Generating watermarked text with seed={args.seed}")
    print(f"Input: {input_file} | Output dir: {output_dir}")
    print(f"Device: {device}, dtype: {dtype}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    # Pre-build tensor of token hashes on target device for fast vectorization
    token_hashes_tensor = torch.tensor(config["token_hashes"], dtype=torch.int64, device=device)
    threshold_bi = int(gamma_bi * modulus)
    threshold_uni = int(gamma_uni * modulus)

    # Precompute unigram green mask (static across all steps)
    unigram_hashes = (token_hashes_tensor + secret_key) % modulus
    unigram_green_mask = (unigram_hashes < threshold_uni)

    items: List[Dict[str, Any]] = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Loaded {len(items)} prompts to process.")
    results: List[Dict[str, Any]] = []

    # Generator for reproducible multinomial sampling
    sampling_gen = torch.Generator(device=device).manual_seed(args.seed)

    with torch.no_grad():
        for idx, item in enumerate(items):
            prompt_text = item["prompt"]
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            # Initial forward pass to compute prompt KV cache
            outputs = model(input_tensor, use_cache=True)
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :].clone().squeeze(0)  # Shape: [vocab_size]

            prev_token_id = prompt_ids[-1] if prompt_ids else tokenizer.eos_token_id
            generated_token_ids: List[int] = []

            for step in range(200):
                # 1. Bigram green mask computation: H_bi(v) = (T_v + T_{w_prev} * MULT + secret_key) % MOD
                prev_hash = token_hashes_tensor[prev_token_id]
                bigram_hashes = (token_hashes_tensor + prev_hash * hash_multiplier + secret_key) % modulus
                bigram_green_mask = (bigram_hashes < threshold_bi)

                # 2. Apply watermark biases to logits
                biased_logits = next_logits.clone()
                biased_logits[bigram_green_mask] += delta_bi
                biased_logits[unigram_green_mask] += delta_uni

                # 3. Pure multinomial sampling at temperature 1.0 (no top-k, no top-p)
                probs = torch.softmax(biased_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=sampling_gen).item()

                generated_token_ids.append(next_token)
                prev_token_id = next_token

                # 4. Step forward 1 token using KV cache if not at the 200th token
                if step < 199:
                    step_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
                    step_out = model(step_input, past_key_values=past_key_values, use_cache=True)
                    past_key_values = step_out.past_key_values
                    next_logits = step_out.logits[:, -1, :].squeeze(0)

            assert len(generated_token_ids) == 200, f"Expected 200 tokens, got {len(generated_token_ids)}"
            decoded_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

            results.append({
                "id": item["id"],
                "text": decoded_text,
                "token_ids": generated_token_ids
            })

            if (idx + 1) % 50 == 0 or (idx + 1) == len(items):
                print(f"Generated {idx + 1}/{len(items)} prompts.")

    output_file = output_dir / f"generated_{args.seed}.jsonl"
    with output_file.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Watermark generation complete: 200 tokens each written to {output_file}")


if __name__ == "__main__":
    main()
