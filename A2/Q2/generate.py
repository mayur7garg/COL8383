import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type = str)
    parser.add_argument("output_dir", type = str)
    parser.add_argument("seed", type = int)
    parser.add_argument("--batch_size", type = int, default = 16)
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
            torch_dtype = dtype,
            device_map = "auto",
            local_files_only = True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype = dtype,
            device_map = "auto"
        )
    model.eval()

    token_hashes_tensor = torch.tensor(config["token_hashes"], dtype = torch.int64, device = device)
    threshold_bi = int(gamma_bi * modulus)
    threshold_uni = int(gamma_uni * modulus)

    unigram_hashes = (token_hashes_tensor + secret_key) % modulus
    unigram_green_mask = (unigram_hashes < threshold_uni)

    items: List[Dict[str, Any]] = []
    with input_file.open("r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Loaded {len(items)} prompts to process.")
    results: List[Dict[str, Any]] = []

    sampling_gen = torch.Generator(device=device).manual_seed(args.seed)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            prompts = [tokenizer.encode(it["prompt"], add_special_tokens = False) or [tokenizer.eos_token_id] for it in batch]
            max_len = max(len(p) for p in prompts)
            input_ids = torch.tensor([[pad_id] * (max_len - len(p)) + p for p in prompts], device=device)
            attention_mask = torch.tensor([[0] * (max_len - len(p)) + [1] * len(p) for p in prompts], device=device)
            position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

            outputs = model(
                input_ids = input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache = True
            )
            prev_tokens = input_ids[:, -1]
            generated = []

            for step in range(200):
                # Bigram green list
                prev_hash = token_hashes_tensor[prev_tokens].unsqueeze(-1)
                bigram_green_mask = (token_hashes_tensor.unsqueeze(0) + prev_hash * hash_multiplier + secret_key) % modulus < threshold_bi

                logits = outputs.logits[:, -1, :].float()
                logits = logits + delta_bi * bigram_green_mask + delta_uni * unigram_green_mask

                # Multinomial sampling at temperature 1
                next_tokens = torch.multinomial(torch.softmax(logits, dim = -1), num_samples = 1, generator = sampling_gen)
                generated.append(next_tokens)
                prev_tokens = next_tokens.squeeze(-1)

                if step < 199:
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens)], dim=-1)
                    position_ids = position_ids[:, -1:] + 1
                    outputs = model(
                        input_ids = next_tokens,
                        attention_mask = attention_mask,
                        position_ids = position_ids,
                        past_key_values = outputs.past_key_values, 
                        use_cache = True
                    )

            for it, token_ids in zip(batch, torch.cat(generated, dim=1).tolist()):
                results.append({
                    "id": it["id"],
                    "text": tokenizer.decode(token_ids, skip_special_tokens=True),
                    "token_ids": token_ids
                })
            print(f"Generated {len(results)}/{len(items)} prompts.")

    output_file = output_dir / f"generated_{args.seed}.jsonl"
    with output_file.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Watermark generation complete: 200 tokens each written to {output_file}")

if __name__ == "__main__":
    main()