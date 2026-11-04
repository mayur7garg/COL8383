import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type = str)
    parser.add_argument("output_dir", type = str)
    parser.add_argument("seed", type = int)
    parser.add_argument("condition", nargs = "?", default = "all", choices = ["all", "clean", "short", "substitution", "paraphrase"])
    return parser.parse_args()

def load_generated_items(output_dir: Path, seed: int) -> List[Dict[str, Any]]:
    path = output_dir / f"generated_{seed}.jsonl"
    if not path.exists():
        print(f"Error: {path} not found.")
        sys.exit(1)
    with path.open("r", encoding = "utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(items)} generated items from {path}")
    return items

def apply_clean(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"id": it["id"], "text": it["text"]} for it in items]

def apply_short(items: List[Dict[str, Any]], tokenizer) -> List[Dict[str, Any]]:
    results = []
    for it in items:
        token_ids = it["token_ids"][:50]
        short_text = tokenizer.decode(token_ids, skip_special_tokens = True)
        results.append({"id": it["id"], "text": short_text})
    return results

def apply_substitution(items: List[Dict[str, Any]], tokenizer, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(seed + 999)
    special_ids = set(tokenizer.all_special_ids)
    non_special_tokens = np.array([t for t in range(len(tokenizer)) if t not in special_ids])
    results = []

    for it in items:
        toks = list(it["token_ids"][:200])
        replace_indices = rng.choice(len(toks), size=min(40, len(toks)), replace=False)
        for idx, new_token in zip(replace_indices, rng.choice(non_special_tokens, size=len(replace_indices))):
            toks[idx] = int(new_token)
        sub_text = tokenizer.decode(toks, skip_special_tokens=True)
        results.append({"id": it["id"], "text": sub_text})
    return results

def apply_paraphrase(items: List[Dict[str, Any]], instruct_model_name: str) -> List[Dict[str, Any]]:
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {instruct_model_name} for paraphrasing on {device}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(instruct_model_name, local_files_only = True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(instruct_model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            instruct_model_name,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            instruct_model_name,
            torch_dtype=dtype,
            device_map="auto"
        )
    model.eval()

    results = []
    batch_size = 8

    with torch.no_grad():
        for start_idx in range(0, len(items), batch_size):
            batch = items[start_idx : start_idx + batch_size]
            prompts = []
            for it in batch:
                messages = [
                    {"role": "system", "content": "You are an expert editor. Rewrite the text to paraphrase it completely while preserving the original meaning. Output only the paraphrased text."},
                    {"role": "user", "content": f"Paraphrase the following text:\n\n{it['text']}"}
                ]
                prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            gen_out = model.generate(
                **inputs,
                max_new_tokens = 220,
                do_sample = True,
                temperature = 0.7,
                top_p = 0.9,
                pad_token_id = tokenizer.pad_token_id
            )

            for b_idx, it in enumerate(batch):
                prompt_len = inputs["input_ids"][b_idx].shape[0]
                new_tokens = gen_out[b_idx][prompt_len:]
                para_text = tokenizer.decode(new_tokens, skip_special_tokens=  True).strip()
                results.append({"id": it["id"], "text": para_text})

            if (start_idx + batch_size) % 50 == 0 or (start_idx + len(batch)) == len(items):
                print(f"Paraphrased {min(start_idx + batch_size, len(items))}/{len(items)} items.")

    return results

def write_condition_file(output_dir: Path, condition: str, seed: int, records: List[Dict[str, Any]]):
    out_file = output_dir / f"generated_{condition}_{seed}.jsonl"
    with out_file.open("w", encoding = "utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} perturbed items to {out_file}")

def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    config_path = output_dir / "watermark_config.json"

    if not config_path.exists():
        print(f"Error: watermark_config.json not found in {output_dir}")
        sys.exit(1)

    with config_path.open("r", encoding = "utf-8") as f:
        config = json.load(f)

    items = load_generated_items(output_dir, args.seed)
    base_model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B")

    try:
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=True)
    except Exception:
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    conditions_to_run = ["clean", "short", "substitution", "paraphrase"] if args.condition == "all" else [args.condition]

    for cond in conditions_to_run:
        print(f"Applying condition: '{cond}'...")
        if cond == "clean":
            records = apply_clean(items)
        elif cond == "short":
            records = apply_short(items, base_tokenizer)
        elif cond == "substitution":
            records = apply_substitution(items, base_tokenizer, args.seed)
        elif cond == "paraphrase":
            records = apply_paraphrase(items, "Qwen/Qwen2.5-1.5B-Instruct")
        else:
            continue

        write_condition_file(output_dir, cond, args.seed, records)

    print("Perturbations completed successfully.")

if __name__ == "__main__":
    main()