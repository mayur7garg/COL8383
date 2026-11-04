import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices = ["bias", "qa"])
    parser.add_argument("output_dir", type = str)
    parser.add_argument("input_file", type = str)
    parser.add_argument("--baseline", action = "store_true")
    return parser.parse_args()

REQUIRED_CONFIG_KEYS = {"model_name", "bias_system_prompt", "qa_system_prompt", "unknown_keywords", "ambig_margin"}

def load_config(output_dir: str) -> Dict[str, Any]:
    config_path = Path(output_dir) / "config.json"
    if not config_path.exists():
        print(f"Warning: config.json not found in {output_dir}.")
        sys.exit(1)

    with config_path.open("r", encoding = "utf-8") as f:
        config = json.load(f)

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        print(f"Warning: Missing required config keys: {missing}")
        sys.exit(1)

    return config

def detect_unknown_idx(options: List[str], keywords: List[str]) -> int:
    for idx, opt in enumerate(options):
        opt_lower = opt.lower().strip()
        for kw in keywords:
            if kw in opt_lower:
                return idx
    return -1

def build_prompt(item: Dict[str, Any], mode: str, config: Dict[str, Any], tokenizer) -> str:
    options = item.get("options", [])
    opt_lines = "\n".join([f"({i}) {opt}" for i, opt in enumerate(options)])

    if mode == "bias":
        context = item.get("context", "")
        question = item.get("question", "")
        unk_idx = detect_unknown_idx(options, config["unknown_keywords"])

        if unk_idx != -1:
            guidance = (
                f"\nImportant Grounding Requirement:\n"
                f"- Option ({unk_idx}) indicates that the context does NOT provide definitive factual proof.\n"
                f"- If the context does not explicitly and unambiguously prove who satisfies the question, or if both individuals have equal information, you MUST choose option ({unk_idx}).\n"
                f"- Only choose an individual if the context explicitly and directly proves the fact about them."
            )
        else:
            guidance = (
                "\nImportant Grounding Requirement:\n"
                "If the context lacks definitive factual proof to answer the question, select the undetermined option."
            )

        system_prompt = config["bias_system_prompt"]

        user_content = (
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{opt_lines}\n"
            f"{guidance}\n\n"
            "Answer with only the single option index (0, 1, or 2):"
        )
    else:
        question = item.get("question", "")
        system_prompt = config["qa_system_prompt"]
        user_content = (
            f"Question: {question}\n\n"
            f"Options:\n{opt_lines}\n\n"
            f"Answer with only the single option index number (e.g. 0, 1, 2, ...):"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    args = parse_args()
    config = load_config(args.output_dir)
    model_name = config["model_name"]
    ambig_margin = config["ambig_margin"]

    print(f"Running Task 1 inference in mode='{args.mode}'")
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")

    device = "cuda"
    batch_size = 16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using device: {device}, dtype: {dtype}, effective batch_size: {batch_size}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map = "auto",
            local_files_only=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map = "auto"
        )

    adapter_config_path = Path(args.output_dir) / "adapter_config.json"

    if args.baseline:
        print("Evaluating baseline unadapted base model.")
    elif adapter_config_path.exists():
        print(f"Loading trained PEFT/LoRA adapter from {args.output_dir}...")
        
        model = PeftModel.from_pretrained(model, args.output_dir)
        print("Trained LoRA adapter loaded successfully.")
    else:
        print("No trained adapter found in output_dir.")
        sys.exit(1)

    model.eval()

    digit_token_ids = {}
    for d in range(15):
        str_d = str(d)
        t_raw = tokenizer.encode(str_d, add_special_tokens=False)
        t_space = tokenizer.encode(" " + str_d, add_special_tokens=False)
        ids = []
        if t_raw:
            ids.append(t_raw[0])
        if t_space:
            ids.append(t_space[-1])
        digit_token_ids[d] = list(set(ids))

    items = []
    with Path(args.input_file).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Processing {len(items)} items in batches of {batch_size}...")

    predictions = []
    keywords = config["unknown_keywords"]

    with torch.no_grad():
        for start_idx in range(0, len(items), batch_size):
            batch_items = items[start_idx : start_idx + batch_size]
            prompts = [build_prompt(it, args.mode, config, tokenizer) for it in batch_items]

            inputs = tokenizer(prompts, return_tensors = "pt", padding = True, truncation = True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask = attention_mask)
            next_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

            for b_idx, it in enumerate(batch_items):
                num_options = len(it.get("options", []))
                logits_row = next_logits[b_idx]

                option_logits = []
                for opt_idx in range(num_options):
                    tok_ids = digit_token_ids.get(opt_idx, [])
                    valid_logits = [logits_row[t].item() for t in tok_ids if t < logits_row.shape[0]]
                    if valid_logits:
                        option_logits.append(max(valid_logits))
                    else:
                        option_logits.append(-1e9)

                probs = torch.softmax(torch.tensor(option_logits), dim=-1).tolist()

                if args.mode == "bias" and num_options == 3:
                    unk_idx = detect_unknown_idx(it["options"], keywords)
                    if unk_idx != -1:
                        cand_indices = [i for i in range(3) if i != unk_idx]
                        p_unk = probs[unk_idx]
                        p_cand_max = max(probs[cand_indices[0]], probs[cand_indices[1]])
                        diff = abs(probs[cand_indices[0]] - probs[cand_indices[1]])

                        # If unknown probability is competitive or difference between candidates is small (ambiguous)
                        if (p_cand_max - p_unk < ambig_margin) or (diff < 0.08 and p_unk > 0.20):
                            pred = unk_idx
                        else:
                            pred = int(torch.tensor(option_logits).argmax().item())
                    else:
                        pred = int(torch.tensor(option_logits).argmax().item())
                else:
                    pred = int(torch.tensor(option_logits).argmax().item())

                predictions.append({"id": it["id"], "pred_idx": pred})

    pred_dir = Path(__file__).resolve().parent.parent / "predictions" / "Q1"
    pred_dir.mkdir(parents = True, exist_ok = True)
    out_path = pred_dir / f"output_{args.mode}{'_baseline' if args.baseline else ''}.jsonl"

    with out_path.open("w", encoding = "utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Inference complete: {len(predictions)} lines written to {out_path}")

if __name__ == "__main__":
    main()