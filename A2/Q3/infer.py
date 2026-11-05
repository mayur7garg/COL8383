import argparse
import json
import re
import string
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

SYSTEM_PROMPT = (
    "You are a direct, factual question-answering assistant. Provide only the concise, direct answer to the question in as few words as possible without any explanation."
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type = str)
    parser.add_argument("output_dir", type = str)
    return parser.parse_args()

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding = "utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def normalize_answer(ans: str) -> str:
    ans = ans.lower().translate(str.maketrans("", "", string.punctuation))
    ans = re.sub(r"\b(a|an|the)\b", " ", ans)
    return " ".join(ans.split())

def contains(ans: str, sub: str) -> bool:
    t, s = normalize_answer(ans).split(), normalize_answer(sub).split()
    return bool(s) and any(t[i:i + len(s)] == s for i in range(len(t) - len(s) + 1))

def load_model(model_name: str):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only = True)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype =dtype, device_map="auto", local_files_only = True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype =dtype, device_map="auto")
    tokenizer.padding_side = "left"
    return tokenizer, model.eval()

def build_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\nAnswer:"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)


def clean_answer(text: str) -> str:
    text = text.strip()
    return text[7:].strip() if text.lower().startswith("answer:") else text


@torch.no_grad()
def answer_with_features(items: List[Dict[str, Any]], tokenizer, model, cfg: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
    k = cfg["num_samples"]
    common = dict(
        max_new_tokens = cfg["max_new_tokens"],
        repetition_penalty = 1.0,
        top_k = None, top_p = None,
        pad_token_id = tokenizer.pad_token_id
    )
    answers, features = [], []
    bs = cfg["batch_size"]

    for start in range(0, len(items), bs):
        batch = items[start : start + bs]
        prompts = [build_prompt(it["question"], tokenizer) for it in batch]
        inputs = tokenizer(prompts, return_tensors = "pt", padding = True).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        greedy = model.generate(
            **inputs,
            do_sample = False,
            temperature = None,
            output_logits = True,
            return_dict_in_generate = True,
            **common
        )
        seqs = greedy.sequences[:, prompt_len:]
        token_lp = torch.stack([
            torch.log_softmax(step.float(), dim=-1).gather(1, seqs[:, i:i + 1]).squeeze(1) for i, step in enumerate(greedy.logits)
            ],
            dim=1
        )
        valid = (seqs != tokenizer.pad_token_id)

        sampled = None
        if k >0:
            out = model.generate(**inputs, do_sample = True, temperature = cfg["temperature"], num_return_sequences  = k, **common)
            sampled = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens = True)

        for b in range(len(batch)):
            lp = token_lp[b][valid[b]]
            ans = clean_answer(tokenizer.decode(seqs[b], skip_special_tokens = True))
            feats = [lp.mean().item(), lp.min().item()] if len(lp) else [-10.0, -10.0]

            if sampled is not None:
                samples = [clean_answer(s) for s in sampled[b * k : (b + 1) * k]]
                feats.append(float(np.mean([contains(s, ans) or contains(ans, s) for s in samples])))
            answers.append(ans)
            features.append(feats)

        print(f"Processed {min(start + bs, len(items))}/{len(items)} items.", end="\r", flush=True)
    return answers, np.array(features, dtype=np.float64)

def confidence(features: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    logits = features @ np.array(cfg["coef"]) + cfg["intercept"]
    return 1.0 / (1.0 + np.exp(-logits))

def main():
    args = parse_args()
    input_path = Path(args.input_file).resolve()
    output_dir = Path(args.output_dir).resolve()

    config_path = output_dir / "calibration_config.json"
    if not config_path.exists():
        print(f"Error: calibration_config.json not found in {output_dir}")
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    set_seed(config["seed"])

    items = read_jsonl(input_path)
    print(f"Running inference on {len(items)} items from {input_path}...")
    tokenizer, model = load_model(config["model_name"])
    answers, features = answer_with_features(items, tokenizer, model, config)
    confs = confidence(features, config)

    target_path = output_dir / f"pred_{input_path.name}"
    with target_path.open("w", encoding="utf-8") as f:
        for item, ans, conf in zip(items, answers, confs):
            f.write(json.dumps({
                "id": item["id"],
                "answer": ans,
                "confidence": float(conf),
                "abstain": bool(conf < config["abstain_threshold"])
            }) + "\n")

    print(f"Predictions written to {target_path}")

if __name__ == "__main__":
    main()