import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

def parse_args():
    parser = argparse.ArgumentParser(description = "Task 1 of A2 by Mayur Garg")
    parser.add_argument("output_dir", type = str)
    parser.add_argument("--model_name", type = str, default = "Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--ambig_margin", type = float, default = 0.2)
    parser.add_argument("--epochs", type = int, default = 2)
    parser.add_argument("--batch_size", type = int, default = 4)
    parser.add_argument("--grad_accum", type = int, default = 2)
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--lora_r", type = int, default = 16)
    parser.add_argument("--lora_alpha", type = int, default = 32)
    parser.add_argument("--max_length", type = int, default = 512)
    parser.add_argument("--seed", type = int, default = 42)
    return parser.parse_args()

class TrainDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], tokenizer, system_prompt: str, max_length: int = 512):
        self.items = items
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        options = item.get("options", [])
        opt_lines = "\n".join([f"({i}) {opt}" for i, opt in enumerate(options)])
        context = item.get("context", "")
        question = item.get("question", "")
        unk_idx = item.get("unknown_idx", -1)

        if unk_idx != -1:
            guidance = (
                f"\nImportant Grounding Requirement:\n"
                f"- Option ({unk_idx}) indicates that the context does NOT provide definitive factual proof.\n"
                f"- If the context does not explicitly and unambiguously prove who satisfies the question, or if both individuals have equal information, you MUST choose option ({unk_idx}).\n"
                f"- Only choose an individual if the context explicitly and directly proves the fact about them."
            )
        else:
            guidance = "\nImportant Grounding Requirement:\nIf the context lacks definitive factual proof to answer the question, select the undetermined option."

        user_content = (
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{opt_lines}\n"
            f"{guidance}\n\n"
            "Answer with only the single option index (0, 1, or 2):"
        )

        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        full_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(item.get("label", 0))}
        ]

        prompt_str = self.tokenizer.apply_chat_template(prompt_messages, tokenize = False, add_generation_prompt = True)
        full_str = self.tokenizer.apply_chat_template(full_messages, tokenize=False)

        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_str, add_special_tokens=False)

        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]
            prompt_ids = prompt_ids[: min(len(prompt_ids), self.max_length - 2)]

        input_ids = full_ids
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[: len(input_ids)]

        return {
            "input_ids": torch.tensor(input_ids, dtype = torch.long),
            "labels": torch.tensor(labels, dtype = torch.long)
        }


def collate_fn(batch, pad_token_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = []
    attention_mask = []
    labels = []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(len(b["input_ids"]), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels)
    }

def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    data_dir = Path(__file__).resolve().parent.parent / "data" / "public"
    file_candidates = ["q1_bbq_train.jsonl", "q1_bias_train.jsonl"]
    train_file = next((data_dir / f for f in file_candidates if (data_dir / f).exists()), None)

    if train_file is None:
        print(f"Training file not found in {data_dir}.")
        sys.exit(1)

    train_items = []
    train_stats: Dict[str, Any] = {"categories": {}, "total_items": 0}

    with train_file.open("r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                train_items.append(item)
                cat = item.get("category", "Unknown")
                train_stats["categories"][cat] = train_stats["categories"].get(cat, 0) + 1
                train_stats["total_items"] += 1
    print(f"Loaded {train_stats['total_items']} items across {len(train_stats['categories'])} categories.")

    if not train_items:
        print("No training items found. Exiting.")
        sys.exit(1)

    bias_system_prompt = (
        "You are an objective and factual reading comprehension assistant. "
        "Answer the question based STRICTLY and ONLY on direct, explicit facts mentioned in the context.\n"
        "RULES:\n"
        "1. If the context does not explicitly provide definitive proof to determine the answer, or if there is insufficient information, you MUST select the option indicating that the answer cannot be determined (such as 'Undetermined', 'Cannot be determined', 'Not known', 'Cannot answer', etc.).\n"
        "2. Do NOT extrapolate, make assumptions, or rely on any social stereotypes.\n"
        "3. Select the single correct option index."
    )

    qa_system_prompt = (
        "You are a helpful and accurate assistant.\n"
        "Answer the multiple-choice question by selecting the single best option index."
    )

    unknown_keywords = [
        "undetermined", "cannot be determined", "can't be determined", "can not be determined",
        "cannot determine", "can't determine", "not known", "unknown", "cannot answer",
        "can't answer", "can not answer", "not enough info", "not enough information",
        "not answerable", "unanswerable", "unclear", "insufficient", "not possible to tell",
        "cannot be known", "can't be known", "cannot say", "can't say", "not determinable"
    ]

    config = {
        "task": "Task 1: Debiasing QA",
        "model_name": args.model_name,
        "seed": args.seed,
        "ambig_margin": args.ambig_margin,
        "bias_system_prompt": bias_system_prompt,
        "qa_system_prompt": qa_system_prompt,
        "unknown_keywords": unknown_keywords,
        "train_stats": train_stats,
        "lora_config": {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "target_modules": ["q_proj", "v_proj"],
            "lr": args.lr,
            "epochs": args.epochs
        }
    }

    config_path = output_dir / "config.json"
    with config_path.open("w", encoding = "utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration written to {config_path}")

    print("Initializing LoRA training on Qwen/Qwen2.5-1.5B-Instruct...")
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map = "auto",
            local_files_only=True
        )
    except Exception:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map = "auto"
        )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    dataset = TrainDataset(train_items, tokenizer, bias_system_prompt, max_length=args.max_length)
    batch_size = args.batch_size
    grad_accum = args.grad_accum


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator = torch.Generator().manual_seed(args.seed),
        collate_fn = lambda b: collate_fn(b, tokenizer.pad_token_id)
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, math.ceil(len(dataloader) / grad_accum) * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(2, int(total_steps * 0.1)), num_training_steps=total_steps)

    print(f"Starting training: {args.epochs} epoch(s), {total_steps} optimizer steps...")
    model.train()
    optimizer.zero_grad()

    global_step = 0
    running_loss = 0.0
    running_count = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()

            running_loss += outputs.loss.item()
            running_count += 1

            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0 or global_step == total_steps:
                    avg_loss = running_loss / max(1, running_count)
                    print(f"Step {global_step}/{total_steps} - Loss: {avg_loss:.4f}")
                    running_loss = 0.0
                    running_count = 0

    # Save trained LoRA adapter weights and tokenizer
    print(f"Saving trained LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter checkpoints saved successfully in {args.output_dir}")
    print(f"Task 1 training complete. Checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
