"""QLoRA SFT training script for Turkish Legal RAG.

Trains a LoRA adapter on top of 4-bit quantized Qwen2.5-3B-Instruct
using the SFT data prepared by prepare_sft_data.py.

Designed for 8 GB VRAM (RTX 3070 Laptop GPU).

Usage:
  set PYTHONUTF8=1
  python scripts/train_sft_qlora.py
  python scripts/train_sft_qlora.py --config configs/sft_config.yaml
  python scripts/train_sft_qlora.py --resume outputs/sft_qlora/checkpoint-100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sft_dataset(path: str, fmt: str = "chat") -> Dataset:
    """Load JSONL SFT data and return a HuggingFace Dataset."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if fmt == "chat":
        # Each row has {"messages": [{"role":..., "content":...}, ...]}
        return Dataset.from_list([{"messages": r["messages"]} for r in rows])
    elif fmt == "sharegpt":
        # Each row has {"system":..., "conversations": [...]}
        converted = []
        for r in rows:
            msgs = [{"role": "system", "content": r["system"]}]
            for turn in r["conversations"]:
                role = "user" if turn["from"] == "human" else "assistant"
                msgs.append({"role": role, "content": turn["value"]})
            converted.append({"messages": msgs})
        return Dataset.from_list(converted)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def formatting_prompts_func(examples, tokenizer):
    """Apply chat template to each example's messages."""
    outputs = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        outputs.append(text)
    return {"text": outputs}


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT training")
    parser.add_argument("--config", default="configs/sft_config.yaml")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    # -- Tokenizer --
    print(f"[INFO] Loading tokenizer: {model_cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # -- Quantization config --
    compute_dtype = getattr(torch, model_cfg.get("bnb_4bit_compute_dtype", "float16"))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # -- Model --
    print(f"[INFO] Loading model: {model_cfg['name']} (4-bit)")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )

    # -- LoRA config (passed to SFTTrainer, not applied manually) --
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    # -- Dataset --
    print(f"[INFO] Loading SFT data: {data_cfg['train_path']}")
    dataset = load_sft_dataset(data_cfg["train_path"], fmt=data_cfg.get("format", "chat"))
    print(f"  {len(dataset)} training examples")

    # Check token lengths on a sample
    sample_texts = []
    for msgs in dataset["messages"][:5]:
        t = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        sample_texts.append(t)
    lengths = [len(tokenizer.encode(t)) for t in sample_texts]
    print(f"  Sample token lengths (first 5): {lengths}")

    # -- SFT config (TRL 1.4 API) --
    output_dir = train_cfg["output_dir"]
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        max_length=train_cfg["max_seq_length"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        fp16=train_cfg.get("fp16", True),
        bf16=train_cfg.get("bf16", False),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        seed=train_cfg.get("seed", 42),
        report_to="none",
        max_grad_norm=0.3,
        completion_only_loss=True,
        dataset_text_field="messages",
    )

    # -- Trainer --
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # -- Train --
    print(f"[INFO] Starting training for {train_cfg['num_train_epochs']} epochs")
    print(f"  Effective batch size: {train_cfg['per_device_train_batch_size']} * {train_cfg['gradient_accumulation_steps']} = {train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']}")

    resume_path = args.resume
    trainer.train(resume_from_checkpoint=resume_path)

    # -- Save --
    final_dir = Path(output_dir) / "final"
    print(f"[INFO] Saving final adapter to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("[DONE] Training complete!")


if __name__ == "__main__":
    main()
