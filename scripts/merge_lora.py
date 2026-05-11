"""Merge LoRA adapter back into base model and save full model.

Usage:
  python scripts/merge_lora.py --adapter outputs/sft_qlora/final --output outputs/sft_qlora/merged
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument("--push-to-hub", default=None, help="Optional HF Hub repo name")
    args = parser.parse_args()

    print(f"[INFO] Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"[INFO] Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("[INFO] Merging adapter...")
    model = model.merge_and_unload()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving merged model to {output_path}")
    model.save_pretrained(str(output_path))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_path))

    if args.push_to_hub:
        print(f"[INFO] Pushing to HF Hub: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)

    print("[DONE] Merge complete!")


if __name__ == "__main__":
    main()
