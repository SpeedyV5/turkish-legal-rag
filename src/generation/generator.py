from __future__ import annotations

from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"YAML config is empty or invalid: {path}")
    return data


class LocalGenerator:
    def __init__(self, config_path: str = "configs/generation_config.yaml") -> None:
        self.cfg = load_yaml(config_path)

        self.model_name = self.cfg["model"]["name"]
        self.load_in_4bit = bool(self.cfg["model"]["load_in_4bit"])

        self.max_new_tokens = int(self.cfg["generation"]["max_new_tokens"])
        self.temperature = float(self.cfg["generation"]["temperature"])
        self.top_p = float(self.cfg["generation"]["top_p"])
        self.do_sample = bool(self.cfg["generation"]["do_sample"])
        self.repetition_penalty = float(self.cfg["generation"]["repetition_penalty"])

        self.system_prompt = self.cfg["prompting"]["system_prompt"]

        print(f"[INFO] Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[INFO] Loading model: {self.model_name}")
        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
        }

        if self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["dtype"] = torch.float16
        else:
            model_kwargs["dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

    def generate(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "repetition_penalty": self.repetition_penalty,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }

            if self.do_sample:
                generate_kwargs["temperature"] = self.temperature
                generate_kwargs["top_p"] = self.top_p

            output_ids = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return output_text.strip()