from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset, DatasetDict, Dataset


def load_config(config_path: str = "configs/data_config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_split_as_jsonl(split: Dataset, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row in split:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    config = load_config()
    dataset_cfg = config["datasets"]["primary_hf_dataset"]

    dataset_name = dataset_cfg["name"]
    save_dir = Path(dataset_cfg["save_dir"])
    ensure_dir(save_dir)

    print(f"[INFO] Downloading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    if isinstance(ds, DatasetDict):
        print(f"[INFO] Found splits: {list(ds.keys())}")
        for split_name, split_data in ds.items():
            output_file = save_dir / f"{split_name}.jsonl"
            save_split_as_jsonl(split_data, output_file)
            print(f"[INFO] Saved split '{split_name}' to {output_file}")
    elif isinstance(ds, Dataset):
        output_file = save_dir / "full.jsonl"
        save_split_as_jsonl(ds, output_file)
        print(f"[INFO] Saved dataset to {output_file}")
    else:
        raise TypeError(f"Unexpected dataset type: {type(ds)}")

    print("[INFO] Download completed successfully.")


if __name__ == "__main__":
    main()