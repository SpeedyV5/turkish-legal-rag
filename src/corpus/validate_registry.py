from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml


REQUIRED_COLUMNS = {
    "doc_id",
    "source_family",
    "source_name",
    "doc_type",
    "title",
    "language",
    "version_status",
    "jurisdiction",
    "text",
}


def load_config(config_path: str = "configs/corpus_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    registry_path = Path(config["output"]["registry_path"])

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    df = pd.read_csv(registry_path)
    cols = set(df.columns)

    missing = REQUIRED_COLUMNS - cols
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("[INFO] Registry schema is valid.")
    print(f"[INFO] Number of rows: {len(df)}")
    print(f"[INFO] Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()