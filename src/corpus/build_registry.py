from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml


REGISTRY_COLUMNS = [
    "doc_id",
    "source_family",
    "source_name",
    "doc_type",
    "title",
    "official_no",
    "official_date",
    "url",
    "language",
    "version_status",
    "jurisdiction",
    "text",
    "summary",
    "article_refs",
    "court_chamber",
    "tags",
]


def load_config(config_path: str = "configs/corpus_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    registry_path = Path(config["output"]["registry_path"])
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(columns=REGISTRY_COLUMNS)
    df.to_csv(registry_path, index=False, encoding="utf-8")

    print(f"[INFO] Empty corpus registry created at: {registry_path}")
    print(f"[INFO] Columns: {REGISTRY_COLUMNS}")


if __name__ == "__main__":
    main()