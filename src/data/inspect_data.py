from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import yaml


def load_config(config_path: str = "configs/data_config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def safe_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(str(value))


def guess_text_columns(df: pd.DataFrame) -> list[str]:
    candidate_cols = []
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(20).tolist()
        if not sample:
            continue
        avg_len = sum(len(x) for x in sample) / len(sample)
        if avg_len > 15:
            candidate_cols.append(col)
    return candidate_cols


def build_report_for_split(df: pd.DataFrame, split_name: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "split_name": split_name,
        "num_rows": int(len(df)),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    text_columns = guess_text_columns(df)
    report["candidate_text_columns"] = text_columns

    text_stats = {}
    for col in text_columns:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue
        lengths = series.map(len).tolist()
        text_stats[col] = {
            "min_len": min(lengths),
            "max_len": max(lengths),
            "avg_len": round(mean(lengths), 2),
            "empty_string_count": int((series.str.strip() == "").sum()),
            "top_5_most_common_values": Counter(series).most_common(5),
        }

    report["text_stats"] = text_stats

    preview_rows = df.head(3).to_dict(orient="records")
    report["preview_rows"] = preview_rows

    return report


def save_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
    config = load_config()
    raw_dir = Path(config["datasets"]["primary_hf_dataset"]["save_dir"])
    reports_dir = Path(config["output"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No JSONL files found in {raw_dir}. Run download_data.py first."
        )

    for jsonl_file in jsonl_files:
        split_name = jsonl_file.stem
        print(f"[INFO] Inspecting {jsonl_file}")
        rows = load_jsonl(jsonl_file)
        df = pd.DataFrame(rows)

        report = build_report_for_split(df, split_name)
        output_file = reports_dir / f"{split_name}_report.json"
        save_report(report, output_file)

        print(f"[INFO] Report saved to {output_file}")
        print(f"[INFO] Rows: {report['num_rows']}")
        print(f"[INFO] Columns: {report['columns']}")
        print(f"[INFO] Candidate text columns: {report['candidate_text_columns']}")
        print("-" * 60)


if __name__ == "__main__":
    main()