from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"YAML config is empty or invalid: {path}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    cfg = load_yaml("configs/retrieval_config.yaml")

    model_name = cfg["embedding"]["model_name"]
    passage_prefix = cfg["embedding"]["passage_prefix"]
    batch_size = int(cfg["embedding"]["batch_size"])
    normalize_embeddings = bool(cfg["embedding"]["normalize_embeddings"])

    chunks_path = Path(cfg["input"]["chunks_path"])
    embeddings_path = Path(cfg["output"]["embeddings_path"])
    metadata_path = Path(cfg["output"]["metadata_path"])

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    rows = load_jsonl(chunks_path)
    if not rows:
        raise ValueError("No chunks found in chunks file.")

    texts = [passage_prefix + row["text"] for row in rows]

    print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[INFO] Encoding {len(texts)} chunks")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
    )

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)

    metadata_rows = []
    for row in rows:
        metadata_rows.append(
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "title": row["title"],
                "doc_type": row["doc_type"],
                "source_family": row["source_family"],
                "source_name": row["source_name"],
                "url": row["url"],
                "article_ref": row["article_ref"],
                "subchunk_index": row["subchunk_index"],
                "text": row["text"],
                "text_len": row["text_len"],
            }
        )

    save_jsonl(metadata_rows, metadata_path)

    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    print(f"[INFO] Saved embeddings to: {embeddings_path}")
    print(f"[INFO] Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()