from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"YAML config is empty or invalid: {path}")
    return data


def main() -> None:
    cfg = load_yaml("configs/retrieval_config.yaml")

    embeddings_path = Path(cfg["output"]["embeddings_path"])
    faiss_index_path = Path(cfg["output"]["faiss_index_path"])

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    embeddings = np.load(embeddings_path)

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(faiss_index_path))

    print(f"[INFO] Built FAISS index with {index.ntotal} vectors")
    print(f"[INFO] Vector dimension: {dim}")
    print(f"[INFO] Saved index to: {faiss_index_path}")


if __name__ == "__main__":
    main()