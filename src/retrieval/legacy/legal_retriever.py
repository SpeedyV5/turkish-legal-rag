"""DEPRECATED LEGACY RETRIEVER.

This is the original Phase-1 retriever which contained hardcoded keyword
bonuses for a handful of seed queries (kasten oldurme, bosanma,
cumhuriyet, "Madde N"). Those bonuses biased its rankings and made it
unsuitable as a "clean" baseline for ablation. We keep it here only for
historical reference. Do NOT use in evaluation or in the pipeline.

The clean replacement is `src.retrieval.hybrid_retriever.DenseRetriever`.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

warnings.warn(
    "src.retrieval.legacy.legal_retriever.LegalRetriever is deprecated. "
    "Use src.retrieval.hybrid_retriever.DenseRetriever instead.",
    DeprecationWarning,
    stacklevel=2,
)


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


class LegalRetriever:
    def __init__(self, config_path: str = "configs/retrieval_config.yaml") -> None:
        self.cfg = load_yaml(config_path)

        self.model_name = self.cfg["embedding"]["model_name"]
        self.query_prefix = self.cfg["embedding"]["query_prefix"]
        self.normalize_embeddings = bool(self.cfg["embedding"]["normalize_embeddings"])
        self.default_top_k = int(self.cfg["retrieval"]["top_k"])

        self.index_path = Path(self.cfg["output"]["faiss_index_path"])
        self.metadata_path = Path(self.cfg["output"]["metadata_path"])

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        print(f"[INFO] Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        print(f"[INFO] Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        print(f"[INFO] Loading metadata from: {self.metadata_path}")
        self.metadata = load_jsonl(self.metadata_path)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index size ({self.index.ntotal}) and metadata size ({len(self.metadata)}) do not match."
            )

    def encode_query(self, query: str) -> np.ndarray:
        text = self.query_prefix + query.strip()
        emb = self.model.encode(
            [text],
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        k = top_k if top_k is not None else self.default_top_k

        # Önce biraz daha fazla aday çekelim
        initial_k = max(k * 3, 10)
        query_vec = self.encode_query(query)

        scores, indices = self.index.search(query_vec, initial_k)

        results = []
        query_lower = query.lower()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            item = dict(self.metadata[idx])
            item["score"] = float(score)

            bonus = 0.0
            text_lower = item["text"].lower()
            article_ref = str(item.get("article_ref", "")).lower()

            # Basit anahtar kelime bonusları
            if "kasten öldürme" in query_lower and "kasten öldürme" in text_lower:
                bonus += 0.04
            if "boşanma" in query_lower and "boşanma" in text_lower:
                bonus += 0.04
            if "cumhuriyetin nitelikleri" in query_lower and "cumhuriyet" in text_lower:
                bonus += 0.03

            # Sorguda madde numarası geçiyorsa bonus ver
            import re
            query_articles = re.findall(r"madde\s+(\d+)", query_lower)
            if query_articles:
                for qa in query_articles:
                    if qa in article_ref:
                        bonus += 0.05

            item["rerank_score"] = item["score"] + bonus
            results.append(item)

        results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return results[:k]


def pretty_print_results(results: list[dict[str, Any]]) -> None:
    if not results:
        print("[INFO] No results found.")
        return

    for i, item in enumerate(results, start=1):
        print("=" * 100)
        print(f"Rank       : {i}")
        print(f"Score      : {item['score']:.4f}")
        print(f"Doc ID     : {item['doc_id']}")
        print(f"Title      : {item['title']}")
        print(f"Doc Type   : {item['doc_type']}")
        print(f"Article Ref: {item.get('article_ref')}")
        print(f"Chunk ID   : {item['chunk_id']}")
        print(f"Text Len   : {item['text_len']}")
        print("-" * 100)
        print(item["text"][:1200])
        print()


def main() -> None:
    retriever = LegalRetriever()

    sample_queries = [
        "Cumhuriyetin nitelikleri nelerdir?",
        "Kasten öldürme suçu nedir?",
        "Boşanma sebepleri nelerdir?",
    ]

    for q in sample_queries:
        print("\n" + "#" * 100)
        print(f"QUERY: {q}")
        print("#" * 100)
        results = retriever.search(q)
        pretty_print_results(results)


if __name__ == "__main__":
    main()