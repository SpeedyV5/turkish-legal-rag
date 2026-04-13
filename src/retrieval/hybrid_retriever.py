from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from src.retrieval.bm25_retriever import BM25Retriever, tokenize_turkish


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


class DenseRetriever:
    """Pure dense retriever without any hardcoded keyword bonuses."""

    def __init__(self, config_path: str = "configs/retrieval_config.yaml") -> None:
        cfg = load_yaml(config_path)

        self.model_name = cfg["embedding"]["model_name"]
        self.query_prefix = cfg["embedding"]["query_prefix"]
        self.normalize = bool(cfg["embedding"]["normalize_embeddings"])
        self.default_top_k = int(cfg["retrieval"]["top_k"])

        index_path = Path(cfg["output"]["faiss_index_path"])
        metadata_path = Path(cfg["output"]["metadata_path"])

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        print(f"[INFO] Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        print(f"[INFO] Loading FAISS index: {index_path}")
        self.index = faiss.read_index(str(index_path))

        self.metadata = load_jsonl(metadata_path)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index ({self.index.ntotal}) and metadata ({len(self.metadata)}) size mismatch."
            )

    def encode_query(self, query: str) -> np.ndarray:
        text = self.query_prefix + query.strip()
        emb = self.model.encode(
            [text], normalize_embeddings=self.normalize, convert_to_numpy=True,
        )
        return emb.astype(np.float32)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        k = top_k if top_k is not None else self.default_top_k
        query_vec = self.encode_query(query)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            item["retrieval_method"] = "dense"
            results.append(item)

        return results

    def search_with_scores(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_vec = self.encode_query(query)
        scores, indices = self.index.search(query_vec, top_k)
        return [
            (int(idx), float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]


class HybridRetriever:
    """Combines dense and BM25 retrieval using reciprocal rank fusion or weighted scores."""

    def __init__(
        self,
        config_path: str = "configs/retrieval_config.yaml",
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
    ) -> None:
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k

        print("[INFO] Initializing HybridRetriever...")
        self.dense = DenseRetriever(config_path)
        self.bm25 = BM25Retriever(config_path)
        self.metadata = self.dense.metadata

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        k = top_k if top_k is not None else self.dense.default_top_k
        candidate_k = max(k * 5, 30)

        dense_results = self.dense.search(query, top_k=candidate_k)
        bm25_results = self.bm25.search(query, top_k=candidate_k)

        if self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(dense_results, bm25_results, k)
        else:
            return self._weighted_score_fusion(dense_results, bm25_results, k)

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        top_k: int,
    ) -> list[dict[str, Any]]:
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        for rank, item in enumerate(dense_results):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0)
            rrf_scores[cid] += self.dense_weight / (self.rrf_k + rank + 1)
            chunk_map[cid] = item

        for rank, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0)
            rrf_scores[cid] += self.bm25_weight / (self.rrf_k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = item

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        results = []
        for cid in sorted_ids:
            item = dict(chunk_map[cid])
            item["score"] = rrf_scores[cid]
            item["retrieval_method"] = "hybrid_rrf"
            results.append(item)

        return results

    def _weighted_score_fusion(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        top_k: int,
    ) -> list[dict[str, Any]]:
        def min_max_normalize(items: list[dict]) -> list[dict]:
            if not items:
                return items
            scores = [it["score"] for it in items]
            lo, hi = min(scores), max(scores)
            rng = hi - lo if hi != lo else 1.0
            for it in items:
                it["norm_score"] = (it["score"] - lo) / rng
            return items

        dense_results = min_max_normalize(dense_results)
        bm25_results = min_max_normalize(bm25_results)

        combined: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        for item in dense_results:
            cid = item["chunk_id"]
            combined[cid] = self.dense_weight * item["norm_score"]
            chunk_map[cid] = item

        for item in bm25_results:
            cid = item["chunk_id"]
            combined[cid] = combined.get(cid, 0.0) + self.bm25_weight * item["norm_score"]
            if cid not in chunk_map:
                chunk_map[cid] = item

        sorted_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]

        results = []
        for cid in sorted_ids:
            item = dict(chunk_map[cid])
            item["score"] = combined[cid]
            item["retrieval_method"] = "hybrid_weighted"
            results.append(item)

        return results


def main() -> None:
    retriever = HybridRetriever(fusion_method="rrf")

    sample_queries = [
        "Cumhuriyetin nitelikleri nelerdir?",
        "Kasten öldürme suçunun cezası nedir?",
        "Boşanma sebepleri nelerdir?",
        "Tutuklama koşulları nelerdir?",
    ]

    for q in sample_queries:
        print(f"\n{'#' * 80}")
        print(f"QUERY: {q}")
        print("#" * 80)
        results = retriever.search(q, top_k=5)
        for i, item in enumerate(results, 1):
            print(
                f"  [{i}] score={item['score']:.4f} | {item['title']} "
                f"| {item.get('article_ref')} | method={item['retrieval_method']}"
            )


if __name__ == "__main__":
    main()
