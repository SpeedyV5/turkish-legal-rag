from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from rank_bm25 import BM25Okapi


TURKISH_STOP_WORDS = {
    "bir", "ve", "bu", "da", "de", "ile", "için", "olan", "veya", "ya",
    "ise", "den", "dan", "dir", "dır", "olarak", "gibi", "kadar", "sonra",
    "önce", "her", "o", "ne", "mi", "mu", "mü", "mı", "ki", "ama",
    "ancak", "fakat", "çünkü", "eğer", "ya", "hem", "dahi", "üzere",
    "karşı", "göre", "ayrıca", "şekilde", "dolayı", "ait", "ilgili",
    "olan", "olup", "olan", "eder", "edilir", "yapılır", "verilir",
}


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


def tokenize_turkish(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in TURKISH_STOP_WORDS and len(t) > 1]


class BM25Retriever:
    def __init__(self, config_path: str = "configs/retrieval_config.yaml") -> None:
        cfg = load_yaml(config_path)

        metadata_path = Path(cfg["output"]["metadata_path"])
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        print("[INFO] Loading chunk metadata for BM25...")
        self.metadata = load_jsonl(metadata_path)

        print(f"[INFO] Building BM25 index over {len(self.metadata)} chunks...")
        corpus_tokens = [tokenize_turkish(m["text"]) for m in self.metadata]
        self.bm25 = BM25Okapi(corpus_tokens)

        self.default_top_k = int(cfg["retrieval"]["top_k"])
        print("[INFO] BM25 index ready.")

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        k = top_k if top_k is not None else self.default_top_k

        query_tokens = tokenize_turkish(query)
        scores = self.bm25.get_scores(query_tokens)

        top_indices = scores.argsort()[::-1][:k]

        results = []
        for idx in top_indices:
            item = dict(self.metadata[idx])
            item["score"] = float(scores[idx])
            item["retrieval_method"] = "bm25"
            results.append(item)

        return results

    def get_scores_all(self, query: str) -> list[tuple[int, float]]:
        query_tokens = tokenize_turkish(query)
        scores = self.bm25.get_scores(query_tokens)
        return [(i, float(s)) for i, s in enumerate(scores)]


def main() -> None:
    retriever = BM25Retriever()

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
            print(f"\n  [{i}] score={item['score']:.4f} | {item['title']} | {item.get('article_ref')}")
            print(f"      {item['text'][:200]}...")


if __name__ == "__main__":
    main()
