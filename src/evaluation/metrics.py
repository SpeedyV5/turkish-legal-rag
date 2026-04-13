from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        rel = 1.0 if rid in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    return 1.0 if any(rid in relevant_ids for rid in top_k) else 0.0


def normalize_article_ref(ref: str | None) -> str | None:
    if not ref:
        return None
    match = re.search(r"(\d+)", ref)
    if match:
        return match.group(1)
    return None


def build_relevant_chunk_ids(
    retrieved_chunks: list[dict[str, Any]],
    gold_doc_ids: list[str],
    gold_articles: list[str],
) -> set[str]:
    """Identify which retrieved chunks match the gold relevant articles."""
    gold_article_nums = set()
    for art in gold_articles:
        num = normalize_article_ref(art)
        if num:
            gold_article_nums.add(num)

    gold_doc_set = set(gold_doc_ids)
    relevant = set()

    for chunk in retrieved_chunks:
        doc_id = chunk.get("doc_id", "")
        article_ref = chunk.get("article_ref", "")

        if doc_id not in gold_doc_set:
            continue

        chunk_art_num = normalize_article_ref(article_ref)
        if chunk_art_num and chunk_art_num in gold_article_nums:
            relevant.add(chunk["chunk_id"])

    return relevant


def f1_token(prediction: str, reference: str) -> float:
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    if not common:
        return 0.0

    p = len(common) / len(pred_tokens)
    r = len(common) / len(ref_tokens)

    return 2 * p * r / (p + r)


def exact_match(prediction: str, reference: str) -> float:
    def normalize(text: str) -> str:
        text = re.sub(r"[^\w\s]", "", text.lower())
        return " ".join(text.split())

    return 1.0 if normalize(prediction) == normalize(reference) else 0.0


def compute_retrieval_metrics(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results: dict[str, float] = {}
    results["mrr"] = mrr(retrieved_chunk_ids, relevant_chunk_ids)

    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)
        results[f"precision@{k}"] = precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)
        results[f"hit@{k}"] = hit_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)

    return results


def aggregate_metrics(all_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not all_metrics:
        return {}

    aggregated: dict[str, float] = defaultdict(float)
    for metrics in all_metrics:
        for key, value in metrics.items():
            aggregated[key] += value

    n = len(all_metrics)
    return {key: round(value / n, 4) for key, value in sorted(aggregated.items())}
