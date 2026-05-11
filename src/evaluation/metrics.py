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


def build_corpus_relevant_index(
    corpus_metadata: list[dict[str, Any]],
) -> dict[tuple[str, str], set[str]]:
    """Pre-index the full corpus by (doc_id, article_number) -> set of chunk_ids.

    This enables corpus-wide gold relevance computation instead of
    only-among-retrieved (which inflates recall).
    """
    index: dict[tuple[str, str], set[str]] = {}
    for chunk in corpus_metadata:
        doc_id = str(chunk.get("doc_id", ""))
        art_num = normalize_article_ref(chunk.get("article_ref", ""))
        if not doc_id or not art_num:
            continue
        key = (doc_id, art_num)
        index.setdefault(key, set()).add(chunk["chunk_id"])
    return index


def build_relevant_chunk_ids_corpus_wide(
    corpus_index: dict[tuple[str, str], set[str]],
    gold_doc_ids: list[str],
    gold_articles: list[str],
) -> set[str]:
    """Build the gold relevant chunk_id set using the FULL corpus index.

    A chunk is relevant iff its (doc_id, article_number) appears in the gold
    cross-product of (gold_doc_ids x gold_articles). Multiple sub-chunks of the
    same article are all considered relevant.
    """
    gold_article_nums = []
    for art in gold_articles:
        num = normalize_article_ref(art)
        if num:
            gold_article_nums.append(num)

    relevant: set[str] = set()
    for doc_id in gold_doc_ids:
        for art_num in gold_article_nums:
            key = (str(doc_id), art_num)
            if key in corpus_index:
                relevant |= corpus_index[key]
    return relevant


def build_relevant_chunk_ids(
    retrieved_chunks: list[dict[str, Any]],
    gold_doc_ids: list[str],
    gold_articles: list[str],
) -> set[str]:
    """DEPRECATED: only-among-retrieved relevance (biased). Kept for backwards
    compatibility. Use build_relevant_chunk_ids_corpus_wide instead.
    """
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


def compute_article_level_metrics(
    retrieved_chunks: list[dict[str, Any]],
    gold_doc_ids: list[str],
    gold_articles: list[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Article-level retrieval metrics. Treats every sub-chunk of a gold
    article as the same retrieval target (which is what matters for a
    legal QA system: did we surface the right article?).

    - relevance: (doc_id, article_number) tuple
    - recall@k: |unique gold articles retrieved in top-k| / |gold articles|
    - precision@k: counts each unique gold article hit (no double counting)
    - MRR: 1 / (rank of first chunk whose (doc_id, art) is in gold)
    - nDCG@k: graded by article hits, deduplicated per article
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    gold_set: set[tuple[str, str]] = set()
    for d in gold_doc_ids:
        for a in gold_articles:
            num = normalize_article_ref(a)
            if num:
                gold_set.add((str(d), num))

    retrieved_articles: list[tuple[str, str] | None] = []
    for c in retrieved_chunks:
        d = str(c.get("doc_id", ""))
        num = normalize_article_ref(c.get("article_ref", ""))
        retrieved_articles.append((d, num) if d and num else None)

    results: dict[str, float] = {}

    # MRR
    mrr_score = 0.0
    for i, ra in enumerate(retrieved_articles):
        if ra is not None and ra in gold_set:
            mrr_score = 1.0 / (i + 1)
            break
    results["mrr"] = mrr_score

    n_gold = len(gold_set)

    for k in k_values:
        top_k_articles = retrieved_articles[:k]
        unique_hits: set[tuple[str, str]] = {ra for ra in top_k_articles if ra is not None and ra in gold_set}
        n_hits = len(unique_hits)

        results[f"recall@{k}"] = (n_hits / n_gold) if n_gold else 0.0
        results[f"precision@{k}"] = (n_hits / k) if k else 0.0
        results[f"hit@{k}"] = 1.0 if n_hits > 0 else 0.0

        # nDCG: dedup gold-article hits along ranking
        seen_articles: set[tuple[str, str]] = set()
        dcg = 0.0
        for i, ra in enumerate(top_k_articles):
            rel = 0.0
            if ra is not None and ra in gold_set and ra not in seen_articles:
                rel = 1.0
                seen_articles.add(ra)
            dcg += rel / math.log2(i + 2)
        ideal_hits = min(n_gold, k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        results[f"ndcg@{k}"] = (dcg / idcg) if idcg > 0 else 0.0

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
