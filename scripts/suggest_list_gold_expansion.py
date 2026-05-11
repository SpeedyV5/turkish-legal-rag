"""For every list-type question, suggest additional relevant articles
that may also answer it. Uses the best retriever (e5-large + ML reranker)
to surface candidates inside the gold document; produces a human-review
file. We DO NOT auto-modify the gold benchmark.

Output:
- data/benchmark/list_gold_expansion_candidates.jsonl

Each row contains: id, question, current gold articles, candidate
extras with text snippets to inspect, and a `suggested_extra_articles`
field that picks high-scoring same-doc articles ABOVE a threshold.

Usage:
  python -m scripts.suggest_list_gold_expansion
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


THRESHOLD = 0.3  # rerank score threshold for "suggested" articles
TOP_K = 15

INPUT = Path("data/benchmark/gold_benchmark.jsonl")
OUTPUT = Path("data/benchmark/list_gold_expansion_candidates.jsonl")


def normalize_article_num(ref: str | None) -> str | None:
    if not ref:
        return None
    m = re.search(r"(\d+)", ref)
    return m.group(1) if m else None


def main() -> None:
    rows = [json.loads(l) for l in INPUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    list_qs = [r for r in rows if r["question_type"] == "list"]
    print(f"[INFO] {len(list_qs)} list-type questions out of {len(rows)} total")

    # Best retriever: e5-large + multilingual reranker
    from src.retrieval.hybrid_retriever import DenseRetriever
    from src.retrieval.reranker import RerankedRetriever

    base = DenseRetriever("configs/retrieval_config_e5large.yaml")
    retriever = RerankedRetriever(
        base, reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )

    candidates_out: list[dict] = []
    for r in list_qs:
        results = retriever.search(r["question"], top_k=TOP_K)

        gold_docs = set(r["relevant_doc_ids"])
        gold_arts = {normalize_article_num(a) for a in r["relevant_articles"]}
        gold_arts.discard(None)

        # Group by article ref, dedup
        seen: set[tuple[str, str]] = set()
        cand_list: list[dict] = []
        for c in results:
            doc = c.get("doc_id", "")
            art_num = normalize_article_num(c.get("article_ref", ""))
            if not art_num or doc not in gold_docs:
                continue
            key = (doc, art_num)
            if key in seen:
                continue
            seen.add(key)
            already_gold = art_num in gold_arts
            cand_list.append({
                "doc_id": doc,
                "article_ref": c.get("article_ref"),
                "article_num": art_num,
                "score": float(c.get("score", 0.0)),
                "already_in_gold": already_gold,
                "text_preview": (c.get("text", "") or "")[:240],
            })

        suggested = [
            c["article_ref"] for c in cand_list
            if (not c["already_in_gold"]) and c["score"] >= THRESHOLD
        ]

        candidates_out.append({
            "id": r["id"],
            "question": r["question"],
            "source_law": r["source_law"],
            "current_gold_articles": r["relevant_articles"],
            "suggested_extra_articles": suggested,
            "candidates": cand_list,
        })

        print(f"  [{r['id']}] suggested extras: {suggested}")

    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in candidates_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_with_suggestions = sum(1 for c in candidates_out if c["suggested_extra_articles"])
    print(f"\n[OK] Wrote {len(candidates_out)} rows -> {OUTPUT}")
    print(f"[OK] {n_with_suggestions} questions have suggested expansions to review")


if __name__ == "__main__":
    main()
