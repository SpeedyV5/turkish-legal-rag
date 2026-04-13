from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.metrics import (
    aggregate_metrics,
    build_relevant_chunk_ids,
    compute_retrieval_metrics,
)


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_benchmark(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def create_retriever(system_name: str, config_path: str = "configs/retrieval_config.yaml"):
    if system_name == "baseline_dense":
        from src.retrieval.hybrid_retriever import DenseRetriever
        return DenseRetriever(config_path)

    elif system_name == "bm25_only":
        from src.retrieval.bm25_retriever import BM25Retriever
        return BM25Retriever(config_path)

    elif system_name == "hybrid":
        from src.retrieval.hybrid_retriever import HybridRetriever
        return HybridRetriever(
            config_path, dense_weight=0.85, bm25_weight=0.15, fusion_method="rrf",
        )

    elif system_name == "hybrid_reranked":
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.reranker import RerankedRetriever
        base = HybridRetriever(
            config_path, dense_weight=0.85, bm25_weight=0.15, fusion_method="rrf",
        )
        return RerankedRetriever(base)

    elif system_name == "dense_reranked":
        from src.retrieval.hybrid_retriever import DenseRetriever
        from src.retrieval.reranker import RerankedRetriever
        base = DenseRetriever(config_path)
        return RerankedRetriever(base)

    elif system_name == "dense_reranked_ml":
        from src.retrieval.hybrid_retriever import DenseRetriever
        from src.retrieval.reranker import RerankedRetriever
        base = DenseRetriever(config_path)
        return RerankedRetriever(
            base, reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        )

    elif system_name == "hybrid_reranked_ml":
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.reranker import RerankedRetriever
        base = HybridRetriever(
            config_path, dense_weight=0.85, bm25_weight=0.15, fusion_method="rrf",
        )
        return RerankedRetriever(
            base, reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        )

    elif system_name == "e5large_dense":
        from src.retrieval.hybrid_retriever import DenseRetriever
        return DenseRetriever(config_path)

    elif system_name == "e5large_reranked_ml":
        from src.retrieval.hybrid_retriever import DenseRetriever
        from src.retrieval.reranker import RerankedRetriever
        base = DenseRetriever(config_path)
        return RerankedRetriever(
            base, reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        )

    else:
        raise ValueError(f"Unknown system: {system_name}")


def evaluate_retrieval(
    retriever,
    benchmark: list[dict],
    k_values: list[int] | None = None,
    top_k_search: int = 10,
) -> dict[str, Any]:
    if k_values is None:
        k_values = [1, 3, 5, 10]

    all_metrics: list[dict[str, float]] = []
    per_question_results: list[dict] = []
    error_cases: list[dict] = []

    total = len(benchmark)

    for i, q in enumerate(benchmark):
        qid = q["id"]
        question = q["question"]

        try:
            results = retriever.search(question, top_k=top_k_search)
        except Exception as e:
            print(f"  [ERROR] {qid}: {e}")
            error_cases.append({"id": qid, "error": str(e)})
            continue

        relevant_ids = build_relevant_chunk_ids(
            results,
            q["relevant_doc_ids"],
            q["relevant_articles"],
        )

        retrieved_ids = [r["chunk_id"] for r in results]

        metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids, k_values)
        all_metrics.append(metrics)

        retrieved_articles = [
            {"chunk_id": r["chunk_id"], "doc_id": r["doc_id"],
             "article_ref": r.get("article_ref"), "score": r["score"]}
            for r in results[:5]
        ]

        per_question_results.append({
            "id": qid,
            "question": question,
            "question_type": q["question_type"],
            "difficulty": q["difficulty"],
            "source_law": q["source_law"],
            "gold_articles": q["relevant_articles"],
            "gold_doc_ids": q["relevant_doc_ids"],
            "retrieved_top5": retrieved_articles,
            "num_relevant_found": len(relevant_ids),
            "metrics": metrics,
        })

        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{total}] processed...")

    aggregated = aggregate_metrics(all_metrics)

    by_type: dict[str, list[dict]] = {}
    by_difficulty: dict[str, list[dict]] = {}
    by_law: dict[str, list[dict]] = {}

    for pqr in per_question_results:
        by_type.setdefault(pqr["question_type"], []).append(pqr["metrics"])
        by_difficulty.setdefault(pqr["difficulty"], []).append(pqr["metrics"])
        by_law.setdefault(pqr["source_law"], []).append(pqr["metrics"])

    breakdown = {
        "by_question_type": {k: aggregate_metrics(v) for k, v in by_type.items()},
        "by_difficulty": {k: aggregate_metrics(v) for k, v in by_difficulty.items()},
        "by_source_law": {k: aggregate_metrics(v) for k, v in by_law.items()},
    }

    zero_recall_cases = [
        {"id": pqr["id"], "question": pqr["question"],
         "gold_articles": pqr["gold_articles"], "retrieved_top5": pqr["retrieved_top5"]}
        for pqr in per_question_results
        if pqr["metrics"].get("recall@5", 0) == 0
    ]

    return {
        "aggregated_metrics": aggregated,
        "breakdown": breakdown,
        "per_question_results": per_question_results,
        "error_cases": error_cases,
        "zero_recall_at_5": zero_recall_cases,
        "num_questions": total,
        "num_evaluated": len(all_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval system on gold benchmark")
    parser.add_argument(
        "--system",
        type=str,
        default="baseline_dense",
        choices=[
            "baseline_dense", "bm25_only", "hybrid",
            "hybrid_reranked", "dense_reranked",
            "dense_reranked_ml", "hybrid_reranked_ml",
            "e5large_dense", "e5large_reranked_ml",
        ],
    )
    parser.add_argument("--benchmark", type=str, default="data/benchmark/gold_benchmark.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    parser.add_argument("--config", type=str, default="configs/retrieval_config.yaml")
    args = parser.parse_args()

    print(f"[INFO] System: {args.system}")
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Benchmark: {args.benchmark}")

    benchmark = load_benchmark(args.benchmark)
    print(f"[INFO] Loaded {len(benchmark)} questions")

    print(f"[INFO] Initializing retriever: {args.system}...")
    retriever = create_retriever(args.system, config_path=args.config)

    print("[INFO] Running evaluation...")
    start = time.time()
    results = evaluate_retrieval(retriever, benchmark)
    elapsed = time.time() - start

    results["system"] = args.system
    results["elapsed_seconds"] = round(elapsed, 2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_{args.system}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {args.system}")
    print(f"{'=' * 60}")
    print(f"Questions evaluated: {results['num_evaluated']}/{results['num_questions']}")
    print(f"Time: {elapsed:.1f}s")

    agg = results["aggregated_metrics"]
    print(f"\n  MRR:          {agg.get('mrr', 0):.4f}")
    print(f"  Recall@1:     {agg.get('recall@1', 0):.4f}")
    print(f"  Recall@3:     {agg.get('recall@3', 0):.4f}")
    print(f"  Recall@5:     {agg.get('recall@5', 0):.4f}")
    print(f"  Recall@10:    {agg.get('recall@10', 0):.4f}")
    print(f"  nDCG@5:       {agg.get('ndcg@5', 0):.4f}")
    print(f"  nDCG@10:      {agg.get('ndcg@10', 0):.4f}")
    print(f"  Hit@5:        {agg.get('hit@5', 0):.4f}")

    print(f"\nZero recall@5 cases: {len(results['zero_recall_at_5'])}")
    if results["zero_recall_at_5"]:
        for case in results["zero_recall_at_5"][:10]:
            print(f"  - [{case['id']}] {case['question'][:60]}...")
            print(f"    Gold: {case['gold_articles']}")

    print(f"\n[INFO] Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
