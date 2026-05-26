"""End-to-end QA evaluation: retrieval -> generation -> metrics.

Pipeline:
1. Load benchmark split (default: dev).
2. Build retriever (default: e5large_reranked_bge).
3. For each question: retrieve top_k contexts.
4. Free retriever memory (optional), then load LLM.
5. For each question: build prompt, generate answer.
6. Compute EM, F1, citation precision/recall/F1, lexical faithfulness.
7. Save per-question + aggregated results to outputs/qa_eval/.

Designed to be GPU-friendly on a 4 GB card: retrieval done first in one
pass, then LLM is loaded (4-bit quantized) and generation done in a
second pass.

Usage examples:
  python -m src.evaluation.run_qa_eval --split dev --system e5large_reranked_bge
  python -m src.evaluation.run_qa_eval --split dev --max-questions 10
  python -m src.evaluation.run_qa_eval --split test --output-tag final
  python -m src.evaluation.run_qa_eval --benchmark path/to/custom_benchmark.jsonl --output-tag custom
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from src.evaluation.qa_metrics import (
    aggregate_qa_metrics,
    citation_metrics,
    exact_match,
    extract_cited_articles,
    faithfulness_lexical,
    overlap_generation_metrics,
    token_f1,
)
from src.evaluation.run_retrieval_eval import create_retriever, load_benchmark


SPLIT_FILES = {
    "full": "data/benchmark/gold_benchmark.jsonl",
    "train": "data/benchmark/gold_benchmark_train.jsonl",
    "dev": "data/benchmark/gold_benchmark_dev.jsonl",
    "test": "data/benchmark/gold_benchmark_test.jsonl",
}


def stage1_retrieve(
    retriever,
    benchmark: list[dict],
    top_k: int,
) -> dict[str, list[dict]]:
    print(f"[INFO] Stage 1: retrieving top-{top_k} for {len(benchmark)} questions")
    out: dict[str, list[dict]] = {}
    for i, q in enumerate(benchmark):
        try:
            results = retriever.search(q["question"], top_k=top_k)
        except Exception as e:
            print(f"  [ERROR] retrieve failed for {q['id']}: {e}")
            results = []
        out[q["id"]] = results
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(benchmark)}] retrieved")
    return out


def free_retriever(retriever) -> None:
    """Best-effort GPU memory cleanup before loading the LLM."""
    try:
        import torch
        del retriever
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def stage2_generate(
    benchmark: list[dict],
    retrieved: dict[str, list[dict]],
    config_path: str,
    lora_adapter_path: str | None = None,
) -> list[dict[str, Any]]:
    """Load LLM and generate answers for every question."""
    from src.generation.generator import LocalGenerator
    from src.generation.prompt_builder import build_user_prompt, ensure_dayanak

    print(f"[INFO] Stage 2: loading LLM from {config_path}")
    if lora_adapter_path:
        print(f"[INFO] Using LoRA adapter: {lora_adapter_path}")
    gen = LocalGenerator(config_path=config_path, lora_adapter_path=lora_adapter_path)

    rows: list[dict[str, Any]] = []
    print(f"[INFO] Generating answers for {len(benchmark)} questions...")
    start = time.time()
    for i, q in enumerate(benchmark):
        contexts = retrieved.get(q["id"], [])
        prompt = build_user_prompt(q["question"], contexts[:5])
        t0 = time.time()
        try:
            answer = gen.generate(prompt)
            answer = ensure_dayanak(answer, contexts[:5])
        except Exception as e:
            answer = f"[GENERATION_ERROR] {e}"
        gen_seconds = time.time() - t0

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold_articles": q["relevant_articles"],
            "gold_doc_ids": q["relevant_doc_ids"],
            "question_type": q["question_type"],
            "difficulty": q["difficulty"],
            "source_law": q["source_law"],
            "retrieved_top5": [
                {
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "article_ref": c.get("article_ref"),
                    "score": c.get("score"),
                    "text": c.get("text", "")[:400],
                }
                for c in contexts[:5]
            ],
            "answer": answer,
            "gen_seconds": round(gen_seconds, 2),
        })
        elapsed = time.time() - start
        avg = elapsed / (i + 1)
        remaining = avg * (len(benchmark) - i - 1)
        print(f"  [{i + 1}/{len(benchmark)}] {q['id']}: gen={gen_seconds:.1f}s "
              f"(avg={avg:.1f}s, ETA={remaining/60:.1f}m)")
    return rows


def compute_metrics_for_rows(rows: list[dict[str, Any]]) -> tuple[list[dict], dict]:
    print("[INFO] Computing QA metrics...")
    per_q: list[dict[str, Any]] = []
    metric_dicts: list[dict[str, float]] = []

    for r in rows:
        ans = r["answer"]
        ref = r["expected_answer"]
        contexts = [c["text"] for c in r["retrieved_top5"]]

        em = exact_match(ans, ref)
        f1 = token_f1(ans, ref)
        cite = extract_cited_articles(ans)
        cite_metrics = citation_metrics(cite["articles"], r["gold_articles"])
        faith = faithfulness_lexical(ans, contexts)
        overlap = overlap_generation_metrics(ans, ref)

        m: dict[str, float] = {"em": em}
        m.update({f"answer_{k}": v for k, v in f1.items()})
        m.update(cite_metrics)
        m.update(faith)
        m.update(overlap)
        m["has_dayanak"] = 1.0 if cite["has_dayanak"] else 0.0

        record = dict(r)
        record["cited_articles"] = cite["articles"]
        record["has_dayanak"] = cite["has_dayanak"]
        record["metrics"] = m
        per_q.append(record)
        metric_dicts.append(m)

    aggregated = aggregate_qa_metrics(metric_dicts)

    # Breakdown by type / difficulty / law
    from collections import defaultdict
    by_type: dict[str, list[dict]] = defaultdict(list)
    by_diff: dict[str, list[dict]] = defaultdict(list)
    by_law: dict[str, list[dict]] = defaultdict(list)
    for pq in per_q:
        by_type[pq["question_type"]].append(pq["metrics"])
        by_diff[pq["difficulty"]].append(pq["metrics"])
        by_law[pq["source_law"]].append(pq["metrics"])

    breakdown = {
        "by_question_type": {k: aggregate_qa_metrics(v) for k, v in by_type.items()},
        "by_difficulty": {k: aggregate_qa_metrics(v) for k, v in by_diff.items()},
        "by_source_law": {k: aggregate_qa_metrics(v) for k, v in by_law.items()},
    }

    return per_q, {"aggregated": aggregated, "breakdown": breakdown}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system", default="e5large_reranked_bge",
        choices=[
            "baseline_dense", "bm25_only", "hybrid",
            "hybrid_reranked", "dense_reranked",
            "dense_reranked_ml", "hybrid_reranked_ml",
            "e5large_dense", "e5large_reranked_ml",
            "e5large_reranked_bge",
        ],
    )
    parser.add_argument(
        "--split", default="dev", choices=list(SPLIT_FILES.keys()),
    )
    parser.add_argument(
        "--benchmark", default=None,
        help="Optional JSONL benchmark path. Overrides --split when provided.",
    )
    parser.add_argument(
        "--retrieval-config", default="configs/retrieval_config_e5large.yaml",
    )
    parser.add_argument(
        "--generation-config", default="configs/generation_config.yaml",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-questions", type=int, default=0,
                        help="If > 0, only run on first N questions.")
    parser.add_argument("--lora-adapter", default=None,
                        help="Path to LoRA adapter dir for fine-tuned model.")
    parser.add_argument("--output-dir", default="outputs/qa_eval")
    parser.add_argument("--output-tag", default="",
                        help="Suffix for the output filename.")
    args = parser.parse_args()

    benchmark_path = args.benchmark if args.benchmark else SPLIT_FILES[args.split]
    split_name = "custom" if args.benchmark else args.split
    print(f"[INFO] Split: {split_name} ({benchmark_path})")
    benchmark = load_benchmark(benchmark_path)
    if args.max_questions > 0:
        benchmark = benchmark[: args.max_questions]
    print(f"[INFO] Evaluating on {len(benchmark)} questions")

    # Stage 1: retrieval
    retriever = create_retriever(args.system, config_path=args.retrieval_config)
    retrieved = stage1_retrieve(retriever, benchmark, top_k=args.top_k)
    free_retriever(retriever)

    # Stage 2: generation
    rows = stage2_generate(benchmark, retrieved, args.generation_config,
                           lora_adapter_path=args.lora_adapter)

    # Stage 3: metrics
    per_q, summary = compute_metrics_for_rows(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    out_path = output_dir / f"qa_eval_{split_name}_{args.system}{suffix}.json"

    payload = {
        "config": {
            "system": args.system,
            "split": split_name,
            "benchmark_path": benchmark_path,
            "retrieval_config": args.retrieval_config,
            "generation_config": args.generation_config,
            "top_k": args.top_k,
            "n_questions": len(benchmark),
        },
        "aggregated": summary["aggregated"],
        "breakdown": summary["breakdown"],
        "per_question": per_q,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Pretty print
    print("\n" + "=" * 60)
    print(f"QA RESULTS: {args.system} | split={split_name} | n={len(benchmark)}")
    print("=" * 60)
    agg = summary["aggregated"]
    keys_priority = [
        "em", "answer_f1", "answer_precision", "answer_recall",
        "bleu1", "bleu2", "rouge_l_f1",
        "citation_f1", "citation_precision", "citation_recall", "citation_exact",
        "has_dayanak", "faithfulness_lexical",
    ]
    for k in keys_priority:
        if k in agg:
            print(f"  {k:<22} {agg[k]:.4f}")
    print(f"\n[INFO] Saved to: {out_path}")


if __name__ == "__main__":
    main()
