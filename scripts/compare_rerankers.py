"""A/B comparison: mMiniLMv2 vs bge-reranker-v2-m3 on top of e5-large."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    ml = json.loads(Path("outputs/evaluation/eval_e5large_reranked_ml.json").read_text(encoding="utf-8"))
    bge = json.loads(Path("outputs/evaluation/eval_e5large_reranked_bge.json").read_text(encoding="utf-8"))

    print(f"{'metric':<12}{'mMiniLM':>10}{'bge-m3':>10}{'delta':>10}")
    print("-" * 42)
    for k in ["mrr", "recall@1", "recall@3", "recall@5", "recall@10", "ndcg@5", "ndcg@10", "hit@5"]:
        a = ml["aggregated_metrics"][k]
        b = bge["aggregated_metrics"][k]
        sign = "+" if b - a >= 0 else ""
        print(f"{k:<12}{a:>10.4f}{b:>10.4f}{sign}{b - a:>9.4f}")


if __name__ == "__main__":
    main()
