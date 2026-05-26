"""Print a clean comparison table from outputs/evaluation/comparison_report.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


SYSTEMS = [
    "baseline_dense", "bm25_only", "hybrid",
    "dense_reranked", "hybrid_reranked",
    "dense_reranked_ml", "hybrid_reranked_ml",
    "e5large_dense", "e5large_reranked_ml",
    "e5large_reranked_bge",
]


def main() -> None:
    path = Path("outputs/evaluation/comparison_report.json")
    report = json.loads(path.read_text(encoding="utf-8"))

    short = {
        "baseline_dense": "base",
        "bm25_only": "bm25",
        "hybrid": "hyb",
        "dense_reranked": "d+EN",
        "hybrid_reranked": "h+EN",
        "dense_reranked_ml": "d+ML",
        "hybrid_reranked_ml": "h+ML",
        "e5large_dense": "L",
        "e5large_reranked_ml": "L+ML",
        "e5large_reranked_bge": "L+BGE",
    }

    header = f"{'metric':<14}" + "".join(f"{short[s]:>8}" for s in SYSTEMS)
    print(header)
    print("-" * len(header))
    for row in report["comparison_table"]:
        line = f"{row['metric']:<14}"
        for s in SYSTEMS:
            v = row.get(s, "-")
            if isinstance(v, float):
                line += f"{v:>8.4f}"
            else:
                line += f"{str(v):>8}"
        print(line)


if __name__ == "__main__":
    main()
