from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


EVAL_DIR = Path("outputs/evaluation")

SYSTEM_ORDER = [
    "baseline_dense", "bm25_only", "hybrid",
    "dense_reranked", "hybrid_reranked",
    "dense_reranked_ml", "hybrid_reranked_ml",
    "e5large_dense", "e5large_reranked_ml", "e5large_reranked_bge",
]

KEY_METRICS = ["mrr", "recall@1", "recall@3", "recall@5", "recall@10", "ndcg@5", "ndcg@10", "hit@5"]


def load_eval_results(system: str) -> dict[str, Any] | None:
    path = EVAL_DIR / f"eval_{system}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison_table(all_results: dict[str, dict]) -> list[dict]:
    rows = []
    for metric in KEY_METRICS:
        row = {"metric": metric}
        for system in SYSTEM_ORDER:
            if system in all_results:
                agg = all_results[system].get("aggregated_metrics", {})
                row[system] = agg.get(metric, "-")
            else:
                row[system] = "-"
        rows.append(row)
    return rows


def build_breakdown_comparison(
    all_results: dict[str, dict],
    breakdown_key: str,
    metric: str = "recall@5",
) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}

    for system, results in all_results.items():
        bd = results.get("breakdown", {}).get(breakdown_key, {})
        for category, metrics in bd.items():
            if category not in comparison:
                comparison[category] = {}
            comparison[category][system] = metrics.get(metric, 0.0)

    return comparison


def find_improvements(all_results: dict[str, dict]) -> list[dict]:
    improvements = []

    if "baseline_dense" not in all_results:
        return improvements

    baseline_per_q = {
        pqr["id"]: pqr
        for pqr in all_results["baseline_dense"].get("per_question_results", [])
    }

    compare_systems = [s for s in SYSTEM_ORDER if s != "baseline_dense"]
    for system in compare_systems:
        if system not in all_results:
            continue

        for pqr in all_results[system].get("per_question_results", []):
            qid = pqr["id"]
            base = baseline_per_q.get(qid)
            if not base:
                continue

            base_recall5 = base["metrics"].get("recall@5", 0)
            new_recall5 = pqr["metrics"].get("recall@5", 0)

            if new_recall5 > base_recall5:
                improvements.append({
                    "id": qid,
                    "question": pqr["question"][:80],
                    "system": system,
                    "baseline_recall@5": base_recall5,
                    "improved_recall@5": new_recall5,
                })

    return improvements


def find_regressions(all_results: dict[str, dict]) -> list[dict]:
    regressions = []

    if "baseline_dense" not in all_results:
        return regressions

    baseline_per_q = {
        pqr["id"]: pqr
        for pqr in all_results["baseline_dense"].get("per_question_results", [])
    }

    compare_systems = [s for s in SYSTEM_ORDER if s != "baseline_dense"]
    for system in compare_systems:
        if system not in all_results:
            continue

        for pqr in all_results[system].get("per_question_results", []):
            qid = pqr["id"]
            base = baseline_per_q.get(qid)
            if not base:
                continue

            base_recall5 = base["metrics"].get("recall@5", 0)
            new_recall5 = pqr["metrics"].get("recall@5", 0)

            if new_recall5 < base_recall5:
                regressions.append({
                    "id": qid,
                    "question": pqr["question"][:80],
                    "system": system,
                    "baseline_recall@5": base_recall5,
                    "regressed_recall@5": new_recall5,
                })

    return regressions


def main() -> None:
    print("[INFO] Loading evaluation results...")

    all_results: dict[str, dict] = {}
    for system in SYSTEM_ORDER:
        result = load_eval_results(system)
        if result:
            all_results[system] = result
            print(f"  [OK] {system}: loaded")
        else:
            print(f"  [SKIP] {system}: no results found")

    if not all_results:
        print("[ERROR] No evaluation results found. Run evaluations first.")
        return

    comparison_table = build_comparison_table(all_results)

    print(f"\n{'=' * 90}")
    print("RETRIEVAL SYSTEM COMPARISON")
    print(f"{'=' * 90}")

    header = f"{'Metric':<16}"
    for system in SYSTEM_ORDER:
        if system in all_results:
            header += f"{system:<20}"
    print(header)
    print("-" * 90)

    for row in comparison_table:
        line = f"{row['metric']:<16}"
        for system in SYSTEM_ORDER:
            if system in all_results:
                val = row.get(system, "-")
                if isinstance(val, float):
                    line += f"{val:<20.4f}"
                else:
                    line += f"{str(val):<20}"
        print(line)

    print(f"\n{'=' * 90}")
    print("RECALL@5 BY QUESTION TYPE")
    print(f"{'=' * 90}")
    by_type = build_breakdown_comparison(all_results, "by_question_type", "recall@5")
    for category, scores in sorted(by_type.items()):
        line = f"  {category:<16}"
        for system in SYSTEM_ORDER:
            if system in scores:
                line += f"{scores[system]:<20.4f}"
        print(line)

    print(f"\n{'=' * 90}")
    print("RECALL@5 BY DIFFICULTY")
    print(f"{'=' * 90}")
    by_diff = build_breakdown_comparison(all_results, "by_difficulty", "recall@5")
    for category, scores in sorted(by_diff.items()):
        line = f"  {category:<16}"
        for system in SYSTEM_ORDER:
            if system in scores:
                line += f"{scores[system]:<20.4f}"
        print(line)

    print(f"\n{'=' * 90}")
    print("RECALL@5 BY SOURCE LAW")
    print(f"{'=' * 90}")
    by_law = build_breakdown_comparison(all_results, "by_source_law", "recall@5")
    for category, scores in sorted(by_law.items()):
        line = f"  {category[:30]:<32}"
        for system in SYSTEM_ORDER:
            if system in scores:
                line += f"{scores[system]:<20.4f}"
        print(line)

    improvements = find_improvements(all_results)
    regressions = find_regressions(all_results)

    if improvements:
        print(f"\n{'=' * 90}")
        print(f"IMPROVEMENTS vs BASELINE ({len(improvements)} cases)")
        print(f"{'=' * 90}")
        for imp in improvements[:20]:
            print(
                f"  [{imp['id']}] {imp['system']}: "
                f"{imp['baseline_recall@5']:.2f} -> {imp['improved_recall@5']:.2f} | "
                f"{imp['question']}"
            )

    if regressions:
        print(f"\n{'=' * 90}")
        print(f"REGRESSIONS vs BASELINE ({len(regressions)} cases)")
        print(f"{'=' * 90}")
        for reg in regressions[:20]:
            print(
                f"  [{reg['id']}] {reg['system']}: "
                f"{reg['baseline_recall@5']:.2f} -> {reg['regressed_recall@5']:.2f} | "
                f"{reg['question']}"
            )

    report = {
        "comparison_table": comparison_table,
        "breakdown_by_type": by_type,
        "breakdown_by_difficulty": by_diff,
        "breakdown_by_law": by_law,
        "improvements": improvements,
        "regressions": regressions,
        "systems_evaluated": list(all_results.keys()),
    }

    report_path = EVAL_DIR / "comparison_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Comparison report saved to: {report_path}")


if __name__ == "__main__":
    main()
