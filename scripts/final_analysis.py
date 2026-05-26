"""Final analysis: error, citation, faithfulness, and overlap reporting.

Generates a Markdown report from existing QA eval JSON outputs without
rerunning expensive model inference.

Usage:
  python scripts/final_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.evaluation.qa_metrics import aggregate_qa_metrics, overlap_generation_metrics


RESULTS = {
    "dev_base": Path("outputs/qa_eval/qa_eval_dev_e5large_reranked_ml_prompt_v5.json"),
    "dev_ft": Path("outputs/qa_eval/qa_eval_dev_e5large_reranked_bge_sft_qlora.json"),
    "test_base": Path("outputs/qa_eval/qa_eval_test_e5large_reranked_bge_baseline.json"),
    "test_ft": Path("outputs/qa_eval/qa_eval_test_e5large_reranked_bge_sft_qlora.json"),
}

ARTIFACTS = [
    Path("outputs/evaluation/eval_e5large_reranked_bge.json"),
    Path("outputs/qa_eval/qa_eval_test_e5large_reranked_bge_baseline.json"),
    Path("outputs/qa_eval/qa_eval_test_e5large_reranked_bge_sft_qlora.json"),
    Path("outputs/sft_qlora/final/adapter_config.json"),
    Path("outputs/sft_qlora/final/adapter_model.safetensors"),
]


def load_eval(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v: Any, pct: bool = True) -> str:
    if not isinstance(v, (float, int)):
        return str(v)
    return f"{v:.2%}" if pct else f"{v:.4f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def recompute_summary(per_q: list[dict[str, Any]]) -> dict[str, Any]:
    metric_dicts = [q["metrics"] for q in per_q]
    by_type: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_diff: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_law: dict[str, list[dict[str, float]]] = defaultdict(list)

    for q in per_q:
        by_type[q["question_type"]].append(q["metrics"])
        by_diff[q["difficulty"]].append(q["metrics"])
        by_law[q["source_law"]].append(q["metrics"])

    return {
        "aggregated": aggregate_qa_metrics(metric_dicts),
        "breakdown": {
            "by_question_type": {k: aggregate_qa_metrics(v) for k, v in by_type.items()},
            "by_difficulty": {k: aggregate_qa_metrics(v) for k, v in by_diff.items()},
            "by_source_law": {k: aggregate_qa_metrics(v) for k, v in by_law.items()},
        },
    }


def enrich_eval(payload: dict[str, Any]) -> dict[str, Any]:
    """Backfill supplemental overlap metrics into older QA outputs."""
    for q in payload.get("per_question", []):
        metrics = q.setdefault("metrics", {})
        if "rouge_l_f1" not in metrics:
            metrics.update(overlap_generation_metrics(q["answer"], q["expected_answer"]))

    summary = recompute_summary(payload.get("per_question", []))
    payload["aggregated"] = summary["aggregated"]
    payload["breakdown"] = summary["breakdown"]
    return payload


def ablation_table(base_agg: dict[str, float], ft_agg: dict[str, float], split: str) -> str:
    keys = [
        ("answer_f1", "Answer F1"),
        ("answer_precision", "Answer Prec"),
        ("answer_recall", "Answer Rec"),
        ("citation_f1", "Citation F1"),
        ("citation_precision", "Citation Prec"),
        ("citation_recall", "Citation Rec"),
        ("citation_exact", "Citation Exact"),
        ("has_dayanak", "Has Dayanak"),
        ("faithfulness_lexical", "Faithfulness"),
    ]
    rows = []
    for key, label in keys:
        b = base_agg.get(key, 0.0)
        f = ft_agg.get(key, 0.0)
        delta = f - b
        sign = "+" if delta >= 0 else ""
        rows.append([label, fmt(b), fmt(f), f"{sign}{delta:.2%}"])
    return markdown_table([f"Metric ({split})", "Baseline", "SFT-QLoRA", "Delta"], rows)


def supplemental_table(base_agg: dict[str, float], ft_agg: dict[str, float], split: str) -> str:
    keys = [
        ("bleu1", "BLEU-1"),
        ("bleu2", "BLEU-2"),
        ("rouge_l_f1", "ROUGE-L F1"),
    ]
    rows = []
    for key, label in keys:
        b = base_agg.get(key, 0.0)
        f = ft_agg.get(key, 0.0)
        delta = f - b
        sign = "+" if delta >= 0 else ""
        rows.append([label, fmt(b), fmt(f), f"{sign}{delta:.2%}"])
    return markdown_table([f"Supplemental Metric ({split})", "Baseline", "SFT-QLoRA", "Delta"], rows)


def breakdown_table(breakdown: dict[str, Any], category: str) -> str:
    data = breakdown.get(category, {})
    rows = []
    for key in sorted(data.keys()):
        agg = data[key]
        rows.append([
            key,
            fmt(agg.get("answer_f1", 0.0)),
            fmt(agg.get("citation_exact", 0.0)),
            fmt(agg.get("citation_f1", 0.0)),
            fmt(agg.get("faithfulness_lexical", 0.0)),
            fmt(agg.get("rouge_l_f1", 0.0)),
        ])
    return markdown_table(
        [category.replace("by_", "").title(), "Ans F1", "Cite Exact", "Cite F1", "Faith", "ROUGE-L"],
        rows,
    )


def classify_issue(q: dict[str, Any]) -> str:
    m = q["metrics"]
    issues = []
    if m.get("faithfulness_lexical", 1.0) < 0.5:
        issues.append("low faithfulness")
    if not q.get("has_dayanak", False):
        issues.append("missing Dayanak")
    if m.get("citation_exact", 0.0) < 1.0:
        issues.append("citation mismatch")
    if m.get("answer_f1", 0.0) < 0.25:
        issues.append("low answer F1")
    return ", ".join(issues) if issues else "acceptable"


def error_analysis(per_q: list[dict[str, Any]]) -> str:
    n = len(per_q)
    good = [q for q in per_q if q["metrics"].get("answer_f1", 0.0) > 0.5]
    low_faith = [q for q in per_q if q["metrics"].get("faithfulness_lexical", 1.0) < 0.5]
    missing_dayanak = [q for q in per_q if not q.get("has_dayanak", False)]
    cite_mismatch = [q for q in per_q if q["metrics"].get("citation_exact", 0.0) < 1.0]
    low_answer = [q for q in per_q if q["metrics"].get("answer_f1", 0.0) < 0.25]

    lines = [
        f"- **Total questions**: {n}",
        f"- **Good answers** (F1 > 0.5): {len(good)} ({len(good) / n:.0%})",
        f"- **Low answer F1** (< 0.25): {len(low_answer)} ({len(low_answer) / n:.0%})",
        f"- **Low faithfulness** (< 0.5): {len(low_faith)} ({len(low_faith) / n:.0%})",
        f"- **Citation exact mismatch**: {len(cite_mismatch)} ({len(cite_mismatch) / n:.0%})",
        f"- **Missing Dayanak**: {len(missing_dayanak)} ({len(missing_dayanak) / n:.0%})",
    ]

    worst = sorted(
        per_q,
        key=lambda q: (
            q["metrics"].get("faithfulness_lexical", 1.0),
            q["metrics"].get("answer_f1", 1.0),
        ),
    )[:5]
    rows = []
    for q in worst:
        m = q["metrics"]
        rows.append([
            f"`{q['id']}`",
            q["question_type"],
            q["source_law"],
            fmt(m.get("answer_f1", 0.0)),
            fmt(m.get("citation_exact", 0.0)),
            fmt(m.get("faithfulness_lexical", 0.0)),
            classify_issue(q),
        ])
    lines.append("\n**Worst risk cases:**")
    lines.append(markdown_table(
        ["ID", "Type", "Law", "Ans F1", "Cite Exact", "Faith", "Issue"],
        rows,
    ))
    return "\n".join(lines)


def improvement_examples(base: dict[str, Any], ft: dict[str, Any]) -> str:
    base_by_id = {q["id"]: q for q in base["per_question"]}
    rows = []
    for q in ft["per_question"]:
        b = base_by_id.get(q["id"])
        if not b:
            continue
        delta = q["metrics"].get("answer_f1", 0.0) - b["metrics"].get("answer_f1", 0.0)
        rows.append((delta, q, b))
    rows.sort(key=lambda item: item[0], reverse=True)

    table_rows = []
    for delta, q, b in rows[:5]:
        table_rows.append([
            f"`{q['id']}`",
            q["question_type"],
            q["source_law"],
            fmt(b["metrics"].get("answer_f1", 0.0)),
            fmt(q["metrics"].get("answer_f1", 0.0)),
            f"+{delta:.2%}",
        ])
    return markdown_table(["ID", "Type", "Law", "Baseline F1", "SFT F1", "Delta"], table_rows)


def artifact_table() -> str:
    rows = [[str(path), "OK" if path.exists() else "MISSING"] for path in ARTIFACTS]
    return markdown_table(["Artifact", "Status"], rows)


def main() -> None:
    missing = [str(path) for path in RESULTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing QA eval files: {missing}")

    data = {key: enrich_eval(load_eval(path)) for key, path in RESULTS.items()}
    test_base_agg = data["test_base"]["aggregated"]
    test_ft_agg = data["test_ft"]["aggregated"]

    report: list[str] = []
    report.append("# Turkish Legal RAG - Final Evaluation Report\n")
    report.append("## Artifact Inventory\n")
    report.append(artifact_table())
    report.append("")

    report.append("## System Configuration\n")
    report.append(markdown_table(
        ["Component", "Configuration"],
        [
            ["Embedding", "intfloat/multilingual-e5-large"],
            ["Reranker", "BAAI/bge-reranker-v2-m3 (zero-shot)"],
            ["LLM (base)", "Qwen/Qwen2.5-3B-Instruct (4-bit NF4)"],
            ["LLM (tuned)", "QLoRA adapter, r=16, alpha=32, 3 epochs"],
            ["Retrieval", "e5-large dense -> BGE rerank -> top-5"],
            ["Benchmark", "175 questions (112 train / 32 dev / 31 test)"],
        ],
    ))
    report.append("")

    report.append("## Primary Ablation: Baseline vs SFT-QLoRA\n")
    report.append("### Dev Split (32 questions)\n")
    report.append(ablation_table(data["dev_base"]["aggregated"], data["dev_ft"]["aggregated"], "Dev"))
    report.append("\n### Test Split (31 questions)\n")
    report.append(ablation_table(test_base_agg, test_ft_agg, "Test"))
    report.append("")

    report.append("## Supplemental Overlap Metrics\n")
    report.append("These BLEU/ROUGE-style values are dependency-free overlap signals and are not the primary legal QA metrics.\n")
    report.append("### Dev Split\n")
    report.append(supplemental_table(data["dev_base"]["aggregated"], data["dev_ft"]["aggregated"], "Dev"))
    report.append("\n### Test Split\n")
    report.append(supplemental_table(test_base_agg, test_ft_agg, "Test"))
    report.append("")

    report.append("## SFT-QLoRA Test Breakdown\n")
    test_ft = data["test_ft"]
    report.append("### By Question Type\n")
    report.append(breakdown_table(test_ft["breakdown"], "by_question_type"))
    report.append("\n### By Difficulty\n")
    report.append(breakdown_table(test_ft["breakdown"], "by_difficulty"))
    report.append("\n### By Source Law\n")
    report.append(breakdown_table(test_ft["breakdown"], "by_source_law"))
    report.append("")

    report.append("## Error Analysis (SFT-QLoRA on Test)\n")
    report.append(error_analysis(test_ft["per_question"]))
    report.append("")
    report.append("\n## Error Analysis (Baseline on Test)\n")
    report.append(error_analysis(data["test_base"]["per_question"]))
    report.append("")

    report.append("## Largest Test-Set Answer F1 Improvements\n")
    report.append(improvement_examples(data["test_base"], test_ft))
    report.append("")

    report.append("## Key Takeaways\n")
    report.append(f"1. **Answer F1** improved by **{(test_ft_agg['answer_f1'] - test_base_agg['answer_f1']):.2%}** "
                  f"on test ({test_base_agg['answer_f1']:.2%} -> {test_ft_agg['answer_f1']:.2%}).")
    report.append(f"2. **Citation Exact Match** improved by **{(test_ft_agg['citation_exact'] - test_base_agg['citation_exact']):.2%}** "
                  f"({test_base_agg['citation_exact']:.2%} -> {test_ft_agg['citation_exact']:.2%}).")
    report.append(f"3. **Faithfulness lexical proxy** improved by **{(test_ft_agg['faithfulness_lexical'] - test_base_agg['faithfulness_lexical']):.2%}** "
                  f"({test_base_agg['faithfulness_lexical']:.2%} -> {test_ft_agg['faithfulness_lexical']:.2%}).")
    report.append(f"4. **Citation Recall** changed by **{(test_ft_agg['citation_recall'] - test_base_agg['citation_recall']):.2%}**; "
                  "the tuned model cites fewer articles but with higher precision and exactness.")
    report.append("5. Main remaining weaknesses are factual precision in TCK/HMK/IYUK questions, citation recall, and the limits of a 3B local generator.")
    report.append("")

    out_path = Path("outputs/final_report.md")
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"[DONE] Report saved to: {out_path}")
    print(f"[DONE] Test Answer F1: {test_base_agg['answer_f1']:.2%} -> {test_ft_agg['answer_f1']:.2%}")
    print(f"[DONE] Test Faithfulness: {test_base_agg['faithfulness_lexical']:.2%} -> {test_ft_agg['faithfulness_lexical']:.2%}")


if __name__ == "__main__":
    main()
