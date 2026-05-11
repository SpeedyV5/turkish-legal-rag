"""Final analysis: error breakdown, citation accuracy, faithfulness, hallucination.

Generates a comprehensive Markdown report from QA eval JSON outputs.

Usage:
  python scripts/final_analysis.py
"""
import json
from collections import defaultdict
from pathlib import Path


def load_eval(path: str) -> dict:
    return json.load(open(path, "r", encoding="utf-8"))


def fmt(v, pct=True):
    if pct:
        return f"{v:.2%}" if isinstance(v, float) else str(v)
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def ablation_table(base_agg: dict, ft_agg: dict, split: str) -> str:
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
    for k, label in keys:
        b = base_agg.get(k, 0)
        f = ft_agg.get(k, 0)
        d = f - b
        sign = "+" if d >= 0 else ""
        rows.append([label, fmt(b), fmt(f), f"{sign}{d:.2%}"])
    return markdown_table(
        [f"Metric ({split})", "Baseline", "SFT-QLoRA", "Δ"], rows
    )


def breakdown_table(breakdown: dict, category: str, metric: str = "answer_f1") -> str:
    data = breakdown.get(category, {})
    rows = []
    for key in sorted(data.keys()):
        agg = data[key]
        rows.append([
            key,
            fmt(agg.get("answer_f1", 0)),
            fmt(agg.get("citation_f1", 0)),
            fmt(agg.get("faithfulness_lexical", 0)),
            str(int(agg.get("n_answer_tokens", 0))),
        ])
    return markdown_table(
        [category.replace("by_", "").title(), "Ans F1", "Cite F1", "Faith", "Avg Tokens"],
        rows,
    )


def error_analysis(per_q: list[dict]) -> str:
    """Categorize errors and provide examples."""
    hallucination = []  # low faithfulness
    bad_citation = []    # has_dayanak=0 or citation_exact=0
    good = []            # answer_f1 > 0.5

    for q in per_q:
        m = q["metrics"]
        if m.get("faithfulness_lexical", 1) < 0.5:
            hallucination.append(q)
        if not q.get("has_dayanak", True):
            bad_citation.append(q)
        if m.get("answer_f1", 0) > 0.5:
            good.append(q)

    lines = []
    lines.append(f"- **Total questions**: {len(per_q)}")
    lines.append(f"- **Good answers** (F1 > 0.5): {len(good)} ({len(good)/len(per_q):.0%})")
    lines.append(f"- **Low faithfulness** (< 0.5): {len(hallucination)} ({len(hallucination)/len(per_q):.0%})")
    lines.append(f"- **Missing Dayanak**: {len(bad_citation)} ({len(bad_citation)/len(per_q):.0%})")

    if hallucination:
        lines.append("\n**Worst faithfulness cases:**")
        for q in sorted(hallucination, key=lambda x: x["metrics"]["faithfulness_lexical"])[:3]:
            lines.append(f"  - `{q['id']}`: faith={q['metrics']['faithfulness_lexical']:.2f}, "
                        f"F1={q['metrics']['answer_f1']:.2f}")

    if bad_citation:
        lines.append("\n**Missing Dayanak examples:**")
        for q in bad_citation[:3]:
            lines.append(f"  - `{q['id']}`: answer='{q['answer'][:80]}...'")

    return "\n".join(lines)


def main():
    # Paths
    results = {
        "dev_base": "outputs/qa_eval/qa_eval_dev_e5large_reranked_ml_prompt_v5.json",
        "dev_ft": "outputs/qa_eval/qa_eval_dev_e5large_reranked_bge_sft_qlora.json",
        "test_base": "outputs/qa_eval/qa_eval_test_e5large_reranked_bge_baseline.json",
        "test_ft": "outputs/qa_eval/qa_eval_test_e5large_reranked_bge_sft_qlora.json",
    }

    data = {k: load_eval(v) for k, v in results.items()}

    report = []
    report.append("# Turkish Legal RAG — Final Evaluation Report\n")

    # System description
    report.append("## System Configuration\n")
    report.append("| Component | Configuration |")
    report.append("| --- | --- |")
    report.append("| Embedding | intfloat/multilingual-e5-large |")
    report.append("| Reranker | BAAI/bge-reranker-v2-m3 (zero-shot) |")
    report.append("| LLM (base) | Qwen/Qwen2.5-3B-Instruct (4-bit NF4) |")
    report.append("| LLM (tuned) | + QLoRA r=16, α=32, 3 epochs |")
    report.append("| Retrieval | e5-large dense → bge rerank → top-5 |")
    report.append("| Benchmark | 175 questions (112 train / 32 dev / 31 test) |")
    report.append("")

    # Main ablation tables
    report.append("## Ablation: Baseline vs SFT-QLoRA\n")
    report.append("### Dev Split (32 questions)\n")
    report.append(ablation_table(data["dev_base"]["aggregated"],
                                 data["dev_ft"]["aggregated"], "Dev"))
    report.append("\n### Test Split (31 questions)\n")
    report.append(ablation_table(data["test_base"]["aggregated"],
                                 data["test_ft"]["aggregated"], "Test"))
    report.append("")

    # Breakdown tables for SFT model on test
    report.append("## SFT-QLoRA Test Breakdown\n")
    test_ft = data["test_ft"]
    if "breakdown" in test_ft:
        report.append("### By Question Type\n")
        report.append(breakdown_table(test_ft["breakdown"], "by_question_type"))
        report.append("\n### By Difficulty\n")
        report.append(breakdown_table(test_ft["breakdown"], "by_difficulty"))
        report.append("\n### By Source Law\n")
        report.append(breakdown_table(test_ft["breakdown"], "by_source_law"))
    report.append("")

    # Error analysis on test SFT
    report.append("## Error Analysis (SFT-QLoRA on Test)\n")
    report.append(error_analysis(test_ft["per_question"]))
    report.append("")

    # Error analysis on test baseline for comparison
    report.append("\n## Error Analysis (Baseline on Test)\n")
    report.append(error_analysis(data["test_base"]["per_question"]))
    report.append("")

    # Key takeaways
    test_base_agg = data["test_base"]["aggregated"]
    test_ft_agg = data["test_ft"]["aggregated"]
    report.append("## Key Takeaways\n")
    report.append(f"1. **Answer F1** improved by **{(test_ft_agg['answer_f1']-test_base_agg['answer_f1']):.2%}** "
                  f"on test ({test_base_agg['answer_f1']:.2%} → {test_ft_agg['answer_f1']:.2%})")
    report.append(f"2. **Citation Precision** improved by **{(test_ft_agg['citation_precision']-test_base_agg['citation_precision']):.2%}** "
                  f"({test_base_agg['citation_precision']:.2%} → {test_ft_agg['citation_precision']:.2%})")
    report.append(f"3. **Citation Exact Match** improved by **{(test_ft_agg['citation_exact']-test_base_agg['citation_exact']):.2%}** "
                  f"({test_base_agg['citation_exact']:.2%} → {test_ft_agg['citation_exact']:.2%})")
    report.append(f"4. **Faithfulness** improved by **{(test_ft_agg['faithfulness_lexical']-test_base_agg['faithfulness_lexical']):.2%}** "
                  f"({test_base_agg['faithfulness_lexical']:.2%} → {test_ft_agg['faithfulness_lexical']:.2%})")
    report.append(f"5. **Citation Recall** dropped by **{(test_ft_agg['citation_recall']-test_base_agg['citation_recall']):.2%}** "
                  f"— model is more conservative (fewer but more accurate citations)")
    report.append(f"6. **Has Dayanak** improved to **{test_ft_agg['has_dayanak']:.2%}** "
                  f"(from {test_base_agg['has_dayanak']:.2%})")
    report.append("")

    # Write report
    out_path = Path("outputs/final_report.md")
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"[DONE] Report saved to: {out_path}")
    print("\n" + "\n".join(report))


if __name__ == "__main__":
    main()
