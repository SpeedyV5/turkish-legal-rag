"""Compare baseline vs SFT-QLoRA QA eval results on dev and test splits."""
import json

splits = {
    "DEV": {
        "base": "outputs/qa_eval/qa_eval_dev_e5large_reranked_ml_prompt_v5.json",
        "ft": "outputs/qa_eval/qa_eval_dev_e5large_reranked_bge_sft_qlora.json",
    },
    "TEST": {
        "base": "outputs/qa_eval/qa_eval_test_e5large_reranked_bge_baseline.json",
        "ft": "outputs/qa_eval/qa_eval_test_e5large_reranked_bge_sft_qlora.json",
    },
}

keys = [
    "answer_f1", "answer_precision", "answer_recall",
    "citation_f1", "citation_precision", "citation_recall", "citation_exact",
    "has_dayanak", "faithfulness_lexical",
]

for split_name, paths in splits.items():
    base = json.load(open(paths["base"], "r", encoding="utf-8"))["aggregated"]
    ft = json.load(open(paths["ft"], "r", encoding="utf-8"))["aggregated"]

    print(f"\n{'=' * 60}")
    print(f"  {split_name} SPLIT: Baseline vs SFT-QLoRA")
    print(f"{'=' * 60}")
    header = f"{'Metric':<24} {'Baseline':>10} {'SFT-QLoRA':>10} {'Delta':>10}"
    print(header)
    print("-" * 56)
    for k in keys:
        b = base.get(k, 0)
        f = ft.get(k, 0)
        d = f - b
        sign = "+" if d >= 0 else ""
        print(f"{k:<24} {b:>10.4f} {f:>10.4f} {sign}{d:>9.4f}")
