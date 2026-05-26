# Turkish Legal RAG - Final Demo Guide

## Demo Command

Run the live-demo-safe SFT-QLoRA system:

```bash
python -m src.pipeline.rag_pipeline --lora-adapter outputs/sft_qlora/final --demo-safe
```

This keeps the final retrieval stack (`e5large_reranked_bge`) but asks for short answers and appends citations from retrieved sources. Full benchmark results are reported from the normal evaluation pipeline, not from this demo-safe display mode.

If the LoRA adapter is unavailable, run retrieval + untuned generation:

```bash
python -m src.pipeline.rag_pipeline --system e5large_reranked_bge
```

## Custom Document Demo

If the evaluator provides a folder of PDFs, rebuild the local corpus:

```bash
python scripts/prepare_custom_pdfs.py --input-dir path/to/custom_pdfs --reset
python -m src.corpus.build_registry
python -m src.corpus.register_pdfs
python -m src.retrieval.chunking
python -m src.retrieval.embedder --config configs/retrieval_config_e5large.yaml
python -m src.retrieval.vector_store --config configs/retrieval_config_e5large.yaml
python -m src.pipeline.rag_pipeline --lora-adapter outputs/sft_qlora/final --demo-safe
```

## Custom Benchmark Evaluation

If the evaluator provides a benchmark JSONL file in the same schema as `data/benchmark/gold_benchmark.jsonl`, run:

```bash
python -m src.evaluation.run_qa_eval --benchmark path/to/custom_benchmark.jsonl --system e5large_reranked_bge --lora-adapter outputs/sft_qlora/final --output-tag custom
```

## Demo Questions

| Order | Purpose | Question | Expected point |
| --- | --- | --- | --- |
| 1 | Short factual warm-up | Türkiye Devletinin yönetim şekli nedir? | Model should answer briefly and cite Madde 1. |
| 2 | List answer and citation discipline | Temel hak ve hürriyetlerin sınırlanması hangi şartlara bağlıdır? | Model should summarize the limitation conditions and cite Madde 13. |
| 3 | Procedure-oriented legal QA | Müdafiin görevlendirilmesi hangi hallerde zorunludur? | Shows procedural QA and source-backed answer behavior. |
| 4 | Known limitation case | Yağma suçunun cezası nedir? | Use only if you want to discuss remaining hallucination/citation risk. |

## 15-Minute Presentation Flow

1. Problem and motivation: Turkish legal QA needs grounded answers with article citations.
2. Phase 1 baseline: local RAG over 7 official legislation PDFs.
3. Phase 2 evaluation: 175-question gold benchmark and retrieval metrics.
4. Retrieval improvements: e5-large model selection and multilingual/BGE reranking.
5. Phase 3 generation: citation-aware prompting, QA eval, and QLoRA SFT.
6. Ablation results: baseline -> model selection -> BGE reranker -> QLoRA.
7. Phase 4 final analysis: error categories, overlap metrics, and remaining weak cases.
8. Live demo with the demo-safe command.
9. Limitations: no embedding/reranker fine-tuning, lexical faithfulness proxy, no TBMM/Yargıtay corpus.
10. Future work: larger LLM/API backend, stronger judge metric, domain-specific reranker tuning.

## Key Numbers To Mention

| Metric | Baseline | Final SFT-QLoRA |
| --- | --- | --- |
| Retrieval MRR | 0.5896 | 0.6964 |
| Retrieval Recall@5 | 0.7429 | 0.8048 |
| Answer F1 | 0.2567 | 0.4031 |
| Citation Exact | 0.0968 | 0.4516 |
| Faithfulness lexical proxy | 0.7454 | 0.9041 |

## Speaker Notes

The strongest claim is not that the system is production-ready, but that each improvement is measured and the final behavior is more grounded. The retrieval side is relatively mature; the remaining bottleneck is generation quality from a small local 3B model on complex legal questions.
