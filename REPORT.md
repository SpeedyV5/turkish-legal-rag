# Turkish Legal RAG - Final Project Report

## 1. Executive Summary

This project implements a local Retrieval-Augmented Generation (RAG) system for Turkish legal question answering. The system answers questions over a corpus of Turkish legislation and is evaluated with a gold benchmark containing questions, reference answers, and relevant legal articles.

The final deployed system uses:

| Component | Final choice |
| --- | --- |
| Corpus | 7 Turkish legal documents |
| Chunking | Hybrid article-aware chunking |
| Embedding model | `intfloat/multilingual-e5-large` |
| Vector store | FAISS `IndexFlatIP` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Generator | `Qwen/Qwen2.5-3B-Instruct` |
| Fine-tuning | QLoRA SFT LoRA adapter |
| Final demo command | `python -m src.pipeline.rag_pipeline --lora-adapter outputs/sft_qlora/final --demo-safe` |

The project progressed from a local baseline RAG system to a measured and optimized final system. The main improvement is not only higher answer overlap, but better groundedness and citation behavior. On the held-out test split, QLoRA SFT improves Answer F1 from 0.2567 to 0.4031, Citation Exact Match from 0.0968 to 0.4516, and lexical faithfulness from 0.7454 to 0.9041 when using the same final retrieval stack.

## 2. Problem Definition

Turkish legal question answering requires answers that are both useful and grounded in legal sources. A generic LLM can produce fluent answers, but it may hallucinate article numbers or legal conditions. For that reason, this project focuses on a RAG architecture where retrieval provides legal context and the generator answers only from that context.

The project goals are:

- Build a local RAG pipeline over Turkish legal documents.
- Create a gold benchmark with questions, answers, and relevant legal articles.
- Compare Base RAG and Fine-tuned RAG using the same base LLM.
- Evaluate retrieval, answer quality, citation behavior, and faithfulness.
- Provide an ablation study showing the contribution of each major system component.
- Support evaluator-provided custom PDF documents and evaluator-provided benchmark files.

## 3. Legal Corpus

The base corpus contains 7 fundamental Turkish legal documents:

| Document | Law No. |
| --- | --- |
| Constitution of the Republic of Turkey | 2709 |
| Turkish Penal Code | 5237 |
| Criminal Procedure Code | 5271 |
| Turkish Civil Code | 4721 |
| Turkish Code of Obligations | 6098 |
| Civil Procedure Code | 6100 |
| Administrative Procedure Law | 2577 |

The PDFs are processed through the corpus pipeline, converted to text, split into article-aware chunks, embedded, and indexed in FAISS. The current gold benchmark is designed for this legislation corpus. TBMM and Yargitay-style broader legal sources are intentionally left as future work because they require a wider benchmark and a broader ingestion strategy.

## 4. System Architecture

The end-to-end system has four main stages:

1. Corpus processing: PDF ingestion, text extraction, registry creation, and article-aware chunking.
2. Retrieval: dense semantic search with multilingual E5 embeddings and FAISS.
3. Reranking: cross-encoder reranking of retrieved candidate chunks.
4. Generation: Qwen2.5-3B-Instruct generates a Turkish legal answer with a `Dayanak:` citation line.

The final interactive pipeline defaults to the best retrieval stack, `e5large_reranked_bge`. The LoRA adapter is passed explicitly for the fine-tuned final system:

```bash
python -m src.pipeline.rag_pipeline --lora-adapter outputs/sft_qlora/final
```

For live presentation, a separate display mode is available:

```bash
python -m src.pipeline.rag_pipeline --lora-adapter outputs/sft_qlora/final --demo-safe
```

`--demo-safe` does not change the evaluation system. It only makes live answers shorter and appends citations from retrieved sources to reduce presentation-time generation risk.

## 5. Gold Benchmark Dataset

The project includes a gold benchmark dataset with 175 Turkish legal QA examples.

| Property | Value |
| --- | --- |
| Total questions | 175 |
| Train split | 112 |
| Dev split | 32 |
| Test split | 31 |
| Gold fields | question, expected answer, relevant document IDs, relevant articles |
| Question types | definition, list, factual, procedural, yes/no |
| Difficulty labels | easy, medium, hard |

Benchmark files:

| File | Purpose |
| --- | --- |
| `data/benchmark/gold_benchmark.jsonl` | Full 175-question benchmark |
| `data/benchmark/gold_benchmark_train.jsonl` | Training split for SFT data preparation |
| `data/benchmark/gold_benchmark_dev.jsonl` | Development split for prompt/system iteration |
| `data/benchmark/gold_benchmark_test.jsonl` | Held-out test split for final comparison |
| `data/sft/sft_train.jsonl` | SFT training data derived from the train split |

The benchmark is used in two ways. Retrieval metrics compare retrieved article chunks against the gold relevant documents/articles. QA metrics compare generated answers against expected answers and evaluate generated citations against gold articles.

## 6. Metrics

The selected metrics reflect the legal RAG setting. Exact word-for-word generation is not the only goal; the system must retrieve the right legal basis, produce a useful answer, and cite relevant articles.

### 6.1 Retrieval Metrics

| Metric | Meaning |
| --- | --- |
| MRR | How early the first correct relevant article appears |
| Recall@k | Fraction of gold relevant articles found in the top-k results |
| Precision@k | Fraction of top-k retrieved results that are relevant |
| nDCG@k | Ranking quality with higher weight for earlier relevant results |
| Hit@k | Whether at least one relevant article appears in top-k |
| Article-level recall | Whether article references match gold legal articles |

### 6.2 QA Metrics

| Metric | Meaning |
| --- | --- |
| Exact Match | Exact normalized match against the reference answer |
| Token-level F1 | Token overlap between generated and reference answers |
| Citation Precision/Recall/F1 | Match between generated `Dayanak:` articles and gold articles |
| Citation Exact Match | Whether the generated citation set exactly matches the gold citation set |
| Has Dayanak | Whether the answer includes a citation block |
| Faithfulness lexical proxy | Whether answer tokens are supported by retrieved context |

### 6.3 Supplemental Overlap Metrics

Phase 4 also adds dependency-free BLEU-1, BLEU-2, and ROUGE-L F1. These are reported only as supporting overlap signals. They are not used as the primary legal QA metrics because legal answers can be semantically correct while using different wording.

## 7. Base RAG vs Fine-tuned RAG

The professor requested comparison between Base RAG and Fine-tuned RAG using the same LLM. This project compares the same base model, `Qwen/Qwen2.5-3B-Instruct`, with and without a QLoRA adapter.

Both systems use the same final retrieval stack in the held-out test comparison:

| System | Retrieval | LLM |
| --- | --- | --- |
| Base RAG | e5-large + BGE reranker | Untuned Qwen2.5-3B-Instruct |
| Fine-tuned RAG | e5-large + BGE reranker | QLoRA-tuned Qwen2.5-3B-Instruct |

Held-out test split results:

| Metric | Base RAG | Fine-tuned RAG | Delta |
| --- | --- | --- | --- |
| Answer F1 | 0.2567 | 0.4031 | +0.1464 |
| Answer Precision | 0.2273 | 0.4051 | +0.1778 |
| Answer Recall | 0.3622 | 0.4598 | +0.0976 |
| Citation F1 | 0.4851 | 0.5742 | +0.0891 |
| Citation Precision | 0.3946 | 0.5806 | +0.1860 |
| Citation Recall | 0.8065 | 0.6237 | -0.1828 |
| Citation Exact Match | 0.0968 | 0.4516 | +0.3548 |
| Has Dayanak | 0.8387 | 0.9677 | +0.1290 |
| Faithfulness lexical proxy | 0.7454 | 0.9041 | +0.1587 |

The tuned model cites fewer articles, but the citations are more precise and exact. This is acceptable in this project because legal citation exactness is more important than producing many loosely related references.

Supplemental overlap metrics on the same held-out test split:

| Metric | Base RAG | Fine-tuned RAG | Delta |
| --- | --- | --- | --- |
| BLEU-1 | 0.2387 | 0.3460 | +0.1073 |
| BLEU-2 | 0.1869 | 0.3063 | +0.1194 |
| ROUGE-L F1 | 0.2718 | 0.4083 | +0.1365 |

These overlap metrics support the same conclusion as the primary metrics: the fine-tuned system produces answers closer to the gold references while improving citation exactness and faithfulness.

## 8. Retrieval Experiments

The project compares multiple retrieval systems on the 175-question benchmark. The final BGE reranker was added after the earlier Phase 2 systems and became the best retrieval configuration.

| System | MRR | Recall@5 | Recall@10 |
| --- | --- | --- | --- |
| baseline_dense (e5-base) | 0.5896 | 0.7429 | 0.7933 |
| bm25_only | 0.3208 | 0.4657 | 0.5667 |
| hybrid (e5-base + BM25) | 0.5510 | 0.7171 | 0.8057 |
| dense_reranked (English CE) | 0.4139 | 0.6229 | 0.7286 |
| hybrid_reranked (English CE) | 0.4122 | 0.5943 | 0.7257 |
| dense_reranked_ml | 0.6568 | 0.7524 | 0.8086 |
| hybrid_reranked_ml | 0.6587 | 0.7533 | 0.8143 |
| e5large_dense | 0.6773 | 0.7629 | 0.8476 |
| e5large_reranked_ml | 0.6644 | 0.7952 | 0.8514 |
| e5large_reranked_bge | 0.6964 | 0.8048 | 0.8743 |

Key retrieval findings:

- BM25 alone is weak because Turkish morphology makes simple lexical matching difficult.
- English reranking is harmful on Turkish legal text.
- E5-large improves semantic retrieval substantially over E5-base.
- BGE reranking gives the best MRR, Recall@5, and Recall@10 in the final comparison.

## 9. Ablation Study

The ablation study measures the effect of the major system changes. Retrieval metrics are measured on the full benchmark where available, while QA metrics are measured on the held-out test split.

| Variant | Retrieval | LLM | MRR | Recall@5 | Answer F1 | Citation F1 | Citation Exact | Faithfulness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1. Baseline | e5-base dense | Untuned Qwen 3B | 0.5896 | 0.7429 | 0.2453 | 0.3765 | 0.1290 | 0.7228 |
| 2. Model selection | e5-large dense | Untuned Qwen 3B | 0.6773 | 0.7629 | 0.3024 | 0.5233 | 0.1935 | 0.7615 |
| 3. Reranker | e5-large + BGE | Untuned Qwen 3B | 0.6964 | 0.8048 | 0.2567 | 0.4851 | 0.0968 | 0.7454 |
| 4. QLoRA SFT | e5-large + BGE | QLoRA Qwen 3B | 0.6964 | 0.8048 | 0.4031 | 0.5742 | 0.4516 | 0.9041 |

This ablation shows that:

- The embedding model change improves retrieval quality.
- BGE reranking improves retrieval ranking and recall.
- The untuned generator does not automatically benefit from every retrieval change, because better retrieval can still expose answer-generation weaknesses.
- QLoRA SFT significantly improves generation quality, citation exactness, and faithfulness while using the same retrieval stack.

Embedding fine-tuning and reranker fine-tuning were not completed due to local GPU limits and project time constraints. Instead, the project honestly reports model selection for embeddings and zero-shot BGE reranking. This still satisfies the ablation requirement by isolating retrieval model selection, reranking, and LLM fine-tuning effects.

## 10. Fine-tuning Details

The generator is fine-tuned with QLoRA supervised fine-tuning.

| Parameter | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Quantization | 4-bit NF4 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Epochs | 3 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Learning rate | 2e-4 |
| Scheduler | cosine |
| Max sequence length | 1024 |
| Training examples | 112 |
| Hardware | NVIDIA RTX 3070 Laptop GPU, 8GB VRAM |
| Training time | Approximately 22 minutes |

The SFT data is generated from the train split only. Dev and test questions are kept separate to avoid leakage.

## 11. Error Analysis

The final fine-tuned system performs better than the base system but is not perfect.

Held-out test summary:

| Error signal | Base RAG | Fine-tuned RAG |
| --- | --- | --- |
| Good answers (F1 > 0.5) | 4/31 | 13/31 |
| Low answer F1 (< 0.25) | 17/31 | 11/31 |
| Low faithfulness (< 0.5) | 5/31 | 2/31 |
| Missing Dayanak | 5/31 | 1/31 |
| Citation exact mismatch | 28/31 | 17/31 |

Remaining weak areas:

- Turkish Penal Code questions involving exact penalty amounts and qualified forms are still risky.
- Civil procedure and administrative procedure questions can be incomplete even when the answer is faithful to context.
- The lexical faithfulness metric is only a proxy; it does not fully prove legal entailment.
- The 3B local generator is the main bottleneck for complex, multi-condition legal questions.

The project therefore presents the system as a measured academic prototype, not as a production legal advice tool.

## 12. Custom Document Collection Support

The professor must be able to provide a custom document collection. This project supports evaluator-provided PDF folders through `scripts/prepare_custom_pdfs.py`.

Example workflow:

```bash
python scripts/prepare_custom_pdfs.py --input-dir path/to/custom_pdfs --reset
python -m src.corpus.build_registry
python -m src.corpus.register_pdfs
python -m src.retrieval.chunking
python -m src.retrieval.embedder --config configs/retrieval_config_e5large.yaml
python -m src.retrieval.vector_store --config configs/retrieval_config_e5large.yaml
python -m src.pipeline.rag_pipeline --lora-adapter outputs/sft_qlora/final --demo-safe
```

This replaces the local raw PDF inputs, rebuilds the corpus artifacts, re-embeds the corpus, rebuilds the FAISS index, and runs the same RAG system over the evaluator-provided documents.

## 13. Custom Benchmark Support

The professor may also provide a custom benchmark question-answer set. The QA evaluation script supports an external JSONL benchmark through `--benchmark`.

Example:

```bash
python -m src.evaluation.run_qa_eval --benchmark path/to/custom_benchmark.jsonl --system e5large_reranked_bge --lora-adapter outputs/sft_qlora/final --output-tag custom
```

The expected JSONL format should follow the project benchmark fields:

```json
{
  "id": "custom_001",
  "question": "Question text",
  "expected_answer": "Reference answer",
  "relevant_doc_ids": ["doc_id"],
  "relevant_articles": ["Madde 1"],
  "question_type": "definition",
  "difficulty": "easy",
  "source_law": "Custom Document"
}
```

When a custom benchmark is provided in this format, the same retrieval, generation, citation, faithfulness, and overlap metrics are computed on that benchmark. This directly addresses the requirement that all systems be testable on the same evaluator-prepared benchmark.

## 14. Demo Plan

The recommended live demo command is:

```bash
python -m src.pipeline.rag_pipeline --lora-adapter outputs/sft_qlora/final --demo-safe
```

Suggested demo questions:

| Purpose | Question |
| --- | --- |
| Short factual answer | Türkiye Devletinin yönetim şekli nedir? |
| List behavior and citation | Temel hak ve hürriyetlerin sınırlanması hangi şartlara bağlıdır? |
| Procedure-oriented QA | Müdafiin görevlendirilmesi hangi hallerde zorunludur? |
| Limitation discussion | Yağma suçunun cezası nedir? |

The recommended presentation structure is:

1. Motivation: grounded Turkish legal QA.
2. Baseline local RAG system.
3. Gold benchmark and metrics.
4. Retrieval experiments and BGE selection.
5. Base RAG vs Fine-tuned RAG.
6. Ablation study.
7. Error analysis and limitations.
8. Live demo.
9. Custom corpus and custom benchmark support.

## 15. Environment and Hardware

| Item | Value |
| --- | --- |
| Python | 3.11 |
| CUDA | 12.4 |
| PyTorch | 2.7.0+cu124 |
| Phase 2 GPU | NVIDIA RTX 3050 Laptop GPU, 4GB VRAM |
| Phase 3 GPU | NVIDIA RTX 3070 Laptop GPU, 8GB VRAM |
| Phase 4 local demo GPU | NVIDIA RTX 3050 Laptop GPU, 4GB VRAM |
| Main libraries | transformers, sentence-transformers, peft, trl, bitsandbytes, faiss-cpu |

The final system is designed to run locally, but latency is still significant because the reranker and quantized LLM run on consumer hardware. Observed interactive latency is approximately 17-21 seconds per question in benchmark runs and around 26-31 seconds in full smoke-test examples.

## 16. Limitations and Future Work

Limitations:

- No embedding fine-tuning was performed.
- No reranker fine-tuning was performed.
- Faithfulness is measured with a lexical proxy, not NLI or LLM-as-judge.
- The corpus is limited to 7 legislation PDFs.
- The 3B generator can struggle with complex legal reasoning and exact penalty details.
- Live demo mode is optimized for concise presentation, while benchmark results come from the normal evaluation pipeline.

Future work:

- Domain-specific embedding fine-tuning with contrastive learning.
- BGE reranker fine-tuning with Turkish legal query-passage pairs.
- Larger generator or API-backed LLM comparison.
- NLI or LLM-as-judge faithfulness scoring.
- Expansion to court decisions, TBMM records, and broader legal sources.
- Cross-validation or a larger held-out benchmark.

## 17. Compliance With Professor Requirements

| Requirement | Project status |
| --- | --- |
| Submit project files, links, and reports | Repository, README, final report, demo guide, benchmark files, and evaluation artifacts are prepared. |
| Compare Base RAG vs Fine-tuned RAG using the same LLM | Done with Qwen2.5-3B-Instruct base vs QLoRA-tuned Qwen2.5-3B-Instruct on the same retrieval stack. |
| Use meaningful metrics depending on gold benchmark availability | Done. The project has a gold benchmark and reports retrieval metrics, answer F1, citation metrics, faithfulness, and supplemental BLEU/ROUGE-style metrics. |
| Gold benchmark with Question-Answer-Relevant Documents | Done in `data/benchmark/gold_benchmark*.jsonl`. |
| Ablation study | Done for baseline, E5-large model selection, BGE reranker, and QLoRA SFT. Embedding/reranker fine-tuning limitations are explicitly explained. |
| Custom document collection support | Done through `scripts/prepare_custom_pdfs.py` and corpus rebuild commands. |
| Evaluator-prepared custom benchmark support | Done through `run_qa_eval --benchmark path/to/custom_benchmark.jsonl`. |

## 18. Final Conclusion

The final project is a complete, locally runnable Turkish legal RAG system with a documented benchmark, measurable retrieval and QA evaluation, a Base RAG vs Fine-tuned RAG comparison, an ablation study, custom document support, and custom benchmark support.

The strongest final result is the improvement from the base generator to the QLoRA-tuned generator on the same final retrieval stack: Answer F1 improves from 0.2567 to 0.4031, Citation Exact Match improves from 0.0968 to 0.4516, and lexical faithfulness improves from 0.7454 to 0.9041. The remaining bottleneck is not retrieval, but the limited generation capacity of a small local 3B LLM on complex legal questions.
