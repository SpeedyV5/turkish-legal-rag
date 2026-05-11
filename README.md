# Turkish Legal RAG

Türk hukuk mevzuatı üzerinde çalışan, optimize edilmiş bir RAG (Retrieval-Augmented Generation) tabanlı soru-cevap sistemi. Embedding model seçimi, cross-encoder reranking ve QLoRA LLM fine-tuning ile uçtan uca iyileştirilmiştir.

## Proje Özeti

Bu proje, 7 temel Türk kanun metninden oluşan bir korpus üzerinde hukuki soruları yanıtlayan bir pipeline geliştirmeyi amaçlamaktadır. Sistem baseline dense retrieval'dan başlayarak model seçimi, reranking, prompt iteration ve QLoRA SFT ile kademeli olarak iyileştirilmiştir.

Ablation zinciri:

1. **Baseline** — e5-base dense retrieval, untuned Qwen2.5-3B-Instruct
2. **+ Embedding model seçimi** — e5-large (MRR +14.9%)
3. **+ Reranker** — BAAI/bge-reranker-v2-m3 zero-shot
4. **+ Prompt tuning** — citation discipline, Dayanak formatı
5. **+ QLoRA SFT** — 112 örnek, 3 epoch (answer_f1 +14.6%, faithfulness +15.9%)

## Korpus

| Belge | Kanun No |
|-------|----------|
| Türkiye Cumhuriyeti Anayasası | 2709 |
| Türk Ceza Kanunu | 5237 |
| Ceza Muhakemesi Kanunu | 5271 |
| Türk Medeni Kanunu | 4721 |
| Türk Borçlar Kanunu | 6098 |
| Hukuk Muhakemeleri Kanunu | 6100 |
| İdari Yargılama Usulü Kanunu | 2577 |

**Not:** `data/raw/` altında TBMM ve Yargıtay klasörleri yapısal olarak mevcuttur ancak boştur. Mevcut 175 soruluk benchmark tamamen yukarıdaki 7 mevzuat ile cevaplanabilir durumdadır. Bu kaynakların eklenmesi future work olarak planlanmıştır.

## Proje Yapısı

```
turkish-legal-rag/
├── configs/
│   ├── corpus_config.yaml              # Korpus kaynak ayarları
│   ├── data_config.yaml                # HuggingFace veri seti ayarları
│   ├── retrieval_config.yaml           # E5-base embedding, chunking, FAISS
│   ├── retrieval_config_e5large.yaml   # E5-large embedding ayarları [Faz 2]
│   ├── evaluation_config.yaml          # Benchmark ve metrik ayarları
│   ├── generation_config.yaml          # LLM ve prompt ayarları
│   └── sft_config.yaml                # QLoRA SFT eğitim ayarları [Faz 3]
├── src/
│   ├── corpus/                         # PDF indirme, metin çıkarma, registry
│   ├── data/                           # HuggingFace veri indirme ve inceleme
│   ├── retrieval/
│   │   ├── chunking.py                 # Hybrid article-aware chunking
│   │   ├── embedder.py                 # Sentence-transformer embedding
│   │   ├── vector_store.py             # FAISS index oluşturma
│   │   ├── bm25_retriever.py           # BM25 sparse retriever [Faz 2]
│   │   ├── hybrid_retriever.py         # Dense + BM25 hybrid (RRF) [Faz 2]
│   │   ├── reranker.py                 # Cross-encoder reranker [Faz 2]
│   │   └── legacy/                     # Eski keyword-bonus retriever (deprecated)
│   ├── generation/
│   │   ├── generator.py                # Qwen 3B inference + LoRA adapter desteği [Faz 3]
│   │   └── prompt_builder.py           # Soru tipi tespiti, citation-aware prompt [Faz 3]
│   ├── pipeline/
│   │   └── rag_pipeline.py             # Uçtan uca interaktif RAG
│   ├── benchmark/                      # [Faz 2]
│   │   ├── gold_questions.py           # 175 doğrulanmış hukuk sorusu
│   │   └── generate_benchmark.py       # Benchmark JSONL oluşturucu
│   └── evaluation/                     # [Faz 2 + 3]
│       ├── metrics.py                  # Retrieval: Recall@k, MRR, nDCG, article_recall
│       ├── qa_metrics.py               # QA: EM, F1, citation, faithfulness [Faz 3]
│       ├── run_retrieval_eval.py       # Retrieval evaluation runner
│       └── run_qa_eval.py              # End-to-end QA evaluation runner [Faz 3]
├── scripts/
│   ├── split_benchmark.py             # Stratified train/dev/test split [Faz 3]
│   ├── suggest_list_gold_expansion.py  # Liste gold genişletme önerileri [Faz 3]
│   ├── apply_list_gold_expansion.py    # Gold genişletme uygulama [Faz 3]
│   ├── prepare_sft_data.py            # SFT veri hazırlama [Faz 3]
│   ├── train_sft_qlora.py             # QLoRA eğitim scripti [Faz 3]
│   ├── merge_lora.py                  # LoRA adapter → full model merge [Faz 3]
│   ├── compare_baseline_vs_sft.py     # Baseline vs SFT karşılaştırma [Faz 3]
│   └── final_analysis.py              # Final rapor üretici [Faz 3]
├── data/
│   ├── raw/mevzuat/pdfs/              # 7 kanun PDF'i
│   ├── processed/corpus/              # Registry, chunks, embeddings, FAISS index
│   ├── benchmark/                     # Gold benchmark + train/dev/test split [Faz 2+3]
│   └── sft/                           # SFT eğitim verisi [Faz 3]
├── outputs/
│   ├── evaluation/                    # 9 retrieval sistemi sonuçları [Faz 2]
│   ├── qa_eval/                       # QA evaluation sonuçları [Faz 3]
│   ├── sft_qlora/                     # QLoRA checkpoint ve final adapter [Faz 3]
│   └── final_report.md               # Otomatik üretilen sonuç raporu [Faz 3]
├── requirements.txt
├── README.md
└── REPORT.md
```

## Kurulum

```bash
pip install -r requirements.txt
```

GPU kullanımı için CUDA destekli PyTorch gereklidir:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Kullanım

### İnteraktif Pipeline (Best System)

Varsayılan olarak en iyi retrieval stack (e5-large + bge-reranker) kullanılır:
```bash
python -m src.pipeline.rag_pipeline
```

QLoRA fine-tuned model ile çalıştırmak için:
```bash
python -m src.pipeline.rag_pipeline --lora-adapter outputs/sft_qlora/final
```

### Retrieval Evaluation (Faz 2)

9 retrieval sisteminin tamamı:
```bash
python -m src.evaluation.run_retrieval_eval --system baseline_dense
python -m src.evaluation.run_retrieval_eval --system bm25_only
python -m src.evaluation.run_retrieval_eval --system hybrid
python -m src.evaluation.run_retrieval_eval --system dense_reranked
python -m src.evaluation.run_retrieval_eval --system hybrid_reranked
python -m src.evaluation.run_retrieval_eval --system dense_reranked_ml
python -m src.evaluation.run_retrieval_eval --system hybrid_reranked_ml
python -m src.evaluation.run_retrieval_eval --system e5large_dense --config configs/retrieval_config_e5large.yaml
python -m src.evaluation.run_retrieval_eval --system e5large_reranked_ml --config configs/retrieval_config_e5large.yaml
```

### QA Evaluation (Faz 3)

Baseline (untuned) model ile dev set:
```bash
python -m src.evaluation.run_qa_eval --split dev --system e5large_reranked_bge
```

SFT-QLoRA model ile test set:
```bash
python -m src.evaluation.run_qa_eval --split test --system e5large_reranked_bge --lora-adapter outputs/sft_qlora/final --output-tag sft_qlora
```

### SFT / QLoRA Eğitimi (Faz 3)

SFT verisini hazırla (train split'ten):
```bash
python scripts/prepare_sft_data.py
```

QLoRA eğitimi başlat:
```bash
set PYTHONUTF8=1
python scripts/train_sft_qlora.py
```

LoRA adapter'ı base model'e merge et (opsiyonel):
```bash
python scripts/merge_lora.py --adapter outputs/sft_qlora/final --output outputs/merged_model
```

### Benchmark Split ve Gold Expansion (Faz 3)

```bash
python scripts/split_benchmark.py
python scripts/suggest_list_gold_expansion.py
python scripts/apply_list_gold_expansion.py
```

## Mevcut Durum

- **Faz 1** (İbo): ✅ Tamamlandı — Baseline RAG pipeline
- **Faz 2** (Deniz): ✅ Tamamlandı — Benchmark, evaluation altyapısı, 9 retrieval sistemi
- **Faz 3** (İbo): ✅ Büyük ölçüde tamamlandı — QA eval, prompt tuning, QLoRA SFT, ablation
- **Faz 4** (Deniz): 🔜 Devam ediyor — Hata analizi, rapor detaylandırma, sunum

## En İyi Sistem: Neden Bu Seçildi?

Pipeline varsayılanı: **e5-large + bge-reranker-v2-m3 + QLoRA-tuned Qwen2.5-3B-Instruct**

Retrieval tarafında 9 sistem karşılaştırıldı. Seçim gerekçesi:

| Kriter | En iyi sistem | Değer |
|--------|---------------|-------|
| MRR | **e5large_reranked_bge** | **0.6964** |
| Recall@5 | **e5large_reranked_bge** | **0.8048** |
| Recall@10 | **e5large_reranked_bge** | **0.8743** |

`bge-reranker-v2-m3` Faz 3'te entegre edilmiş ve tüm retrieval metriklerinde (MRR, Recall@5, Recall@10) en iyi sonucu veren sistem olmuştur. Faz 2'deki 9 sisteme ek olarak değerlendirilmiş ve pipeline varsayılanı yapılmıştır.

## Sonuç Tabloları

### Retrieval (tüm benchmark — 175 soru)

| Sistem | MRR | Recall@5 | Recall@10 |
|--------|-----|----------|-----------|
| Dense Baseline (e5-base) | 0.5896 | 0.7429 | 0.7933 |
| E5-Large Dense | 0.6773 | 0.7629 | 0.8476 |
| E5-Large + ML Reranker | 0.6644 | 0.7952 | 0.8514 |
| **E5-Large + BGE Reranker** | **0.6964** | **0.8048** | **0.8743** |

### QA (Faz 3, test split — 31 soru)

| Metrik | Baseline | SFT-QLoRA | Delta |
|--------|----------|-----------|-------|
| Answer F1 | 0.2567 | **0.4031** | **+14.6%** |
| Answer Precision | 0.2273 | **0.4051** | **+17.8%** |
| Citation F1 | 0.4851 | **0.5742** | **+8.9%** |
| Citation Exact | 0.0968 | **0.4516** | **+35.5%** |
| Faithfulness | 0.7454 | **0.9041** | **+15.9%** |
| Has Dayanak | 0.8387 | **0.9677** | **+12.9%** |

### Ablation Zinciri (test split — 31 soru)

| Varyant | MRR | R@5 | Ans F1 | Cite F1 | Cite Exact | Faith |
|---------|-----|-----|--------|---------|------------|-------|
| 1. e5-base dense + untuned | 0.5896 | 0.7429 | 0.2453 | 0.3765 | 0.1290 | 0.7228 |
| 2. e5-large dense + untuned | 0.6773 | 0.7629 | 0.3024 | 0.5233 | 0.1935 | 0.7615 |
| 3. e5-large + BGE + untuned | 0.6964 | 0.8048 | 0.2567 | 0.4851 | 0.0968 | 0.7454 |
| 4. e5-large + BGE + QLoRA | 0.6964 | 0.8048 | **0.4031** | **0.5742** | **0.4516** | **0.9041** |

Not: Retrieval metrikleri 175 soru üzerinden, QA metrikleri test split 31 soru üzerinden ölçülmüştür. Adım 2 gerçek embedding fine-tuning değil, daha güçlü pretrained model seçimidir.

## Yapılmayanlar ve Sınırlılıklar

Aşağıdaki çalışmalar 8GB yerel VRAM kısıtı ve proje takvimi nedeniyle yapılamamıştır:

- **Embedding fine-tuning** / contrastive tuning / hard negative mining — VRAM yetersiz; bunun yerine model seçimi (e5-base → e5-large) ile büyük iyileşme sağlandı
- **Reranker fine-tuning** — bge-reranker-v2-m3 zero-shot zaten iyi performans verdi; ertelendi
- **BLEU / ROUGE metrikleri** — Hukuk metinlerinde token-level F1 daha anlamlı, ama eklenebilir
- **TBMM / Yargıtay verisi** — Mevcut benchmark yalnızca 7 mevzuat ile uyumlu; future work

Buna karşılık, 8GB VRAM sınırlarında model selection + reranker + QLoRA hattı önceliklendirilmiş ve answer_f1'de +14.6%, faithfulness'ta +15.9% iyileşme sağlanmıştır.

## Benchmark Dosyaları

| Dosya | İçerik |
|-------|--------|
| `data/benchmark/gold_benchmark.jsonl` | Tam benchmark (175 soru) |
| `data/benchmark/gold_benchmark_train.jsonl` | Eğitim seti (112 soru) — SFT verisi buradan |
| `data/benchmark/gold_benchmark_dev.jsonl` | Geliştirme seti (32 soru) — prompt tuning |
| `data/benchmark/gold_benchmark_test.jsonl` | Test seti (31 soru) — final değerlendirme |
| `data/benchmark/split_manifest.json` | Split metadata |
| `data/sft/sft_train.jsonl` | QLoRA SFT eğitim verisi (chat format) |

## Evaluation Çıktıları

| Klasör | İçerik |
|--------|--------|
| `outputs/evaluation/` | 10 retrieval sisteminin eval JSON dosyaları + comparison_report.json |
| `outputs/qa_eval/` | Dev ve test split QA eval sonuçları (baseline + SFT + ablation varyantları) |
| `outputs/sft_qlora/final/` | Eğitilmiş QLoRA LoRA adapter (~60MB) |
| `outputs/final_report.md` | Otomatik üretilen Markdown sonuç raporu |

## Donanım

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU (8GB VRAM)
- PyTorch: 2.7.0+cu124
- CUDA: 12.4
- QLoRA eğitim süresi: ~22 dakika (3 epoch, 42 step)
- Inference latency: baseline ~17s/soru, SFT-QLoRA ~21s/soru
