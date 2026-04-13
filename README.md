# Turkish Legal RAG

Türk hukuk mevzuatı üzerinde çalışan, optimize edilmiş bir RAG (Retrieval-Augmented Generation) tabanlı soru-cevap sistemi.

## Proje Özeti

Bu proje, 7 temel Türk kanun metninden oluşan bir korpus üzerinde hukuki soruları yanıtlayan bir pipeline geliştirmeyi amaçlamaktadır. Sistem baseline dense retrieval'dan başlayarak hybrid retrieval, reranking ve LLM fine-tuning aşamalarıyla iyileştirilmektedir.

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

## Proje Yapısı

```
turkish-legal-rag/
├── configs/
│   ├── corpus_config.yaml          # Korpus kaynak ayarları
│   ├── data_config.yaml            # HuggingFace veri seti ayarları
│   ├── retrieval_config.yaml       # E5-base embedding, chunking, FAISS ayarları
│   ├── retrieval_config_e5large.yaml # E5-large embedding ayarları [Faz 2]
│   ├── evaluation_config.yaml      # Benchmark ve metrik ayarları
│   └── generation_config.yaml      # LLM ve prompt ayarları
├── src/
│   ├── corpus/                     # PDF indirme, metin çıkarma, registry yönetimi
│   ├── data/                       # HuggingFace veri indirme ve inceleme
│   ├── retrieval/
│   │   ├── chunking.py             # Hybrid article-aware chunking
│   │   ├── embedder.py             # Sentence-transformer embedding
│   │   ├── vector_store.py         # FAISS index oluşturma
│   │   ├── retriever.py            # Dense retriever (baseline, keyword bonus)
│   │   ├── bm25_retriever.py       # BM25 sparse retriever [Faz 2]
│   │   ├── hybrid_retriever.py     # Dense + BM25 hybrid (RRF/weighted) [Faz 2]
│   │   └── reranker.py             # Cross-encoder reranker [Faz 2]
│   ├── generation/
│   │   ├── generator.py            # Qwen2.5-3B-Instruct local inference
│   │   └── prompt_builder.py       # Soru tipi tespiti ve prompt oluşturma
│   ├── pipeline/
│   │   └── rag_pipeline.py         # Uçtan uca interaktif RAG
│   ├── benchmark/                  # [Faz 2]
│   │   ├── gold_questions.py       # 175 doğrulanmış hukuk sorusu
│   │   └── generate_benchmark.py   # Benchmark JSONL oluşturucu
│   └── evaluation/                 # [Faz 2]
│       ├── metrics.py              # Recall@k, MRR, nDCG, F1, EM
│       ├── run_retrieval_eval.py   # Retrieval evaluation runner
│       └── run_comparison.py       # Sistem karşılaştırma raporu
├── data/
│   ├── raw/mevzuat/pdfs/           # 7 kanun PDF'i
│   ├── processed/corpus/           # Registry, chunks, embeddings, FAISS index
│   └── benchmark/                  # Gold benchmark JSONL [Faz 2]
├── outputs/
│   ├── evaluation/                 # Evaluation sonuçları [Faz 2]
│   └── runs/data_reports/          # Veri inceleme raporları
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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Kullanım

### Baseline Pipeline (Faz 1)
```bash
python -m src.pipeline.rag_pipeline
```

### Benchmark Oluşturma (Faz 2)
```bash
python -m src.benchmark.generate_benchmark
```

### Retrieval Evaluation (Faz 2)

E5-base (baseline) config ile:
```bash
python -m src.evaluation.run_retrieval_eval --system baseline_dense
python -m src.evaluation.run_retrieval_eval --system bm25_only
python -m src.evaluation.run_retrieval_eval --system hybrid
python -m src.evaluation.run_retrieval_eval --system dense_reranked
python -m src.evaluation.run_retrieval_eval --system hybrid_reranked
python -m src.evaluation.run_retrieval_eval --system dense_reranked_ml
python -m src.evaluation.run_retrieval_eval --system hybrid_reranked_ml
```

E5-large config ile (ayrı embedding/FAISS indeksi gerekir):
```bash
python -m src.retrieval.embedder --config configs/retrieval_config_e5large.yaml
python -m src.retrieval.vector_store --config configs/retrieval_config_e5large.yaml
python -m src.evaluation.run_retrieval_eval --system e5large_dense --config configs/retrieval_config_e5large.yaml
python -m src.evaluation.run_retrieval_eval --system e5large_reranked_ml --config configs/retrieval_config_e5large.yaml
```

### Sistem Karşılaştırma (Faz 2)
```bash
python -m src.evaluation.run_comparison
```

## Mevcut Durum

- **Faz 1** (İbo): Tamamlandı — Baseline RAG pipeline çalışıyor
- **Faz 2** (Deniz): Tamamlandı — Benchmark, evaluation altyapısı, 9 retrieval sistemi karşılaştırması
- **Faz 3** (İbo): Bekliyor — Reranker fine-tuning, LLM fine-tuning, tam optimize sistem
- **Faz 4** (Deniz): Bekliyor — QA evaluation, hata analizi, rapor, sunum

## En İyi Retrieval Sonuçları

| Sistem | MRR | Recall@5 | Recall@10 | Hit@5 |
|--------|-----|----------|-----------|-------|
| Dense Baseline (e5-base) | 0.5896 | 0.7436 | 0.8229 | 0.7829 |
| **E5-Large + ML Reranker** | **0.6644** | **0.7891** | **0.8857** | **0.8514** |

Baseline'a göre iyileşme: MRR +12.7%, Recall@5 +6.1%, Recall@10 +7.6%, Hit@5 +8.8%

## Donanım

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- PyTorch: 2.11.0+cu128
- CUDA: 13.1
