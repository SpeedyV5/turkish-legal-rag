# Custom PDF Test Rehberi (Hoca / Değerlendirici)

Bu rehber, sunumdan **sonra** kendi PDF dokümanlarınızla sistemi test etmeniz içindir. Canlı sunum sırasında bu adımları çalıştırmanız gerekmez.

## Ne sağlanıyor?

- Kendi PDF klasörünüzü sisteme alıp aynı RAG pipeline ile soru-cevap testi
- İsteğe bağlı custom benchmark JSONL ile otomatik değerlendirme
- Orijinal 7 kanun korpusuna geri dönüş (`CENG493_Turkish_Legal_RAG_data.zip` ile)

**Not:** Web tabanlı dosya yükleme paneli yoktur. Test ortamı bilinçli olarak mevcut corpus build scriptlerini yeniden kullanan **CLI tabanlı** bir akıştır.

## Hızlı başlangıç

### 1) PDF'leri hazırlayın

**PDF'leri proje içine değil, istediğiniz herhangi bir klasöre koyun.** Script bu klasörün yolunu `--input-dir` ile alır; dosyaları otomatik olarak projeye kopyalar.

Örnek (Windows):

```
C:\Users\Deniz\Desktop\test_pdfs\
    kanun_a.pdf
    kanun_b.pdf
    yonetmelik.pdf
```

Örnek (proje dışında veya içinde fark etmez):

```
C:\test_pdfs\mevzuat.pdf
```

Sonra proje kökünden şunu çalıştırırsınız:

```bash
python scripts/prepare_custom_pdfs.py --input-dir C:/test_pdfs --reset
```

Script PDF'leri şuraya kopyalar (elle buraya koymanız gerekmez):

```
turkish-legal-rag/data/raw/mevzuat/pdfs/
```

- Tüm PDF'leri tek bir klasöre koyun; klasör yolu sizin seçiminizdir
- Yalnızca klasörün **doğrudan içindeki** `*.pdf` dosyaları okunur; alt klasörler taranmaz
- Her PDF'in tüm sayfaları metin olarak çıkarılır (PyMuPDF)
- Taranmış/görüntü-only PDF'lerde OCR yoktur; metin boş kalırsa o dosya atlanır

### 2) Custom korpusu yükleyin ve index'i kurun

```bash
python scripts/prepare_custom_pdfs.py --input-dir path/to/custom_pdfs --reset
python -m src.corpus.build_registry
python -m src.corpus.register_pdfs
python -m src.retrieval.chunking
python -m src.retrieval.embedder --config configs/retrieval_config_e5large.yaml
python -m src.retrieval.vector_store --config configs/retrieval_config_e5large.yaml
```

`--reset` mevcut `data/raw/mevzuat/pdfs/` içindeki PDF'leri siler ve yalnızca verdiğiniz dosyaları kullanır.

### 3) Test edin

İnteraktif soru-cevap:

```bash
python -m src.pipeline.rag_pipeline --lora-adapter outputs/sft_qlora/final --demo-safe
```

Custom benchmark ile ölçüm (`gold_benchmark.jsonl` ile aynı şema):

```bash
python -m src.evaluation.run_qa_eval --benchmark path/to/custom_benchmark.jsonl --system e5large_reranked_bge --lora-adapter outputs/sft_qlora/final --output-tag custom
```

## Veri seti bozulur mu?

| Değişen | Değişmeyen |
| --- | --- |
| `data/raw/mevzuat/pdfs/` | `data/benchmark/` (175 soruluk gold benchmark) |
| `data/raw/mevzuat/seed_urls.csv` | `data/sft/` (QLoRA eğitim verisi) |
| `data/processed/corpus/*` (index, embedding, chunks) | `outputs/sft_qlora/final/` (LoRA adapter) |

Custom PDF akışı **birleştirme değil, tam değiştirmedir**: retrieval artık yalnızca yüklediğiniz PDF'lerden arar. Orijinal Anayasa/TCK soruları custom PDF'lerde yoksa anlamsız cevaplar üretebilir.

## Orijinal 7 kanuna geri dönüş

Custom testten sonra varsayılan korpusa dönmek için teslim paketindeki `CENG493_Turkish_Legal_RAG_data.zip` dosyasını açın ve şu klasörleri proje köküne geri kopyalayın:

- `data/raw/`
- `data/processed/`

Windows örneği (PowerShell):

```powershell
Expand-Archive -Path CENG493_Turkish_Legal_RAG_data.zip -DestinationPath . -Force
```

Ardından index yeniden kurmaya gerek kalmadan varsayılan pipeline çalışır.

## Sınırlamalar

- QLoRA modeli orijinal 7 kanun üzerinde eğitildi; custom PDF'lerde retrieval çalışır ama üretim kalitesi dokümana göre değişebilir
- Index yeniden kurulumu (özellikle embedding) birkaç dakika sürebilir
- `--demo-safe` yalnızca canlı demo/sunum formatıdır; resmi benchmark sonuçları `run_qa_eval` çıktısındandır

## İlgili dosyalar

- [DEMO_GUIDE.md](DEMO_GUIDE.md) — sunum akışı ve demo soruları
- [README.md](README.md) — genel proje kullanımı
- [REPORT.md](REPORT.md) — İngilizce final rapor (Bölüm 11: Custom Corpus)
- [scripts/prepare_custom_pdfs.py](scripts/prepare_custom_pdfs.py) — PDF hazırlama scripti
