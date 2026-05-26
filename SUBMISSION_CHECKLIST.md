# Teslim Checklist

Bu dosya final teslimde neyin yükleneceğini ve hangi linklerin paylaşılacağını netleştirmek için hazırlanmıştır.

## Ana Linkler

- GitHub repository linki: `https://github.com/SpeedyV5/turkish-legal-rag`
- Final commit linki: GitHub'a push sonrası ilgili commit URL'i
- Artifact linki: `outputs/` klasörü büyük/yerel artifact içerdiği için GitHub yerine aktiviteye ZIP olarak veya Drive/OneDrive linkiyle yüklenmelidir

## GitHub Repository İçeriği

Repository'de bulunması gerekenler:

- Kaynak kodlar: `src/`, `scripts/`, `configs/`
- Benchmark ve SFT veri dosyaları: `data/benchmark/`, `data/sft/`
- Kurulum bağımlılıkları: `requirements.txt`
- Türkçe kullanım dokümantasyonu: `README.md`
- İngilizce final rapor: `REPORT.md`
- İngilizce demo rehberi: `DEMO_GUIDE.md`
- Teslim checklist'i: `SUBMISSION_CHECKLIST.md`

Repository'ye dahil edilmeyecek yerel notlar:

- `RUNBOOK.md`
- `phase2_rapor.md`
- `phase3_rapor.md`
- Kişisel PDF veya ara progress notları

## Ayrıca Yüklenecek Artifact Paketi

`outputs/` klasörü `.gitignore` kapsamında olduğu için ayrıca ZIP olarak yüklenmelidir:

- `outputs/evaluation/`
- `outputs/qa_eval/`
- `outputs/sft_qlora/final/`
- `outputs/final_report.md`

Bu paket özellikle QLoRA adapter ve önceki evaluation sonuçları için önemlidir.

## Hoca Gereksinimleriyle Eşleşme

- Base RAG vs Fine-tuned RAG karşılaştırması: `REPORT.md`
- Gold benchmark: `data/benchmark/gold_benchmark*.jsonl`
- Ablation study: `REPORT.md`
- Custom PDF doküman koleksiyonu: `scripts/prepare_custom_pdfs.py` ve `README.md`
- Custom benchmark testi: `src/evaluation/run_qa_eval.py --benchmark ...`

## Önerilen Aktivite Yüklemesi

1. GitHub repository linki.
2. İngilizce final rapor dosyası: `REPORT.md` veya PDF'e çevrilmiş hali.
3. Artifact ZIP veya link: `outputs/` içeriği.
4. Demo rehberi: `DEMO_GUIDE.md`.
5. Kısa not: Custom PDF ve custom benchmark kullanım komutlarının README içinde bulunduğu belirtilmeli.
