# Teslim Checklist

Bu dosya final teslimde neyin yükleneceğini ve hangi linklerin paylaşılacağını netleştirmek için hazırlanmıştır.

## Önerilen Dosya İsimleri

1. Ana rapor PDF'i: `CENG493_Turkish_Legal_RAG_Final_Report.pdf`
2. Outputs artifact paketi: `CENG493_Turkish_Legal_RAG_outputs_artifacts.rar`
3. Data paketi: `CENG493_Turkish_Legal_RAG_data.rar`
4. Tam proje paketi: `CENG493_Turkish_Legal_RAG_full_project.rar`

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
- İngilizce final rapor PDF'i: `CENG493_Turkish_Legal_RAG_Final_Report.pdf`
- İngilizce demo rehberi: `DEMO_GUIDE.md`
- Teslim checklist'i: `SUBMISSION_CHECKLIST.md`

Repository'ye dahil edilmeyecek yerel notlar:

- `RUNBOOK.md`
- `phase2_rapor.md`
- `phase3_rapor.md`
- Kişisel PDF veya ara progress notları

## Ayrıca Yüklenecek Outputs Artifact Paketi

`outputs/` klasörü `.gitignore` kapsamında olduğu için ayrıca RAR/ZIP olarak yüklenmelidir. Önerilen isim:

`CENG493_Turkish_Legal_RAG_outputs_artifacts.rar`

İçermesi gerekenler:

- `outputs/evaluation/`
- `outputs/qa_eval/`
- `outputs/sft_qlora/final/`
- `outputs/final_report.md`

Bu paket özellikle QLoRA adapter ve önceki evaluation sonuçları için önemlidir.

## Ayrıca Yüklenecek Data Paketi

`data/` klasörü repoda benchmark/SFT dosyalarını içerir; raw ve processed artifact'ler `.gitignore` kapsamında olduğundan tam veri paketini ayrıca yüklemek iyi olur. Önerilen isim:

`CENG493_Turkish_Legal_RAG_data.rar`

İçermesi gerekenler:

- `data/benchmark/`
- `data/sft/`
- `data/raw/`
- `data/processed/`

## Tam Proje Paketi

Hoca GitHub yerine doğrudan dosyadan kurmak isterse tam proje klasörü ayrıca RAR yapılabilir. Önerilen isim:

`CENG493_Turkish_Legal_RAG_full_project.rar`

Tam proje paketine `.venv/`, `venv/`, `__pycache__/`, kişisel PDF'ler ve geçici dosyalar dahil edilmemelidir.

## Hoca Gereksinimleriyle Eşleşme

- Base RAG vs Fine-tuned RAG karşılaştırması: `REPORT.md`
- Gold benchmark: `data/benchmark/gold_benchmark*.jsonl`
- Ablation study: `REPORT.md`
- Custom PDF doküman koleksiyonu: `scripts/prepare_custom_pdfs.py` ve `README.md`
- Custom benchmark testi: `src/evaluation/run_qa_eval.py --benchmark ...`

## Önerilen Aktivite Yüklemesi

1. GitHub repository linki.
2. İngilizce final rapor PDF'i: `CENG493_Turkish_Legal_RAG_Final_Report.pdf`.
3. Outputs artifact RAR/linki: `CENG493_Turkish_Legal_RAG_outputs_artifacts.rar`.
4. Data RAR/linki: `CENG493_Turkish_Legal_RAG_data.rar`.
5. Tam proje RAR/linki: `CENG493_Turkish_Legal_RAG_full_project.rar`.
6. Kısa not: Custom PDF ve custom benchmark kullanım komutlarının README ve REPORT içinde bulunduğu belirtilmeli.
