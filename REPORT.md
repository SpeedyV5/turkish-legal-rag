Turkish Legal RAG — Proje Raporu
1. Başlangıç Kararı: Neden Lokal Baseline?

Projeye lokal bir baseline ile başlamayı tercih ettik. Bunun temel nedeni, ödev kapsamında deneylerin tekrarlanabilir olması, hiperparametrelerin dokümante edilmesi, GPU kullanımının raporlanması ve embedding/reranker/LLM katmanlarında yapılan iyileştirmelerin ablation deneyleriyle karşılaştırılmasının beklenmesidir. Lokal bir başlangıç, API maliyeti ve erişim bağımlılıklarını azaltırken sistemin retrieval, prompting ve fine-tuning bileşenlerini daha kontrollü biçimde geliştirmemizi sağlar. API tabanlı modeller ise ilerleyen aşamada opsiyonel bir karşılaştırma veya demo backend'i olarak eklenebilir.

2. Chunking Kararı: Neden Hibrit Madde-Aware Chunking?

Retrieval katmanında hibrit madde-aware chunking yaklaşımını tercih ettik. Bunun temel nedeni, Türk hukuk metinlerinin doğal yapısının madde temelli olması ve projenin grounded, kaynak destekli, citation-consistent cevaplar üretmeyi hedeflemesidir. Sabit boyutlu klasik chunking yöntemleri uygulanması kolay olsa da madde sınırlarını bozarak aynı hukuki hükmün birden fazla parçaya dağılmasına neden olabilir. Bu durum retrieval doğruluğunu, faithfulness değerlendirmesini ve citation accuracy'yi olumsuz etkileyebilir. Buna karşılık tamamen madde bazlı bir yaklaşım da bazı maddelerin çok kısa, bazılarının ise aşırı uzun olması nedeniyle dengeli bir retrieval davranışı üretmeyebilir. Bu nedenle önce metni madde sınırlarına göre ayıran, ardından çok uzun maddeleri kontrollü alt parçalara bölen hibrit bir strateji benimsedik. Böylece hem hukuk metninin yapısal bütünlüğünü korumayı hem de embedding ve retrieval aşamalarında daha dengeli ve kullanılabilir chunk'lar üretmeyi amaçladık. Bu tercih aynı zamanda ileride reranker, faithfulness analizi ve kaynak gösterimi açısından daha güçlü bir temel sağlayacaktır.

3. Embedding Başlangıç Kararı: Neden multilingual-e5-base?

İlk embedding modeli olarak intfloat/multilingual-e5-base kullanmayı planladık. Bu tercihin temel nedeni, modelin çok dilli retrieval senaryoları için uygun olması ve baseline aşamasında dengeli bir başlangıç noktası sunmasıdır. Modelin 12 katmanlı ve 768 boyutlu bir embedding yapısına sahip olması, yerel GPU kaynakları üzerinde daha yönetilebilir bir deney ortamı sağlamaktadır. Başlangıç aşamasında amacımız, ilk retrieval sistemini mümkün olduğunca kontrollü, tekrarlanabilir ve hesaplama maliyeti açısından dengeli bir şekilde ayağa kaldırmaktır. Bu nedenle, daha güçlü ve esnek alternatifler mevcut olsa da, ilk basamakta daha hafif ve güvenli bir seçenek tercih edilmiştir. BAAI/bge-m3 güçlü ve çok amaçlı bir retrieval modeli olmasına rağmen, baseline aşamasında gereğinden fazla karmaşıklık ekleyebileceği ve ilk kurulum maliyetini artırabileceği için başlangıç noktası olarak seçilmemiştir. sentence-transformers/paraphrase-multilingual-mpnet-base-v2 ise çok dilli anlamsal benzerlik açısından kullanışlı bir alternatif olsa da, retrieval odaklı başlangıç senaryosunda multilingual-e5-base kadar doğrudan ve güncel bir tercih olarak değerlendirilmemiştir. Bu nedenle multilingual-e5-base, ilk baseline retrieval sistemini hızlı ve düzenli biçimde kurmak, daha sonra ise diğer embedding modelleriyle anlamlı karşılaştırmalar yapabilmek için referans model olarak belirlenmiştir.

4. Generator Başlangıç Kararı: Neden Qwen2.5-7B-Instruct?

İlk baseline generation modeli olarak Qwen/Qwen2.5-7B-Instruct kullanmayı planladık. Bu tercihin temel nedeni, modelin çok dilli kullanım senaryolarını desteklemesi ve uzun bağlamlarla çalışabilmesidir. Model kartında 29'dan fazla dil desteği ve 131,072 token bağlam uzunluğu belirtilmektedir. Bu özellikler, Türkçe hukuk sorularında retrieval katmanından gelen birden fazla mevzuat parçasını aynı bağlam içinde işleyebilmek açısından önemli görülmüştür. Ayrıca modelin transformers ekosistemiyle doğrudan kullanılabilmesi, baseline aşamasında daha hızlı ve kontrollü bir kurulum sağlamaktadır.

Alternatif olarak mistralai/Mistral-7B-Instruct-v0.3 ve meta-llama/Llama-3.1-8B-Instruct modelleri de değerlendirilmiştir. Mistral-7B-Instruct-v0.3 güçlü bir instruct model olup özellikle function calling desteğiyle öne çıkmaktadır; ancak ilk baseline için Türkçe ve çok dilli retrieval-sonrası cevap üretimi açısından Qwen kadar doğrudan bir tercih olarak görülmemiştir. Llama-3.1-8B-Instruct ise güçlü bir multilingual dialogue modeli olmasına rağmen, model kartında desteklenen diller arasında Türkçe açık biçimde listelenmemekte ve erişim/lisans yapısı da daha kısıtlayıcı bir yapı sunmaktadır. Bu nedenle ilk aşamada daha dengeli, erişilebilir ve Türkçe kullanım için daha uygun görünen Qwen2.5-7B-Instruct referans generator olarak seçilmiştir.

5. Generator Çalıştırma Stratejisi: Neden Quantized Local Inference?

Baseline generation katmanında yerel çalıştırma hedefi korunurken, donanım kısıtları nedeniyle quantized inference yaklaşımı tercih edilmiştir. Özellikle 7B ölçeğindeki instruct modellerin standart hassasiyetle çalıştırılması, orta seviye GPU belleği üzerinde kararsız veya maliyetli olabilmektedir. Bu nedenle ilk aşamada 4-bit quantization kullanarak modeli daha yönetilebilir bellek tüketimiyle çalıştırmak hedeflenmiştir. Bu yaklaşım, kalite ile hesaplama maliyeti arasında dengeli bir başlangıç noktası sağlamakta ve baseline sistemin yerel olarak uçtan uca çalıştırılmasını kolaylaştırmaktadır.

6. Generator Revizyonu: Neden 7B'den 3B'ye Geçildi?

İlk denemede Qwen2.5-7B-Instruct modeli ile yerel quantized inference hedeflenmiştir. Ancak uygulama sırasında modelin bir kısmının GPU belleğine sığmaması nedeniyle yükleme aşamasında cihaz dağıtımı hatası alınmıştır. Hugging Face Accelerate dokümantasyonuna göre device_map="auto" kullanıldığında katmanlar önce GPU'ya, gerekirse CPU'ya ve diske dağıtılabilmektedir. Bu durum orta seviye GPU belleğinde 7B ölçekli modeller için pratik kısıtlar oluşturmuştur. Bu nedenle baseline sistemi daha kararlı ve tekrarlanabilir biçimde çalıştırabilmek için daha hafif bir alternatif olan Qwen2.5-3B-Instruct modeline geçilmesi tercih edilmiştir. Bu model çok dilli destek sunmaya devam ederken daha düşük parametre sayısı sayesinde yerel inference açısından daha uygun bir başlangıç noktası oluşturmaktadır.

---

7. Faz 2: Gold Benchmark Set Oluşturma

Faz 2'nin ilk adımı olarak projenin ölçülebilir hale getirilmesi için 175 soruluk bir gold benchmark test seti oluşturulmuştur. Bu set korpustaki 7 kanunun tamamını kapsamakta olup her soru için doğrulanmış cevap metni, ilgili kanun (doc_id), ilgili madde numaraları (relevant_articles), soru tipi (definition/list/factual/procedural/yes_no) ve zorluk seviyesi (easy/medium/hard) bilgileri tutulmaktadır.

Benchmark dağılımı:
- Kanun bazında: Anayasa 30, TCK 30, CMK 22, TMK 28, TBK 25, HMK 22, IYUK 18
- Soru tipi bazında: definition 60, list 61, factual 39, procedural 14, yes_no 1
- Zorluk bazında: easy 62, medium 101, hard 12

Benchmark neden bu şekilde tasarlandı: Projede istenen 150-300 soru aralığına 175 soru ile girilmiştir. Dağılımın dengelenmesinde her kanunun korpustaki hacmi ve hukuk pratiğindeki soru çeşitliliği göz önünde bulundurulmuştur. Soru tipleri bilinçli olarak çeşitlendirilmiştir çünkü Faz 1'deki bilinen problemlerden biri liste sorularında recall düşüklüğüdür; bu tip soruların yeterli sayıda temsil edilmesi analiz için kritiktir.

Benchmark dosyası data/benchmark/gold_benchmark.jsonl yolunda saklanmakta ve git tarafından izlenmektedir (gitignore kapsamı dışında).

8. Faz 2: Retrieval Evaluation Altyapısı

Benchmark setinin oluşturulmasının ardından retrieval kalitesini ölçen bir evaluation altyapısı kurulmuştur. Bu altyapı aşağıdaki metrikleri hesaplamaktadır:

- Recall@k (k=1,3,5,10): İlk k sonuç içinde bulunan ilgili maddelerin oranı
- Precision@k: İlk k sonuçtan kaçının ilgili olduğu
- MRR (Mean Reciprocal Rank): İlk doğru sonucun sıralamasının tersi
- nDCG@k (Normalized Discounted Cumulative Gain): Sıralama kalitesi
- Hit@k: İlk k sonuçta en az bir doğru sonuç var mı
- F1 (token-level): Üretilen cevap ile referans cevabın token bazlı benzerliği
- Exact Match: Normalleştirilmiş tam eşleşme

Evaluation sistemi her sorgu için madde eşleştirmesi yapmaktadır: benchmark'taki gold article referansları (örn. "Madde 81") ile retriever'ın getirdiği chunk'ların article_ref alanları karşılaştırılmaktadır. Eşleşme doc_id (hangi kanun) ve madde numarası (sayısal kısım) bazında case-insensitive yapılmaktadır.

Altyapı ayrıca sonuçları soru tipine, zorluk seviyesine ve kanuna göre kırarak raporlamakta; zero recall vakaları ayrıca listelenmektedir.

9. Faz 2: Retrieval İyileştirme Deneyleri

Baseline dense retriever'ın yanı sıra üç alternatif retrieval stratejisi implement edilmiş ve test edilmiştir:

A) BM25 Retriever
Klasik sparse retrieval yöntemi olan BM25 (Okapi BM25) implement edilmiştir. Türkçe metin tokenizasyonu için özel bir fonksiyon yazılmıştır: küçük harf dönüşümü, noktalama temizliği ve Türkçe stop word filtreleme uygulanmaktadır. rank-bm25 kütüphanesi kullanılmıştır.

B) Hybrid Retriever (Dense + BM25)
Dense retrieval (FAISS) ve BM25 sonuçlarını birleştiren hybrid bir retriever geliştirilmiştir. İki fusion stratejisi implement edilmiştir:
- Reciprocal Rank Fusion (RRF): Her iki sistemdeki sıralama bazlı birleştirme
- Weighted Score Fusion: Min-max normalizasyon sonrası ağırlıklı skor toplama

İlk deneyde ağırlıklar 0.60 dense / 0.40 BM25 olarak ayarlanmış ancak BM25'in Türkçe'deki zayıf performansının dense sonuçları aşağı çektiği görülmüştür. Ağırlıklar 0.85 dense / 0.15 BM25 olarak revize edildikten sonra Recall@10'da baseline'ı geçen sonuçlar elde edilmiştir.

C) Cross-Encoder Reranker (İngilizce — Başarısız)
sentence-transformers kütüphanesinin CrossEncoder sınıfı ile cross-encoder/ms-marco-MiniLM-L-6-v2 modeli kullanılarak reranking implement edilmiştir. Retriever'dan gelen aday sonuçlar (query, passage) çiftleri olarak cross-encoder'a verilmekte ve yeniden puanlanmaktadır.

Ancak bu model İngilizce MS MARCO verisi üzerinde eğitilmiş olduğundan Türkçe hukuk metinlerinde beklenen iyileşmeyi sağlayamamıştır. Aksine, Türkçe metinleri yanlış puanlayarak doğru sonuçları alt sıralara düşürmüştür.

D) Çok Dilli Cross-Encoder Reranker (mMARCO — Başarılı)
İngilizce reranker'ın başarısızlığı üzerine çok dilli bir alternatif denenmiştir: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1. Bu model, MS MARCO veri setinin çok dilli çevirisi (mMARCO) üzerinde eğitilmiş olup Türkçe dahil birçok dili desteklemektedir.

Sonuçlar İngilizce reranker'ın tam tersi olmuştur: çok dilli reranker tüm ana metriklerde dense baseline'ı geçmiştir. Özellikle dense + multilingual reranker kombinasyonunda MRR 0.5896'dan 0.6568'e, Recall@10 0.8229'dan 0.8457'ye yükselmiştir. Bu, doğru dilde eğitilmiş bir cross-encoder'ın retrieval sıralamasını anlamlı biçimde iyileştirebildiğini kanıtlamaktadır.

E) Alternatif Embedding Modeli (E5-Large)
Baseline'da kullanılan intfloat/multilingual-e5-base (278M parametre, 768 boyut) modeline alternatif olarak intfloat/multilingual-e5-large (560M parametre, 1024 boyut) denenmiştir. Bu deney için tüm corpus yeniden embed edilmiş ve ayrı bir FAISS indeksi oluşturulmuştur.

E5-large tek başına (reranker olmadan) bile tüm önceki sistemleri geride bırakmıştır: MRR 0.6773, Recall@10 0.8686, nDCG@10 0.7098. E5-large + çok dilli reranker kombinasyonu ise Recall@5'te 0.7891, Recall@10'da 0.8857 ile projenin en iyi sonuçlarını vermiştir.

10. Faz 2: Retrieval Evaluation Sonuçları

175 soruluk gold benchmark üzerinde 9 farklı retrieval sistemi değerlendirilmiştir.

Ana karşılaştırma tablosu:

| Sistem                        | MRR    | Recall@1 | Recall@5 | Recall@10 | nDCG@5 | nDCG@10 | Hit@5  |
|-------------------------------|--------|----------|----------|-----------|--------|---------|--------|
| Dense Baseline (e5-base)      | 0.5896 | 0.4045   | 0.7436   | 0.8229    | 0.6071 | 0.6371  | 0.7829 |
| BM25 Only                     | 0.3208 | 0.1698   | 0.4567   | 0.6000    | 0.3348 | 0.3847  | 0.4914 |
| Hybrid (0.85/0.15)            | 0.5510 | 0.3440   | 0.7098   | 0.8343    | 0.5640 | 0.6092  | 0.7543 |
| Dense + EN Reranker           | 0.4139 | 0.2233   | 0.6217   | 0.7600    | 0.4420 | 0.4919  | 0.6629 |
| Hybrid + EN Reranker          | 0.4122 | 0.2264   | 0.5943   | 0.7600    | 0.4309 | 0.4905  | 0.6343 |
| Dense + ML Reranker           | 0.6568 | 0.5016   | 0.7522   | 0.8457    | 0.6587 | 0.6940  | 0.7943 |
| Hybrid + ML Reranker          | 0.6587 | 0.4954   | 0.7512   | 0.8514    | 0.6585 | 0.6960  | 0.7943 |
| **E5-Large Dense**            | **0.6773** | 0.4842| 0.7534   | 0.8686    | 0.6692 | **0.7098** | 0.7943 |
| **E5-Large + ML Reranker**    | 0.6644 | **0.4821**| **0.7891** | **0.8857** | **0.6699** | 0.7073 | **0.8514** |

Not: "EN Reranker" = cross-encoder/ms-marco-MiniLM-L-6-v2 (İngilizce), "ML Reranker" = cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (çok dilli).

Soru tipine göre Recall@5 (seçili sistemler):

| Soru Tipi   | Dense Baseline | Hybrid | Dense+ML Reranker | E5-Large | E5-Large+ML Reranker |
|-------------|---------------|--------|-------------------|----------|----------------------|
| definition  | 0.7646        | 0.7146 | 0.7083            | 0.7771   | 0.7572               |
| factual     | 0.9679        | 0.9423 | 0.9103            | 0.9124   | 0.8974               |
| list        | 0.5929        | 0.6025 | 0.7044            | 0.6661   | **0.7650**           |
| procedural  | 0.6667        | 0.4881 | 0.6905            | 0.5714   | 0.7143               |

Zorluk seviyesine göre Recall@5 (seçili sistemler):

| Zorluk | Dense Baseline | Hybrid | Dense+ML Reranker | E5-Large | E5-Large+ML Reranker |
|--------|---------------|--------|-------------------|----------|----------------------|
| easy   | 0.8831        | 0.8185 | 0.8226            | 0.8911   | 0.8629               |
| medium | 0.6819        | 0.6679 | 0.7175            | 0.6725   | **0.7502**           |
| hard   | 0.5417        | 0.5000 | 0.6806            | 0.7222   | **0.7361**           |

11. Faz 2: Sonuçların Analizi ve Bulgular

A) Dense baseline güçlü bir başlangıç noktasıdır
Dense baseline (intfloat/multilingual-e5-base + FAISS) ilk 5 sistemlik karşılaştırmada MRR ve Recall@5'te en iyi sonucu vermiştir. Factual sorularda Recall@5 = 0.97 ile neredeyse mükemmel performans sergilenmektedir.

B) BM25 tek başına yetersizdir
BM25'in Recall@5 = 0.46 ile en zayıf performansı göstermesi beklenen bir sonuçtur. Türkçe'nin zengin morfolojik yapısı (eklemeli dil) basit whitespace tokenization ile iyi eşleşme yapılmasını zorlaştırmaktadır. Örneğin "boşanma" kelimesini arayan BM25, "boşanmanın", "boşanmada", "boşanmaya" gibi çekimli formlarla doğrudan eşleşememektedir. Türkçe stemming veya lemmatization entegrasyonu bu sorunu kısmen çözebilir.

C) Hybrid retrieval Recall@10'da avantaj sağlamaktadır
Hybrid retriever (0.85 dense / 0.15 BM25, RRF fusion) Recall@10'da baseline'ı geçmiştir (0.8343 vs 0.8229). Ancak Recall@5 ve MRR'de baseline'ın gerisinde kalmıştır çünkü BM25'in düşük kaliteli sonuçları üst sıralara gürültü eklemektedir.

D) İngilizce cross-encoder reranker Türkçe'de zararlıdır
cross-encoder/ms-marco-MiniLM-L-6-v2 modeli İngilizce web metinleri üzerinde eğitilmiş olup Türkçe hukuk metinlerini doğru puanlayamamaktadır. Hem dense hem hybrid sonuçları rerank edildikten sonra tüm metriklerde ciddi düşüş yaşanmıştır (MRR 0.59 → 0.41). Bu, dil uyumsuzluğunun reranking'de kritik olduğunu kanıtlamaktadır.

E) Çok dilli reranker (mMARCO) doğru yaklaşımdır
İngilizce reranker'ın başarısızlığı üzerine denenen cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 modeli tüm metriklerde baseline'ı geçmiştir. Dense + ML Reranker kombinasyonunda MRR 0.59 → 0.66 (+11.4%), Recall@10 0.82 → 0.85 (+2.8%), nDCG@10 0.64 → 0.69 (+8.9%) artış sağlanmıştır. Bu, reranker katmanının iyileştirme potansiyelini göstermektedir; ancak doğru dilde eğitilmiş model kullanmak zorunludur.

F) E5-Large daha güçlü bir embedding sağlamaktadır
intfloat/multilingual-e5-large (560M parametre, 1024 boyut) modeli, e5-base'e (278M parametre, 768 boyut) kıyasla tek başına bile çok önemli iyileşme sağlamıştır: MRR 0.59 → 0.68 (+14.9%), Recall@10 0.82 → 0.87 (+5.6%), nDCG@10 0.64 → 0.71 (+11.4%). Daha büyük embedding modeli daha zengin semantik temsiller üreterek özellikle hard sorularda (0.54 → 0.72) büyük fark yaratmıştır.

G) En iyi sistem: E5-Large + Çok Dilli Reranker
Projenin en iyi retrieval sistemi e5large_reranked_ml olarak belirlenmiştir. Bu sistem baseline'a kıyasla: Recall@5 0.74 → 0.79 (+6.1%), Recall@10 0.82 → 0.89 (+7.6%), Hit@5 0.78 → 0.85 (+8.8%) iyileşme sağlamıştır. Ayrıca zero recall@5 vakası sayısı 36'dan 26'ya düşmüştür. Özellikle kritik iyileşmeler: list sorularında Recall@5 0.59 → 0.77 (+29%), hard sorularda 0.54 → 0.74 (+36%).

H) Liste ve hard sorulardaki iyileşme kayda değerdir
Faz 1'deki bilinen problem olan list tipi sorulardaki düşük recall, e5-large + ML reranker ile 0.59'dan 0.77'ye çıkarılmıştır. Hard zorluk seviyesindeki sorularda ise 0.54'ten 0.74'e yükselmiştir. Bu iyileşmeler embedding kalitesi ve doğru dilde reranking'in birlikte etkisidir.

12. GPU Kullanımı ve Performans Notu

Faz 2 çalışmaları sırasında PyTorch'un CPU versiyonunun (2.11.0+cpu) yüklü olduğu tespit edilmiştir. Bu durum, embedding model inference ve cross-encoder reranking işlemlerinin çok yavaş çalışmasına neden olmuştur. CUDA destekli PyTorch (2.11.0+cu128) yüklendikten sonra NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) ile performans dramatik biçimde artmıştır:

- Dense baseline evaluation: 12.8s → 3.0s (4.3x hızlanma)
- Hybrid evaluation: 17.7s → 4.1s (4.3x hızlanma)
- Hybrid + reranker evaluation: 7+ dakika (tamamlanamadı) → 46.2s

Faz 1'de İbo'nun da benzer bir durumla karşılaştığı ve Intel ekran kartından NVIDIA GPU'ya geçişte dramatik hızlanma yaşandığı raporlanmıştır.

---

13. Faz Durumu: Yapılanlar, Kalanlar ve Sonraki Adımlar

FAZ 1 (İbo — Tamamlandı):
İbo'nun Faz 1'de yaptıkları:
✓ Repo yapısı ve konfigürasyon sistemi
✓ PDF indirme ve metin çıkarma pipeline'ı (7 kanun)
✓ Corpus registry ve doküman yönetimi
✓ Hybrid article-aware chunking (3579 chunk)
✓ intfloat/multilingual-e5-base ile embedding (shape: 3579×768)
✓ FAISS IndexFlatIP vector store
✓ Qwen2.5-3B-Instruct 4-bit quantized local inference
✓ Soru tipi tespiti ve prompt builder
✓ Uçtan uca interaktif RAG pipeline
✓ HuggingFace veri seti inceleme (Renicames/turkish-law-chatbot)

Faz 1'de eksik kalan veya bilinen sorunlar:
- Retriever'daki keyword bonus sistemi hardcoded (sadece birkaç sorgu için)
- Liste sorularında recall düşük
- Retrieval her zaman en doğru maddeyi ilk sıraya koyamıyor
- Prompt ile iyileşme olsa da asıl darboğaz retrieval sıralaması

FAZ 2 (Deniz — Tamamlandı):
Bu fazda yapılanlar:
✓ 175 soruluk gold benchmark set (7 kanun, 5 soru tipi, 3 zorluk seviyesi)
✓ Benchmark generation ve validation script'i
✓ Retrieval evaluation altyapısı (Recall@k, MRR, nDCG, Precision@k, Hit@k, F1, EM)
✓ Soru tipi / zorluk / kanun bazında kırılım analizi
✓ BM25 retriever implementasyonu (Türkçe tokenization ile)
✓ Hybrid retriever (Dense + BM25, RRF ve weighted fusion)
✓ Cross-encoder reranker implementasyonu (İngilizce + çok dilli)
✓ İngilizce reranker'ın Türkçe'de zararlı olduğunun tespiti
✓ Çok dilli reranker denemesi (mmarco-mMiniLMv2-L12-H384-v1) — başarılı
✓ Alternatif embedding modeli denemesi (multilingual-e5-large) — başarılı
✓ 9 farklı retrieval sistemin benchmark üzerinde karşılaştırması
✓ Zero recall analizi ve hata tespiti
✓ GPU/CUDA kurulum düzeltmesi (CPU → GPU geçişi)
✓ E5-large için ayrı embedding pipeline ve FAISS indeksi
✓ Karşılaştırma raporu ve sonuç tabloları

Faz 2'de planlanıp YAPILMAYANLAR (dürüst değerlendirme):
✗ Embedding domain adaptation / contrastive fine-tuning — 4GB VRAM ile fine-tuning yapılamadı; ancak alternatif olarak e5-large model denemesi yapıldı ve büyük iyileşme sağlandı
✗ Hard negative mining — Embedding fine-tuning'e bağımlı, mevcut VRAM ile uygulanamadı
✗ QA evaluation — Faz 3'e devredildi ve orada tamamlandı

Not: Faz 3 çalışmaları NVIDIA RTX 3070 Laptop GPU (8GB VRAM) üzerinde gerçekleştirilmiştir. Faz 2'deki 4GB VRAM (RTX 3050) kısıtı artık geçerli değildir.

---

8. Faz 3: Evaluation Bug Fix ve Benchmark Hazırlığı

8.1 Evaluation Bug Fix: build_relevant_chunk_ids

Faz 2'den devralınan evaluation kodunda kritik bir hata tespit edildi. Retrieval metrikleri hesaplanırken, gold relevant chunk ID'leri yalnızca soru bazında eşleştiriliyordu ve corpus-wide chunk→article eşlemesi yapılmıyordu. Bu durum özellikle birden fazla chunk'a sahip maddelerde recall'un yanlış hesaplanmasına yol açıyordu.

Düzeltme: Corpus metadata dosyasından (chunk_metadata.jsonl) tüm chunk→article eşlemesi yüklenerek, her sorunun gold article'larına ait tüm chunk ID'leri corpus genelinde doğru şekilde belirlendi. Ayrıca article-level recall metriği eklendi.

Sonuç: Bug fix sonrası 9 retrieval sistemi yeniden koşturuldu ve sonuçlar güncellendi. Düzeltme öncesi recall değerleri yanıltıcıydı; düzeltme sonrası sonuçlar güvenilir hale geldi.

8.2 Benchmark Train/Dev/Test Split

175 soruluk gold benchmark, stratified split ile üç parçaya ayrıldı:
- Train: 112 soru (SFT verisi için)
- Dev: 32 soru (prompt tuning ve hiperparametre seçimi için)
- Test: 31 soru (final değerlendirme için)

Stratifikasyon source_law, question_type ve difficulty alanlarına göre yapıldı.

8.3 Liste Tipi Gold Expansion

Liste tipi soruların gold relevant article alanları genişletildi. Best retriever (e5-large + multilingual reranker) ile her liste sorusu için ek aday maddeler üretildi, manuel inceleme sonrası uygun olanlar gold benchmark'a eklendi. Bu işlem suggest_list_gold_expansion.py ve apply_list_gold_expansion.py script'leri ile gerçekleştirildi.

---

9. Faz 3: QA Evaluation Methodology

9.1 Metrikler

QA evaluation pipeline'ı (src/evaluation/run_qa_eval.py) şu metrikleri hesaplar:

- **Exact Match (EM)**: Cevabın gold cevapla tamamen eşleşip eşleşmediği (Türkçe normalizasyon sonrası)
- **Token-level F1**: Cevap ve gold cevap arasındaki token düzeyinde precision, recall ve F1
- **Citation F1 / Precision / Recall**: Modelin ürettiği "Dayanak:" satırındaki madde referanslarının gold article'larla eşleşmesi
- **Citation Exact Match**: Dayanak'taki maddelerin gold ile tam eşleşme oranı
- **Has Dayanak**: Cevabın sonunda "Dayanak:" satırı olup olmadığı
- **Faithfulness (lexical proxy)**: Cevaptaki içerik tokenlarının retrieved context'te bulunma oranı

9.2 Citation Parsing

Model cevabının sonunda "Dayanak:" satırı aranır. Bu satırdan regex ile madde referansları (örn. "Madde 114", "md. 26/1") çıkarılır. Bu referanslar gold article listesiyle karşılaştırılarak citation precision, recall ve F1 hesaplanır.

9.3 Faithfulness Ölçümü

Faithfulness, rule-based lexical overlap ile ölçülmektedir. Cevaptaki anlamlı tokenlar (stop word'ler hariç) retrieved context'teki tokenlara karşı eşleştirilir. Bu metrik bir proxy'dir ve LLM-as-judge veya NLI tabanlı yöntemler kadar güçlü değildir, ancak halüsinasyon eğilimini yakalamak için yeterli bir sinyaldir.

Faithfulness < 0.5 olan cevaplar "yüksek halüsinasyon riski" olarak sınıflandırılmıştır.

BLEU ve ROUGE metrikleri eklenmemiştir. Hukuk metinlerinde cevap yapısı farklı olabildiğinden token-level F1 daha anlamlı bir karşılaştırma sağlamaktadır; ancak bu metrikler ileride eklenebilir.

---

10. Faz 3: Reranker Değişikliği

Faz 2'de en iyi reranker olarak mmarco-mMiniLMv2-L12-H384-v1 (çok dilli, 33M parametre) belirlenmiştir. Faz 3'te BAAI/bge-reranker-v2-m3 (çok dilli, 568M parametre) zero-shot olarak entegre edilmiş ve A/B karşılaştırması yapılmıştır.

bge-reranker-v2-m3 daha güncel ve güçlü bir modeldir. Pipeline varsayılanı olarak bge-reranker seçilmiştir. Reranker fine-tuning yapılmamıştır; zero-shot performans yeterli görülmüştür.

İngilizce reranker'ın Türkçe metinlerde zararlı olduğu bulgusu Faz 2'de doğrulanmış ve Faz 3'te de korunmuştur. Pipeline'da yalnızca çok dilli reranker kullanılmaktadır.

---

11. Faz 3: Prompt Tuning

Prompt builder (src/generation/prompt_builder.py) iteratif olarak iyileştirilmiştir. Temel değişiklikler:

- Citation discipline güçlendirildi: Model, cevap sonunda "Dayanak:" satırı ile kaynak maddelerini belirtmeye zorlandı
- Fallback cümlesi kaldırıldı: Önceki versiyonda context dışı sorularda üretilen gereksiz fallback cümlesi kaldırıldı
- max_new_tokens 160'tan 384'e çıkarıldı (truncation azaltmak için)

Dev set üzerinde 5 prompt versiyonu denendi. En iyi versiyon (v5) citation F1'de +0.24, faithfulness'ta +0.08 iyileşme sağladı (answer F1'de −0.07 marjinal düşüşle).

---

12. Faz 3: QLoRA SFT Fine-Tuning

12.1 Amaç

Qwen2.5-3B-Instruct modelini Türk hukuk soruları için ince ayar yaparak cevap kalitesi, citation doğruluğu ve faithfulness'ı artırmak.

12.2 Eğitim Verisi

Train split'teki 112 sorudan SFT verisi hazırlandı. Her örnek şu formatta:
- System: Türkçe hukuk asistanı system prompt'u
- User: Soru + retrieved context (gold article chunk'ları)
- Assistant: Gold cevap + "Dayanak:" satırı

Veri, corpus chunk metadata'sından article referans eşlemesi yapılarak üretildi. Madde referans normalizasyonu uygulandı (benchmark "Madde 114" vs corpus "MADDE 114-" formatı).

12.3 Eğitim Konfigürasyonu

| Parametre | Değer |
|-----------|-------|
| Base model | Qwen/Qwen2.5-3B-Instruct |
| Quantization | 4-bit NF4, double quant |
| Compute dtype | bfloat16 |
| LoRA rank (r) | 16 |
| LoRA alpha (α) | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Epochs | 3 |
| Batch size | 1 (effective 8 with grad accum) |
| Learning rate | 2e-4 |
| Scheduler | cosine |
| Warmup ratio | 0.06 |
| Max sequence length | 1024 |
| Optimizer | paged_adamw_8bit |
| Completion-only loss | Evet (yalnızca assistant tokenleri) |

12.4 Eğitim Detayları

- Donanım: NVIDIA RTX 3070 Laptop GPU, 8GB VRAM
- Süre: ~22 dakika, 42 step
- TRL 1.4 API (SFTConfig + SFTTrainer, peft_config doğrudan trainer'a verildi)
- Windows ortamında PYTHONUTF8=1 ayarı gerekli
- Adapter boyutu: ~60MB (outputs/sft_qlora/final/)

---

13. Faz 3: Ablation Zinciri ve Sonuçlar

13.1 Ablation Zinciri

Sistem kademeli olarak iyileştirilmiştir:

Adım 1 — Baseline: e5-base dense retrieval, untuned Qwen2.5-3B-Instruct
Adım 2 — Embedding model seçimi: e5-base → e5-large (MRR: 0.5896 → 0.6773, +14.9%)
Adım 3 — Reranker ekleme: bge-reranker-v2-m3 zero-shot (MRR: 0.6773 → 0.6964, Recall@5: 0.7629 → 0.8048)
Adım 4 — Prompt tuning: citation discipline, Dayanak formatı (dev set üzerinde iterasyon)
Adım 5 — QLoRA SFT: 112 örnek, 3 epoch fine-tuning (answer_f1 +14.6%, faithfulness +15.9%)

Not: Adım 2'de gerçek embedding fine-tuning (contrastive learning / hard negative mining) yapılmamıştır. Bunun yerine daha güçlü bir pretrained model (e5-large) seçilerek retrieval kalitesi iyileştirilmiştir. Bu karar dürüstçe "model selection" olarak konumlandırılmıştır.

13.1.1 Formal Ablation Tablosu (Test Split, 31 soru)

| Varyant | Retrieval | LLM | MRR | R@5 | Ans F1 | Cite F1 | Cite Exact | Faith | Latency |
|---------|-----------|-----|-----|-----|--------|---------|------------|-------|---------|
| 1. Baseline | e5-base dense | Qwen 3B untuned | 0.5896 | 0.7429 | 0.2453 | 0.3765 | 0.1290 | 0.7228 | ~15s |
| 2. + Model selection | e5-large dense | Qwen 3B untuned | 0.6773 | 0.7629 | 0.3024 | 0.5233 | 0.1935 | 0.7615 | ~15s |
| 3. + Reranker | e5-large + BGE | Qwen 3B untuned | 0.6964 | 0.8048 | 0.2567 | 0.4851 | 0.0968 | 0.7454 | ~17s |
| 4. + QLoRA SFT | e5-large + BGE | Qwen 3B QLoRA | 0.6964 | 0.8048 | **0.4031** | **0.5742** | **0.4516** | **0.9041** | ~21s |

Varyant 3'te answer F1'in varyant 2'ye göre düştüğü görülmektedir. Bunun nedeni reranker'ın daha doğru ama farklı chunk'lar getirmesidir — untuned LLM bu değişikliğe adapte olamamıştır. QLoRA SFT ile (varyant 4) tüm metriklerde dramatik iyileşme sağlanmıştır.

Retrieval metrikleri (MRR, Recall@5) tüm 175 soru üzerinden hesaplanmıştır. QA metrikleri (Ans F1, Cite F1, vb.) test split 31 soru üzerinden hesaplanmıştır.

13.2 Retrieval Sonuçları (Faz 2, 175 soru — tüm benchmark)

| Sistem | MRR | Recall@5 | Recall@10 |
|--------|-----|----------|-----------|
| baseline_dense (e5-base) | 0.5896 | 0.7429 | 0.7933 |
| bm25_only | 0.3208 | 0.4657 | 0.5667 |
| hybrid (e5-base + BM25, RRF) | 0.5510 | 0.7171 | 0.8057 |
| dense_reranked (İngilizce CE) | 0.4139 | 0.6229 | 0.7286 |
| hybrid_reranked (İngilizce CE) | 0.4122 | 0.5943 | 0.7257 |
| dense_reranked_ml (mMiniLMv2) | 0.6568 | 0.7524 | 0.8086 |
| hybrid_reranked_ml (mMiniLMv2) | 0.6587 | 0.7533 | 0.8143 |
| e5large_dense | 0.6773 | 0.7629 | 0.8476 |
| e5large_reranked_ml (mMiniLMv2) | 0.6644 | 0.7952 | 0.8514 |
| **e5large_reranked_bge (BGE-v2-m3)** | **0.6964** | **0.8048** | **0.8743** |

Not: İlk 9 sistem Faz 2'de koşturulmuş olup comparison_report.json'dadır. BGE reranker Faz 3'te entegre edilmiş olup eval_e5large_reranked_bge.json'da ayrı tutulmaktadır. MRR değerleri: comparison_report'ta "mrr" olarak (top-5 MRR), BGE eval'da "chunk_mrr" olarak raporlanmıştır.

Önemli bulgular:
- İngilizce cross-encoder zararlı: MRR'ı 0.59'dan 0.41'e düşürdü
- E5-large tek başına tüm e5-base kombinasyonlarını geçti
- En yüksek Recall@5: **e5large_reranked_bge** (0.8048) — pipeline default
- En yüksek MRR: **e5large_reranked_bge** (0.6964) — BGE reranker hem MRR hem Recall'da lider
- BGE vs mMiniLMv2: MRR +0.032, Recall@5 +0.010, Recall@10 +0.023 — BGE her metrikte üstün

13.3 QA Sonuçları — Dev Split (32 soru)

Not: Dev split'te baseline, prompt iterasyonu sırasında e5large_reranked_ml retrieval sistemiyle koşturulmuştur (prompt_v5). SFT-QLoRA sonuçları ise e5large_reranked_bge ile koşturulmuştur. Bu nedenle dev split'teki iyileşmenin bir kısmı retrieval sistemi farkından kaynaklanabilir. Test split'te her iki sonuç da aynı retrieval sistemi (e5large_reranked_bge) kullanılmıştır ve karşılaştırma güvenilirdir.

| Metrik | Baseline | SFT-QLoRA | Delta |
|--------|----------|-----------|-------|
| Answer F1 | 0.2440 | 0.3793 | +13.5% |
| Answer Precision | 0.2099 | 0.3621 | +15.2% |
| Answer Recall | 0.3376 | 0.4638 | +12.6% |
| Citation F1 | 0.4278 | 0.5156 | +8.8% |
| Citation Precision | 0.3411 | 0.5104 | +16.9% |
| Citation Recall | 0.7031 | 0.5625 | −14.1% |
| Citation Exact | 0.1562 | 0.3750 | +21.9% |
| Has Dayanak | 0.8438 | 0.8438 | +0.0% |
| Faithfulness | 0.7671 | 0.8491 | +8.2% |

13.4 QA Sonuçları — Test Split (31 soru)

| Metrik | Baseline | SFT-QLoRA | Delta |
|--------|----------|-----------|-------|
| Answer F1 | 0.2567 | 0.4031 | +14.6% |
| Answer Precision | 0.2273 | 0.4051 | +17.8% |
| Answer Recall | 0.3622 | 0.4598 | +9.8% |
| Citation F1 | 0.4851 | 0.5742 | +8.9% |
| Citation Precision | 0.3946 | 0.5806 | +18.6% |
| Citation Recall | 0.8065 | 0.6237 | −18.3% |
| Citation Exact | 0.0968 | 0.4516 | +35.5% |
| Has Dayanak | 0.8387 | 0.9677 | +12.9% |
| Faithfulness | 0.7454 | 0.9041 | +15.9% |

13.5 Soru Tipi Bazlı Sonuçlar (SFT-QLoRA, Test)

| Soru Tipi | Answer F1 | Citation F1 | Faithfulness |
|-----------|-----------|-------------|-------------|
| Definition | 47.6% | 74.4% | 92.4% |
| Factual | 26.4% | 33.3% | 67.8% |
| List | 39.8% | 54.7% | 97.1% |
| Procedural | 33.8% | 33.3% | 97.6% |

13.6 Kanun Bazlı Sonuçlar (SFT-QLoRA, Test)

| Kanun | Answer F1 | Citation F1 | Faithfulness |
|-------|-----------|-------------|-------------|
| Ceza Muhakemesi Kanunu | 58.3% | 79.2% | 95.8% |
| Türk Borçlar Kanunu | 56.0% | 93.3% | 92.3% |
| Türk Medeni Kanunu | 47.0% | 65.0% | 95.2% |
| Anayasa | 46.3% | 73.3% | 85.8% |
| Türk Ceza Kanunu | 25.3% | 28.0% | 74.8% |
| Hukuk Muhakemeleri Kanunu | 21.4% | 25.0% | 96.0% |
| İdari Yargılama Usulü Kanunu | 5.7% | 0.0% | 100.0% |

---

14. Faz 3: Error Analysis

14.1 Genel Durum

SFT-QLoRA sonrası test setinde:
- İyi cevaplar (F1 > 0.5): 13/31 (%42)
- Düşük faithfulness (< 0.5): 2/31 (%6)
- Eksik Dayanak: 1/31 (%3)

Baseline ile karşılaştırma:
- İyi cevaplar: 4/31 → 13/31 (%13 → %42)
- Düşük faithfulness: 5/31 → 2/31 (%16 → %6)
- Eksik Dayanak: 5/31 → 1/31 (%16 → %3)

14.2 Düşük Faithfulness Vakaları

tck_025 (faith=0.10, F1=0.05): Model, yağma suçunun cezası hakkında context dışı bilgi üretti. Bu vaka tipik bir halüsinasyon örneğidir — model eğitim verisinden ezberlenmiş bilgiyi context yerine kullandı.

anayasa_003 (faith=0.40, F1=0.32): Model, Anayasa'nın yapısı hakkında kısmen doğru ama context'te bulunmayan detaylar ekledi.

14.3 Kanun Bazlı Zayıflıklar

Türk Ceza Kanunu (TCK): Answer F1 %25.3 — model özellikle ceza miktarı ve nitelikli hallerde düşük performans gösterdi. Faithfulness %74.8 ile diğer kanunlara göre belirgin şekilde düşük.

Hukuk Muhakemeleri Kanunu (HMK): Answer F1 %21.4 — usul hukuku sorularında model yeterli detay veremedi.

İdari Yargılama Usulü Kanunu (İYUK): Answer F1 %5.7, Citation F1 %0.0 — yalnızca 2 soru, ama her ikisinde de ciddi hata. Az örnekle SFT verisi yetersiz kalmış olabilir.

14.4 Soru Tipi Bazlı Zayıflıklar

Factual sorular (F1 %26.4): Spesifik sayısal bilgi veya koşul gerektiren sorularda model yetersiz. 3B parametrelik model bu düzeyde kesinlik için sınırlı.

Procedural sorular (F1 %33.8): Usul adımlarını sıralama gerektiren sorularda eksik adımlar. Ancak faithfulness %97.6 ile iyi — model context'e sadık ama eksik.

14.5 Citation Recall Düşüşü

SFT sonrası citation recall %80.6'dan %62.4'e düştü. Ancak citation precision %39.5'ten %58.1'e ve citation exact %9.7'den %45.2'ye yükseldi. Model daha az ama daha doğru citation üretmeyi öğrendi. Bu trade-off kabul edilebilir çünkü precision ve exact match artışı recall kaybından daha değerlidir.

14.6 Sistemin En Zayıf Halkası

Generation kalitesi en zayıf halkadır. Retrieval tarafı Recall@5 ~%79 ile iyi performans göstermektedir. Ancak 3B parametrelik model, özellikle karmaşık hukuki sorularda (çok maddeli, nitelikli hal, koşullu ceza) yeterli cevap üretememektedir. Daha büyük bir model (7B+) veya API tabanlı inference bu sorunu hafifletebilir.

---

15. Faz 3: Yapılmayanlar ve Savunma

Aşağıdaki çalışmalar Faz 3 kapsamında planlanmış ancak yapılamamıştır:

✗ Embedding fine-tuning / contrastive tuning / hard negative mining — 8GB VRAM sınırlarında model selection (e5-base → e5-large) önceliklendirildi ve MRR'da +14.9% iyileşme sağlandı. Fine-tuning yerine model büyütme stratejisi benimsenmiştir.

✗ Reranker fine-tuning — bge-reranker-v2-m3 zero-shot performansı yeterli görüldü. Reranker fine-tuning için query-positive-negative üçlülerinden oluşan eğitim verisi üretilmesi ve çapraz doğrulama gerektiğinden ertelenmiştir.

✗ BLEU / ROUGE metrikleri — Hukuk metinlerinde n-gram overlap metrikleri yanıltıcı olabilir. Token-level F1, citation accuracy ve faithfulness daha anlamlı metrikler olarak tercih edilmiştir. Ancak karşılaştırma amacıyla ileride eklenebilir.

✗ TBMM / Yargıtay verisi — Mevcut 175 soruluk benchmark tamamen 7 mevzuat metniyle cevaplanabilir durumdadır. İçtihat ve Meclis tutanakları farklı bir ingest pipeline ve daha geniş bir benchmark gerektirir. Scope dışı bırakılmıştır.

Savunma: 8GB yerel VRAM kısıtları dahilinde model selection + reranker + QLoRA hattı stratejik olarak önceliklendirilmiştir. Bu strateji ile answer_f1'de +14.6%, citation exact'ta +35.5% ve faithfulness'ta +15.9% iyileşme sağlanmıştır. Bu kazanımlar, fine-tuning yerine model seçimi ve prompt/SFT optimizasyonu ile elde edilmiştir.

---

16. Future Work

- **Reranker fine-tuning**: bge-reranker-v2-m3 Türkçe hukuk verisine özel fine-tune edilebilir (query-positive-negative üçlüleri ile)
- **Embedding fine-tuning**: Contrastive learning ile domain-specific embedding eğitimi (daha güçlü GPU gerekli)
- **Hard negative mining**: Retrieval hatalarından hard negative örnekleri çıkarılarak hem embedding hem reranker iyileştirilebilir
- **BLEU / ROUGE**: Karşılaştırma amacıyla eklenebilir
- **TBMM / Yargıtay genişletme**: İçtihat ve Meclis tutanakları corpus'a eklenebilir, benchmark genişletilebilir
- **Daha büyük LLM denemesi**: Qwen2.5-7B veya daha büyük modeller API tabanlı inference ile denenebilir
- **Daha güçlü faithfulness değerlendirme**: NLI tabanlı veya LLM-as-judge yöntemi ile daha güvenilir faithfulness ölçümü
- **Çapraz doğrulama**: k-fold cross-validation ile daha güvenilir metrik tahminleri

---

17. Donanım ve Ortam

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU (8GB VRAM)
- PyTorch: 2.7.0+cu124
- CUDA: 12.4
- Python: 3.11
- Temel kütüphaneler: transformers, peft, trl 1.4, bitsandbytes, faiss-cpu, sentence-transformers
- QLoRA eğitim süresi: ~22 dakika (3 epoch, 42 step)
- Inference latency: baseline ~17s/soru, SFT-QLoRA ~21s/soru
