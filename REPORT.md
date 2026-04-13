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
✗ QA evaluation (F1, EM metriklerinin generation çıktıları üzerinde koşturulması) — Qwen 3B modeli 4GB VRAM'de soru başına dakikalarca sürdüğünden 175 soruluk tam benchmark koşturulamadı. QA eval script'i (run_qa_eval.py) hazırlanmış ancak donanım kısıtı nedeniyle Faz 4'e bırakılmıştır. İbo'nun daha güçlü GPU'su veya API tabanlı inference ile koşturulabilir.

FAZ 3 (İbo — Bekliyor):
Bu fazda İbo'dan beklenenler:
- Çok dilli reranker'ın fine-tuning'i (mevcut mmarco modeli iyi ama Türkçe hukuk verisine özel fine-tune ile daha iyi olabilir)
- Alternatif reranker denemesi (BAAI/bge-reranker-v2-m3 gibi daha büyük modeller)
- Retrieval-aware prompting iyileştirmesi
- LLM üzerinde LoRA/QLoRA fine-tuning denemesi
- Tüm ablation varyantlarının koşturulması (en iyi retrieval sistemi: e5large_reranked_ml olarak belirlenmiştir)
- Best system pipeline ve inference script

İbo'ya devredilen sonuçlar ve çıktılar:
- data/benchmark/gold_benchmark.jsonl (175 soru)
- src/evaluation/ altındaki tüm evaluation script'leri
- outputs/evaluation/ altında 9 sistemin eval sonuçları (JSON)
- configs/retrieval_config_e5large.yaml (e5-large konfigürasyonu)
- data/processed/corpus/faiss_index_e5large.bin (e5-large FAISS indeksi)
- En iyi sistem: e5large_reranked_ml (Recall@10=0.89, MRR=0.66, Hit@5=0.85)
- İngilizce cross-encoder zararlı, çok dilli cross-encoder faydalı
- E5-large > e5-base (tek başına bile tüm e5-base kombinasyonlarını geçiyor)
- List sorularında en iyi: e5large_reranked_ml (Recall@5=0.77, baseline 0.59'dan +29%)
- Hard sorularda en iyi: e5large_reranked_ml (Recall@5=0.74, baseline 0.54'ten +36%)

FAZ 4 (Deniz — Bekliyor):
Bu fazda yapılması gerekenler:
- QA metrikleri (EM, F1, BLEU, ROUGE, faithfulness, citation accuracy)
- Halüsinasyon analizi
- Soru tipi bazında başarısızlık sınıflandırması
- Ablation tabloları ve grafikler
- Error analysis (hangi sorularda, neden, hangi kanunlarda hata)
- Raporun deney, sonuç ve error analysis kısımları
- Sunum ve demo senaryosu
