# Caretta Re-ID

Caretta caretta bireylerini baş bölgesindeki pul desenlerinden tanıyan, annotation tabanlı bir re-identification pipeline'ı.

## Kurulum

1. Python 3.11 veya üstünü kullanın.
2. Sanal ortam oluşturup aktif edin.
3. Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

4. `.env.example` dosyasını `.env` olarak kopyalayın ve gerekirse değerleri düzenleyin.

## Veri Kurulumu

Bu repoda büyük veri dosyaları GitHub limitleri nedeniyle tutulmaz. Özellikle aşağıdaki yolları sizin doldurmanız gerekir:

- `data/raw/images/` (ham görseller)
- `data/raw/annotations.json` (COCO annotation)
- `data/raw/metadata.csv`
- `data/raw/metadata_splits.csv`

Beklenen temel dizin yapısı:

```text
data/
	raw/
		images/
		annotations.json
		metadata.csv
		metadata_splits.csv
	processed/
```

Veri dosyalarını yerleştirdikten sonra embedding veritabanını oluşturun:

```bash
python -m caretta_reid.database.embedding_store
```

## .env.example

Örnek yapı:

```env
CARETTA_RAW_DATA_DIR=data/raw
CARETTA_PROCESSED_DATA_DIR=data/processed
CARETTA_ANNOTATIONS_PATH=data/raw/annotations.json
CARETTA_METADATA_CSV_PATH=data/raw/metadata.csv
CARETTA_METADATA_SPLITS_PATH=data/raw/metadata_splits.csv
CARETTA_EMBEDDINGS_PERSIST_DIR=data/processed/chroma
CARETTA_DEMO_OUTPUT_DIR=data/processed/demo
CARETTA_SPLIT_COLUMN=split_closed
CARETTA_CHROMA_COLLECTION_NAME=caretta_embeddings
CARETTA_EMBEDDING_DIMENSION=512
CARETTA_SIMILARITY_THRESHOLD=0.65
CARETTA_TOP_K_MATCHES=5
CARETTA_BATCH_SIZE=8
CARETTA_IMAGE_SIZE=224
CARETTA_PRETRAINED_BACKBONE=true
CARETTA_USE_GPU_IF_AVAILABLE=true
CARETTA_DEBUG_MODE=false
CARETTA_DEV_TURTLE_IDS=t001,t002,t003,t004,t005,t006,t007,t008,t009,t010
```

## Eğitim Komutu

Triplet-loss eğitimi çalıştırmak için:

```bash
python -m caretta_reid.models.siamese
```

Bu komut `metadata_splits.csv` içindeki `CARETTA_SPLIT_COLUMN` kolonunu ve yalnızca `CARETTA_DEV_TURTLE_IDS` listesini kullanır.

## Embedding Veritabanını Doldurma

ChromaDB koleksiyonunu doldurmak için:

```bash
python -m caretta_reid.database.embedding_store
```

Bu komut annotation tabanlı detection, preprocessing ve embedding ajanlarını sıralı biçimde çalıştırır ve üretim verisini `data/processed/chroma` altına yazar.

## Demo Başlatma

Gradio demo uygulamasını açmak için:

```bash
python -m caretta_reid.demo.app
```

## Mimari Kararlar

- Ajanlar tek sorumluluk ilkesine göre ayrıldı; detection, preprocessing, embedding, matching ve karar verme ayrı sınıflarda tutuldu.
- Ajanlar arası iletişim Pydantic v2 modelleriyle yapılıyor; bu, tip güvenliği ve doğrulamayı merkezileştiriyor.
- Backbone için strateji deseni kullanıldı; yeni CNN omurgaları mevcut ajanları değiştirmeden eklenebilir.
- Embedding deposu için ChromaDB seçildi; cosine similarity ile hızlı nearest-neighbor araması sağlanıyor.
- `Settings` sınıfı tüm path ve hiperparametreleri tek yerde topluyor; hardcoded path ve magic number kullanımını azaltıyor.
- `metadata_splits.csv` içindeki hazır split kullanılıyor; rastgele bölme yapılmıyor.
- Geliştirme aşamasında `t001`-`t010` aralığıyla sınırlandırma `CARETTA_DEV_TURTLE_IDS` üzerinden kontrol ediliyor.
