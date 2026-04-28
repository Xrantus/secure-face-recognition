## Refactored Face Recognition Project (OOP + Modular)

Bu klasör, orijinal monolitik kodu **değiştirmeden** (dokunmadan) OOP prensipleriyle modüler hale getirilmiş sürümdür.

## Kurulum

- **Ortak**:
  - Python 3.10+ önerilir
  - OpenCV, NumPy, Ultralytics, InsightFace gerekir
- **RPi (opsiyonel)**:
  - `picamera2` sadece Raspberry Pi tarafında gereklidir.

## Hızlı kurulum (venv)

Bu repo içinde iki doğru kurulum şekli var. **Önemli:** terminalde

- `pip: bad interpreter: .../.venv/bin/python: no such file or directory`

görürsen, büyük ihtimalle `.venv` başka bir yerden kopyalanmış/taşınmış veya eski bir path’e işaret ediyordur. Bu durumda `.venv` klasörünü silip aynı dizinde yeniden oluştur.

### Seçenek A (önerilen): venv repo root’ta

Repo root’ta (yani `thesis-project-image/` içinde):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r refactored_project/requirements.txt
```

Kurulumdan sonra hızlı kontrol:

```bash
source .venv/bin/activate
python -c "from refactored_project.face_recognizer import FaceRecognizer; print('ok')"
```

### Seçenek B: venv `refactored_project/` içinde

Bu README’nin bulunduğu klasörde:

```bash
cd refactored_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
cd ..
```

## Config seçenekleri (`refactored_project/config.py`)

Bu proje 3 ana config grubuna ayrılmıştır:

### HARDWARE_ENV (kamera backend)

- `HARDWARE_ENV`: `Literal["MAC", "RPI", "WIN"]`
  - Sadece **kamera okuma** backend’ini seçer (Mac/Windows: `cv2.VideoCapture`, RPi: `Picamera2`).

### MODEL_CONFIG (model ve inference parametreleri)

- `MODEL_CONFIG.yolo_model_path: str`
  - Varsayılan: `./yolo11-modes/face_yolo11n_int8.onnx` (yoksa `yolo11-modes/` altındaki ilk `.onnx` seçilir)
- `MODEL_CONFIG.recognizer_model_name: str` (varsayılan: `"buffalo_s"`)
- YOLO:
  - `MODEL_CONFIG.yolo_img_size: int` (varsayılan: `640`)
  - `MODEL_CONFIG.yolo_pred_conf: float` (varsayılan: `0.01`)
  - `MODEL_CONFIG.yolo_det_threshold: float` (varsayılan: `0.15`)
  - `MODEL_CONFIG.yolo_iou: float` (varsayılan: `0.45`)
  - `MODEL_CONFIG.max_det: int` (varsayılan: `100`)
- InsightFace / ROI:
  - `MODEL_CONFIG.det_size: tuple[int,int]` (varsayılan: `(160,160)`)
  - `MODEL_CONFIG.landmark_pad: float` (varsayılan: `0.20`)
  - `MODEL_CONFIG.min_face_size: int` (varsayılan: `35`)
- Performans:
  - `MODEL_CONFIG.frame_skip: int` (varsayılan: `2`)
- DB:
  - `MODEL_CONFIG.db_path: str` (varsayılan: `"known_faces_embeddings.npz"`)

### METRIC_CONFIG (metrik ve threshold)

- `METRIC_CONFIG.similarity_metric`: `Literal["cosine","euclidean"]` (varsayılan: `"cosine"`)
- `METRIC_CONFIG.cosine_threshold: float` (varsayılan: `0.50`)  
  - Cosine için **büyük olan iyi**, kabul şartı: `score >= threshold`
- `METRIC_CONFIG.euclidean_threshold: float` (varsayılan: `1.00`)  
  - Euclidean için **küçük olan iyi**, kabul şartı: `distance <= threshold`

### CAMERA_CONFIG (donanım detayı)

- OpenCV camera (Mac/Windows):
  - `CAMERA_CONFIG.opencv_camera_index: int` (varsayılan: `0`)
  - `CAMERA_CONFIG.opencv_frame_width: int` (varsayılan: `1280`)
  - `CAMERA_CONFIG.opencv_frame_height: int` (varsayılan: `720`)
- RPi:
  - `CAMERA_CONFIG.rpi_preview_size: tuple[int,int]` (varsayılan: `(640,480)`)

## Çalıştırma

- **DB oluşturma**:
  - `python -m refactored_project.db_create_refactored`
- **Canlı sistem**:
  - `python -m refactored_project.main`

## DB Create (detaylı)

`db_create_refactored.py` şu yapıyı bekler:

- `./db-images/<KisiAdi>/*.jpg|png|jpeg|webp|bmp`

Çalıştırınca:

- Her kişi klasörü için yüz tespit eder, embedding çıkarır, **kişi başına ortalama embedding** hesaplar.
- Repo root’a `MODEL_CONFIG.db_path` ismiyle `.npz` kaydeder (varsayılan: `known_faces_embeddings.npz`)
  - `.npz` anahtarları: `encodings`, `names`

Örnek:

```bash
source .venv/bin/activate

# DB oluştur
python -m refactored_project.db_create_refactored

# Sonra canlı sistemde kullan
python -m refactored_project.main
```

Not: DB create, `config.MODEL_CONFIG.yolo_model_path` ve `config.MODEL_CONFIG.recognizer_model_name` değerlerini kullanır.

## CLI (benchmark/test override)

`refactored_project/main.py` şu override seçeneklerini destekler:

- `--hardware-env MAC|RPI`
- `--hardware-env MAC|RPI|WIN`
- `--yolo-model-path <path-or-filename>`
  - Dosya adı verirsen `./yolo11-modes/<filename>` altında aranır
- `--recognizer-model-name buffalo_s` (veya başka InsightFace modeli)
- `--metric cosine|euclidean`
- `--threshold <float>` (seçilen metriğin threshold’unu override eder)
- `--db-path <npz>` (repo root’a göre çözülür)
- `--video <path-or-filename>`
  - Dosya adı verirsen `./test-videos/<filename>` altında aranır

Örnekler:

```bash
source .venv/bin/activate

# Varsayılan (kamera)
python -m refactored_project.main

# Video ile test (test-videos/ altından)
python -m refactored_project.main --video sample.mp4

# Modeli yolo11-modes/ altından seç
python -m refactored_project.main --yolo-model-path face_yolo11n_int8.onnx

# Euclidean ile dene
python -m refactored_project.main --metric euclidean --threshold 1.0
```

## Notlar

- `config.py` içindeki `HARDWARE_ENV` sadece kamera backend’ini seçer.
- Model seçimi (`YOLO_MODEL_PATH`) ve metrik seçimi (`SIMILARITY_METRIC`) donanımdan bağımsızdır ve benchmark amaçlı kolayca değiştirilebilir.

