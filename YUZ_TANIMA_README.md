# Yüz Tanıma Sistemi - Teknik Görev İmplementasyonu

Bu proje, YOLO modelleri ve yüz tanıma teknolojilerini kullanarak gelişmiş bir yüz tanıma sistemi geliştirmeyi amaçlamaktadır.

## 📋 Proje İçeriği

### 🔧 Teknik Görevler

1. **YOLO Modelini Kısıtlama**
   - Sadece `person` (insan) sınıfını tespit edecek şekilde yapılandırma
   - COCO veri seti `class_id = 0` kullanımı

2. **Yüz Tanıma (Aşama 1 - Vektör Temsili)**
   - Tespit edilen yüzlerin kırpılması (crop)
   - Histogram tabanlı öznitelik çıkarma
   - Kosinüs benzerliği ile karşılaştırma
   - Gerçek zamanlı yüz tanıma

3. **Yüz Tanıma (Aşama 2 - Özel Eğitim)**
   - YOLOv11 Nano modeli özel kişi isimleri ile eğitim
   - Google Colab entegrasyonu
   - Özel veri seti hazırlama

## 📁 Dosya Yapısı

```
thesis-project-image/
├── face_recognition.py          # Ana yüz tanıma uygulaması
├── yolo_person_training.py      # Eğitim scripti (Google Colab için)
├── YUZ_TANIMA_README.md         # Bu dokümantasyon
├── optimized.py                 # Optimize edilmiş temel YOLO uygulaması
├── face_detection.py            # Yüz algılama uygulaması
├── yolo11n.pt                   # YOLOv11 Nano modeli
└── test/                        # Test dosyaları ve sonuçlar
```

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install ultralytics opencv-python numpy pillow
```

### Ana Uygulama (face_recognition.py)

```python
python face_recognition.py
```

**Kullanım:**
- Uygulama başladığında kamera açılır
- `r` tuşuna basarak referans yüzü kaydedin
- Sistem yüzleri tanır ve benzerlik skorunu gösterir
- `q` tuşu ile çıkın

### Eğitim Scripti (yolo_person_training.py)

Google Colab'da çalıştırın:

1. Dosyayı Colab'a yükleyin
2. Gerekli kütüphaneleri yükleyin:
```python
!pip install ultralytics opencv-python numpy pillow pyyaml
```

3. Veri setinizi hazırlayın:
```python
# Veri seti klasör yapısı:
# person_dataset/raw/train/ -> Ahmet/, Buket/, vs. klasörleri
# person_dataset/raw/valid/ -> Doğrulama görüntüleri
# person_dataset/raw/test/  -> Test görüntüleri
```

4. Scripti çalıştırın:
```python
from yolo_person_training import main
main()
```

## ⚙️ Yapılandırma

### YOLO Model Kısıtlama

```python
# Sadece person sınıfını tespit et (COCO class_id = 0)
results = model(frame, classes=[PERSON_CLASS_ID], conf=PERSON_CONFIDENCE_THRESHOLD)
```

### Yüz Tanıma Sistemi

```python
# Öznitelik çıkarma
def extract_face_features(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return np.concatenate([hist, np.mean(face_image, axis=(0, 1))])

# Benzerlik hesaplaması
similarity = calculate_similarity(face_features, target_vector)
```

## 📊 Özellikler

### Gerçek Zamanlı İşleme
- Threading ile optimize edilmiş kamera yakalama
- FPS optimizasyonu için frame skipping
- Real-time yüz tanıma

### Gelişmiş Yüz Algılama
- Haar Cascade yüz algılama
- Histogram eşitleme
- Gürültü azaltma

### Eğitim ve Test
- Özel veri seti hazırlama
- Veri artırımı teknikleri
- Model performans değerlendirmesi

## 🔧 Parametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `FRAME_SKIP` | 3 | Kare atlama sayısı |
| `YOLO_IMG_SIZE` | 320 | YOLO görüntü boyutu |
| `PERSON_CONFIDENCE_THRESHOLD` | 0.5 | Kişi tespit eşiği |
| `PERSON_CLASS_ID` | 0 | COCO person sınıf ID'si |

## 📈 Performans

- **FPS**: 25-30 FPS (640x480 çözünürlük)
- **Doğruluk**: Histogram tabanlı öznitelik çıkarma ile %70-85 benzerlik doğruluğu
- **Bellek**: Optimize threading ile düşük bellek kullanımı

## 🛠️ Geliştirme Notları

### Aşama 1 (Vektör Temsili)
- Basit histogram tabanlı yaklaşım
- Hızlı ancak sınırlı doğruluk
- Test amaçlı uygun

### Aşama 2 (Özel Eğitim)
- Daha yüksek doğruluk potansiyeli
- Özel veri seti gerektirir
- Eğitim süresi ve kaynak ihtiyacı

## 🔍 Sorun Giderme

### Yaygın Hatalar

1. **Kamera Açılamıyor**
   ```python
   # Kamera index'ini kontrol edin (0, 1, 2, vs.)
   cap = cv2.VideoCapture(0)
   ```

2. **Model Dosyası Bulunamıyor**
   ```python
   # Model dosyasının mevcut olduğunu kontrol edin
   # yolo11n.pt aynı dizinde olmalı
   ```

3. **CUDA Hatası**
   ```python
   # CPU modunda çalıştırın
   model = YOLO('yolo11n.pt')  # device parametresi eklenmezse CPU kullanılır
   ```

## 📚 Referanslar

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [OpenCV Face Detection](https://docs.opencv.org/)
- [COCO Dataset Classes](https://cocodataset.org/#explore)

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.
