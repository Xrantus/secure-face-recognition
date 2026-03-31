# 🎯 Tez Projesi - Object Detection Sistemi

Bu proje, Python kullanarak object detection (nesne tespiti) yapmak için geliştirilmiş kapsamlı bir sistemdir. YOLOv11 (You Only Look Once) modeli temel alınarak geliştirilmiştir.

## 🚀 Özellikler

- **Görüntü Tespiti**: JPG, PNG formatındaki görüntülerde nesne tespiti
- **Video Tespiti**: MP4, AVI formatındaki videolarda nesne tespiti
- **Webcam Desteği**: Gerçek zamanlı webcam ile nesne tespiti ve FPS gösterimi
- **Anlık Konsol Loglama**: Tespit edilen nesneler her frame'de konsola yazdırılır
- **FPS Gösterimi**: Sağ üst köşede gerçek zamanlı FPS değeri gösterilir
- **YOLOv11 Desteği**: En yeni YOLO modeli ile yüksek doğruluk
- **Sonuç Kaydetme**: Tespit sonuçlarını görsel olarak kaydetme
- **Türkçe Destek**: Kullanıcı dostu Türkçe arayüz

## 📋 Gereksinimler

Python 3.7 veya üzeri gereklidir. Gerekli paketler:

```bash
pip install -r requirements.txt
```

### Gerekli Paketler:
- `opencv-python` - Görüntü/video işleme
- `numpy` - Sayısal işlemler
- `matplotlib` - Görselleştirme
- `ultralytics` - YOLO modelleri

## 🛠️ Kurulum

1. Repository'yi klonlayın:
```bash
git clone <repository-url>
cd thesis-project-image
```

2. Gerekli paketleri kurun:
```bash
pip install -r requirements.txt
```

3. İlk modeli indirin:
```python
from ultralytics import YOLO
model = YOLO('yolo11n.pt')  # Nano model (en hızlı)
```

## 📖 Kullanım

### Temel Kullanım

```python
from object_detection import ObjectDetector

# Object detector oluştur
detector = ObjectDetector()

# Görüntü tespiti
results = detector.detect_image("resim.jpg")

# Video tespiti
output_path = detector.detect_video("video.mp4")

# Webcam tespiti (FPS gösterimi ve konsol loglama ile)
detector.detect_webcam()
```

### Örnek Kullanım Script'i

Daha fazla örnek için `example_usage.py` dosyasını çalıştırın:

```bash
python example_usage.py
```

## 🎯 Desteklenen Nesne Türleri

YOLOv8 modeli 80 farklı nesne türünü tespit edebilir:

- İnsanlar (person)
- Araçlar (car, truck, bus, motorcycle, bicycle)
- Hayvanlar (dog, cat, bird, horse, cow, sheep)
- Yiyecekler (pizza, cake, apple, orange, banana)
- Mobilyalar (chair, table, bed, sofa)
- Elektronik (tv, laptop, cell phone, keyboard)
- Spor malzemeleri (ball, racket)
- Ve daha fazlası...

## ⚙️ Yapılandırma

### Model Seçimi

```python
# Farklı model boyutları (YOLOv11)
detector = ObjectDetector('yolo11n.pt')  # Nano (en hızlı, en az doğruluk)
detector = ObjectDetector('yolo11s.pt')  # Small
detector = ObjectDetector('yolo11m.pt')  # Medium
detector = ObjectDetector('yolo11l.pt')  # Large (en yavaş, en doğru)
detector = ObjectDetector('yolo11x.pt')  # Extra Large (en doğru)
```

### Güven Eşiği Ayarlama

```python
# Sadece yüksek güvenilir nesneleri göster
results = detector.detect_image("resim.jpg", conf_threshold=0.7)
```

## 📊 Performans İpuçları

- Daha hızlı işlem için `yolo11n.pt` (nano) modeli kullanın
- Daha doğru sonuçlar için `yolo11x.pt` modeli kullanın
- Video işlemeyi hızlandırmak için daha düşük çözünürlük kullanın
- GPU varsa CUDA desteği otomatik olarak kullanılır
- Webcam kullanımında FPS değerini sağ üst köşeden takip edin

## 🔧 Sorun Giderme

### Yaygın Hatalar

1. **"CUDA out of memory"**: Daha küçük model kullanın veya batch size'ı azaltın
2. **"Video codec error"**: Video dosyasının codec'ini kontrol edin
3. **"Webcam açılamadı"**: Webcam'in başka program tarafından kullanılıp kullanılmadığını kontrol edin

### Destek

Herhangi bir sorun yaşarsanız:
1. Python ve pip sürümlerinizi kontrol edin
2. Gerekli paketlerin doğru kurulduğunu doğrulayın
3. GPU sürücülerinizin güncel olduğunu kontrol edin

## 📁 Proje Yapısı

```
thesis-project-image/
├── README.md                 # Bu dosya
├── requirements.txt          # Gerekli paketler
├── object_detection.py       # Ana object detection sınıfı
├── example_usage.py         # Örnek kullanım script'i
├── results/                 # Tespit sonuçları (otomatik oluşturulur)
└── test_files/              # Test görüntüleri ve videoları
```

## 🎓 Akademik Kullanım

Bu proje tez çalışması için geliştirilmiştir. Kaynak göstererek akademik çalışmalarda kullanabilirsiniz.

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir. Daha fazla bilgi için iletişime geçin.

---

**Başarılar!** 🚀