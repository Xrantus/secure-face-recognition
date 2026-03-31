"""
Object Detection Sistemi - Sistem Testi
=====================================

Bu script sistem bileşenlerinin doğru çalışıp çalışmadığını test eder.
"""

import sys
import cv2
import numpy as np

def test_imports():
    """Gerekli kütüphanelerin import edilebilirliğini test eder"""
    print("🔍 Import testleri başlatılıyor...")

    try:
        import cv2
        print("✅ OpenCV başarıyla import edildi")
    except ImportError as e:
        print(f"❌ OpenCV import hatası: {e}")
        return False

    try:
        import numpy
        print("✅ NumPy başarıyla import edildi")
    except ImportError as e:
        print(f"❌ NumPy import hatası: {e}")
        return False

    try:
        import matplotlib
        print("✅ Matplotlib başarıyla import edildi")
    except ImportError as e:
        print(f"❌ Matplotlib import hatası: {e}")
        return False

    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO başarıyla import edildi")
    except ImportError as e:
        print(f"❌ Ultralytics YOLO import hatası: {e}")
        return False

    return True

def test_opencv():
    """OpenCV'nin temel işlevlerini test eder"""
    print("\n🖼️  OpenCV testleri başlatılıyor...")

    try:
        # OpenCV sürüm kontrolü
        version = cv2.__version__
        print(f"✅ OpenCV sürümü: {version}")

        # Temel görüntü işlemleri
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        print(f"✅ Test görüntüsü oluşturuldu: {img.shape}")

        # Renk dönüşümü
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"✅ Gri tonlama başarılı: {gray.shape}")

        return True
    except Exception as e:
        print(f"❌ OpenCV test hatası: {e}")
        return False

def test_yolo():
    """YOLO modelinin yüklenip yüklenemediğini test eder"""
    print("\n🎯 YOLO modeli testleri başlatılıyor...")

    try:
        from ultralytics import YOLO

        # Nano modelini yükle (en küçük ve hızlı)
        model = YOLO('yolov8n.pt')
        print("✅ YOLO modeli başarıyla yüklendi")

        # Model bilgilerini göster
        print(f"✅ Model sınıf sayısı: {len(model.names)}")
        print(f"✅ İlk 5 sınıf: {list(model.names.values())[:5]}")

        return True
    except Exception as e:
        print(f"❌ YOLO modeli test hatası: {e}")
        print("   Model dosyasının mevcut olup olmadığını kontrol edin")
        return False

def test_webcam():
    """Webcam'in erişilebilir olup olmadığını test eder"""
    print("\n📹 Webcam testi başlatılıyor...")

    try:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Webcam açılamadı")
            return False

        # Webcam özelliklerini al
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        print(f"✅ Webcam başarıyla açıldı: {width}x{height}, {fps} FPS")

        # Test karesi oku
        ret, frame = cap.read()
        if ret:
            print(f"✅ Test karesi okundu: {frame.shape}")
        else:
            print("❌ Test karesi okunamadı")
            return False

        cap.release()
        return True

    except Exception as e:
        print(f"❌ Webcam test hatası: {e}")
        return False

def test_file_operations():
    """Dosya işlemlerini test eder"""
    print("\n📁 Dosya işlemleri testi başlatılıyor...")

    try:
        # Test klasörlerini kontrol et
        import os

        if os.path.exists('test_files'):
            print("✅ test_files klasörü mevcut")
        else:
            print("⚠️  test_files klasörü bulunamadı")

        if os.path.exists('results'):
            print("✅ results klasörü mevcut")
        else:
            print("⚠️  results klasörü bulunamadı")

        # Test dosyası oluştur
        test_file = 'test_files/test.txt'
        with open(test_file, 'w') as f:
            f.write("Bu bir test dosyasıdır.")

        if os.path.exists(test_file):
            print("✅ Test dosyası başarıyla oluşturuldu")
            os.remove(test_file)  # Test dosyasını sil
        else:
            print("❌ Test dosyası oluşturulamadı")
            return False

        return True
    except Exception as e:
        print(f"❌ Dosya işlemleri test hatası: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🧪 Object Detection Sistemi - Kapsamlı Test")
    print("=" * 60)

    tests = [
        ("Import Testleri", test_imports),
        ("OpenCV Testleri", test_opencv),
        ("YOLO Model Testi", test_yolo),
        ("Webcam Testi", test_webcam),
        ("Dosya İşlemleri Testi", test_file_operations),
    ]

    results = []
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
        if result:
            passed += 1

    # Sonuç özeti
    print(f"\n{'='*60}")
    print("📊 TEST SONUÇLARI ÖZETİ")
    print(f"{'='*60}")
    print(f"Toplam Test: {total}")
    print(f"Başarılı: {passed}")
    print(f"Başarısız: {total - passed}")
    print(f"Başarı Oranı: {passed/total*100:.1f}%")

    print("\nDetaylı Sonuçlar:")
    for test_name, result in results:
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"  {test_name}: {status}")

    if passed == total:
        print(f"\n🎉 Tüm testler başarıyla geçti! Sistem kullanıma hazır.")
        print(f"🚀 Artık object detection işlemlerine başlayabilirsiniz.")
    else:
        print(f"\n⚠️  Bazı testler başarısız oldu. Lütfen sorunları giderin.")
        print(f"💡 Daha fazla yardım için README.md dosyasını inceleyin.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
