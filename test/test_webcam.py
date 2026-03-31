#!/usr/bin/env python3
"""
Webcam ile gerçek zamanlı nesne tespiti testi
YOLO v11 kullanarak FPS gösterimi ve anlık konsol loglama
"""

from object_detection import ObjectDetector
import cv2

def test_webcam_detection():
    """
    Webcam ile nesne tespiti testi
    """
    print("YOLO v11 Webcam Nesne Tespiti Testi")
    print("=" * 40)

    # Object detector oluştur
    detector = ObjectDetector()

    print("\nTest başlatılıyor...")
    print("Çıkmak için 'q' tuşuna basın")

    try:
        # Webcam ile gerçek zamanlı tespit
        detector.detect_webcam(conf_threshold=0.5, show_window=True)

    except KeyboardInterrupt:
        print("\nTest kullanıcı tarafından durduruldu")

    except Exception as e:
        print(f"Test sırasında hata oluştu: {e}")

    print("\nTest tamamlandı!")

if __name__ == "__main__":
    test_webcam_detection()
