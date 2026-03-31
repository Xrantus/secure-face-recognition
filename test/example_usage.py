"""
Object Detection Sistemi - Örnek Kullanım
========================================

Bu script object detection sisteminin nasıl kullanılacağını gösterir.
"""

from object_detection import ObjectDetector

def example_image_detection():
    """Görüntü dosyası ile örnek kullanım"""
    print("=== GÖRÜNTÜ TESPİTİ ÖRNEĞİ ===")

    # Object detector oluştur
    detector = ObjectDetector()

    # Örnek bir görüntü ile test et (kendi görüntünüzü kullanın)
    # image_path = "test_images/sample.jpg"
    image_path = input("Test edilecek görüntü dosya yolunu girin: ")

    # Nesne tespiti yap
    results = detector.detect_image(image_path, conf_threshold=0.3)

    # Sonuçları göster
    detector.show_detection_info(results)

    print("Görüntü testi tamamlandı!\n")

def example_video_detection():
    """Video dosyası ile örnek kullanım"""
    print("=== VİDEO TESPİTİ ÖRNEĞİ ===")

    detector = ObjectDetector()

    # Örnek bir video ile test et (kendi videonuzu kullanın)
    # video_path = "test_videos/sample.mp4"
    video_path = input("Test edilecek video dosya yolunu girin: ")

    # Video işleme
    output_path = detector.detect_video(video_path, conf_threshold=0.3)

    print(f"İşlenmiş video kaydedildi: {output_path}\n")

def example_webcam_detection():
    """Webcam ile örnek kullanım"""
    print("=== WEBCAM TESPİTİ ÖRNEĞİ ===")

    detector = ObjectDetector()

    print("Webcam testi başlatılıyor...")
    print("Çıkmak için 'q' tuşuna basın")

    # Webcam'den gerçek zamanlı tespit
    detector.detect_webcam(conf_threshold=0.3)

    print("Webcam testi tamamlandı!\n")

def main():
    """Ana menü"""
    print("🎯 Object Detection Sistemi - Örnek Kullanım")
    print("=" * 50)

    while True:
        print("\nLütfen bir seçenek seçin:")
        print("1. Görüntü dosyası ile test")
        print("2. Video dosyası ile test")
        print("3. Webcam ile gerçek zamanlı test")
        print("4. Çıkış")

        choice = input("\nSeçiminiz (1-4): ")

        if choice == "1":
            example_image_detection()
        elif choice == "2":
            example_video_detection()
        elif choice == "3":
            example_webcam_detection()
        elif choice == "4":
            print("Program sonlandırılıyor...")
            break
        else:
            print("❌ Geçersiz seçim! Lütfen 1-4 arası bir sayı girin.")

if __name__ == "__main__":
    main()
