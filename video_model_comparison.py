import cv2
import time
import statistics
from ultralytics import YOLO

# Kullanılacak modeller
MODELS = {
    'YOLOv8n': 'yolov8n.pt',
    'YOLOv10n': 'yolov10n.pt',
    'YOLO11n': 'yolo11n.pt',
    'YOLO12n': 'yolo12n.pt'
}

# Video dosyasının yolu
VIDEO_PATH = 'demo_video.mp4'

def test_video_with_models():
    """Çoklu modelleri video üzerinde test eder ve karşılaştırır"""

    # Video dosyasını aç
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Hata: {VIDEO_PATH} açılamadı.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Toplam kare sayısı: {total_frames}")
    print("=" * 50)

    # Her model için sonuçları saklayacak sözlükler
    results = {}

    for model_name, model_path in MODELS.items():
        print(f"\n{model_name} test ediliyor...")

        # Modeli yükle
        model = YOLO(model_path)

        # Bu model için verileri sakla
        fps_values = []
        total_objects = 0
        frame_count = 0

        # Video başından itibaren işle
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Kare sayısını artır
            frame_count += 1

            # FPS hesabı için başlangıç zamanı
            start_time = time.time()

            # Nesne tespiti yap
            results_model = model(frame, stream=True, imgsz=640, verbose=False)

            # Sonuçları işle
            detected_objects = 0
            for r in results_model:
                detected_objects += len(r.boxes)

            total_objects += detected_objects

            # FPS hesapla
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            fps_values.append(fps)

            # Her 100 karede bir durum göster
            if frame_count % 100 == 0:
                print(f"  {frame_count}/{total_frames} kare işlendi...")

        # Bu model için istatistikleri hesapla
        avg_fps = statistics.mean(fps_values)
        min_fps = min(fps_values)
        max_fps = max(fps_values)
        avg_objects = total_objects / frame_count

        results[model_name] = {
            'avg_fps': avg_fps,
            'min_fps': min_fps,
            'max_fps': max_fps,
            'avg_objects': avg_objects,
            'total_objects': total_objects,
            'processed_frames': frame_count
        }

        print(f"  Ortalama FPS: {avg_fps:.2f}")
        print(f"  FPS aralığı: {min_fps:.2f} - {max_fps:.2f}")
        print(f"  Kare başına ortalama nesne: {avg_objects:.2f}")
        print(f"  Toplam tespit edilen nesne: {total_objects}")

    # Sonuçları göster
    print("\n" + "=" * 70)
    print("KARŞILAŞTIRMA SONUÇLARI")
    print("=" * 70)
    print(f"{'Model':<12} {'Ort.FPS':<10} {'Min FPS':<10} {'Max FPS':<10} {'Nesne/Kare':<12} {'Toplam Nesne':<12}")
    print("-" * 70)

    for model_name in results:
        data = results[model_name]
        print(f"{model_name:<12} {data['avg_fps']:<10.2f} {data['min_fps']:<10.2f} {data['max_fps']:<10.2f} {data['avg_objects']:<12.2f} {data['total_objects']:<12}")

    print("-" * 70)

    # En iyi performansı bul
    best_fps = max(results.items(), key=lambda x: x[1]['avg_fps'])
    best_object_detection = max(results.items(), key=lambda x: x[1]['avg_objects'])

    print("EN İYİ PERFORMANS:")
    print(f"  En yüksek FPS: {best_fps[0]} ({best_fps[1]['avg_fps']:.2f} FPS)")
    print(f"  En çok nesne tespiti: {best_object_detection[0]} ({best_object_detection[1]['avg_objects']:.2f} nesne/kare)")

    # Kaynakları temizle
    cap.release()

if __name__ == "__main__":
    test_video_with_models()
