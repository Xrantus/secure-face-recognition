import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

class ObjectDetector:
    def __init__(self, model_path='yolo11n.pt'):
        """
        Object detection sınıfı

        Args:
            model_path (str): YOLO model dosya yolu
        """
        self.model = YOLO(model_path)
        print(f"YOLO modeli yüklendi: {model_path}")

    def download_yolo11_model(self, model_name='yolo11n.pt'):
        """
        YOLO v11 modelini indirir

        Args:
            model_name (str): İndirilecek model adı
        """
        try:
            from ultralytics import YOLO
            print(f"{model_name} modeli indiriliyor...")
            model = YOLO(model_name)
            print(f"{model_name} modeli başarıyla indirildi!")
            return model_name
        except Exception as e:
            print(f"Model indirme hatası: {e}")
            print("yolov8n.pt modeli kullanılıyor...")
            return 'yolov8n.pt'

    def detect_image(self, image_path, conf_threshold=0.5, save_result=True):
        """
        Tek bir görüntüde nesne tespiti yapar

        Args:
            image_path (str): Görüntü dosya yolu
            conf_threshold (float): Güven eşiği (0-1 arası)
            save_result (bool): Sonucu kaydetmek için

        Returns:
            dict: Tespit sonuçları
        """
        if not os.path.exists(image_path):
            print(f"Hata: {image_path} dosyası bulunamadı!")
            return None

        # Görüntüyü yükle
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hata: {image_path} yüklenemedi!")
            return None

        # Nesne tespiti
        results = self.model(image, conf=conf_threshold)

        # Sonuçları görselleştir ve kaydet
        if save_result:
            # Sonuçları çiz
            annotated_frame = results[0].plot()

            # Sonuç dosya adı oluştur
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = f"results/{name}_detected{ext}"

            # Results klasörü oluştur
            os.makedirs("results", exist_ok=True)

            # Sonucu kaydet
            cv2.imwrite(output_path, annotated_frame)
            print(f"Sonuç kaydedildi: {output_path}")

        return results

    def detect_video(self, video_path, conf_threshold=0.5, save_result=True):
        """
        Video dosyasında nesne tespiti yapar

        Args:
            video_path (str): Video dosya yolu
            conf_threshold (float): Güven eşiği
            save_result (bool): Sonucu kaydetmek için

        Returns:
            str: Çıktı video dosya yolu
        """
        if not os.path.exists(video_path):
            print(f"Hata: {video_path} dosyası bulunamadı!")
            return None

        # Video dosyasını aç
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Hata: {video_path} açılamadı!")
            return None

        # Video özellikleri
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video bilgileri: {width}x{height}, {fps} FPS, {total_frames} kare")

        # Çıktı video ayarları
        if save_result:
            filename = os.path.basename(video_path)
            name, ext = os.path.splitext(filename)
            output_path = f"results/{name}_detected{ext}"

            os.makedirs("results", exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Her 10 karede bir göster
            if processed_frames % 10 == 0:
                print(f"İşlenen kare: {processed_frames}/{total_frames}")

            # Nesne tespiti
            results = self.model(frame, conf=conf_threshold)

            # Sonuçları çiz
            annotated_frame = results[0].plot()

            if save_result:
                out.write(annotated_frame)

            processed_frames += 1

        cap.release()
        if save_result:
            out.release()
            print(f"İşlenmiş video kaydedildi: {output_path}")
            return output_path

        return None

    def detect_webcam(self, conf_threshold=0.5, show_window=True):
        """
        Webcam'den gerçek zamanlı nesne tespiti

        Args:
            conf_threshold (float): Güven eşiği
            show_window (bool): Pencere gösterilsin mi
        """
        print("Webcam'den nesne tespiti başlatılıyor...")
        print("Çıkmak için 'q' tuşuna basın")

        cap = cv2.VideoCapture(0)  # 0 = default webcam

        if not cap.isOpened():
            print("Hata: Webcam açılamadı!")
            return

        # FPS hesaplama için değişkenler
        prev_time = 0
        fps = 0

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = cv2.getTickCount() / cv2.getTickFrequency()

            # FPS hesaplama
            if prev_time > 0:
                fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Nesne tespiti
            results = self.model(frame, conf=conf_threshold)

            # Sonuçları çiz
            annotated_frame = results[0].plot()

            # FPS'yi sağ üst köşeye ekle
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated_frame, fps_text, (annotated_frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Tespit edilen objeleri konsola yazdır
            detected_objects = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = results[0].names[class_id]
                    confidence = float(box.conf[0])
                    detected_objects.append(f"{class_name} ({confidence:.2f})")

            if detected_objects:
                objects_str = ", ".join(detected_objects)
                print(f"Frame {frame_count}: [{objects_str}]")
            else:
                print(f"Frame {frame_count}: [Hiç nesne tespit edilemedi]")

            frame_count += 1

            if show_window:
                cv2.imshow('Object Detection - Webcam', annotated_frame)

            # 'q' tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Webcam nesne tespiti durduruldu")

    def show_detection_info(self, results):
        """
        Tespit sonuçlarını gösterir

        Args:
            results: YOLO tespit sonuçları
        """
        if results is None:
            return

        print("\n=== NESNE TESPİT SONUÇLARI ===")
        for i, result in enumerate(results):
            print(f"\nGörüntü {i+1}:")
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                print(f"Toplam tespit edilen nesne: {len(boxes)}")

                # Sınıf dağılımı
                class_counts = {}
                confidence_scores = []

                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])

                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    confidence_scores.append(confidence)

                print("Nesne türleri:")
                for class_name, count in class_counts.items():
                    print(f"  - {class_name}: {count}")

                # Güven skorları istatistikleri
                if confidence_scores:
                    avg_conf = sum(confidence_scores) / len(confidence_scores)
                    max_conf = max(confidence_scores)
                    min_conf = min(confidence_scores)
                    print(f"\nGüven skorları istatistikleri:")
                    print(f"  - Ortalama: {avg_conf:.3f}")
                    print(f"  - Maksimum: {max_conf:.3f}")
                    print(f"  - Minimum: {min_conf:.3f}")
            else:
                print("Hiç nesne tespit edilemedi")

        print("=" * 30)

    def generate_report(self, results, output_file="detection_report.txt"):
        """
        Detaylı rapor oluşturur

        Args:
            results: YOLO tespit sonuçları
            output_file (str): Rapor dosya adı
        """
        if results is None:
            print("Rapor oluşturulamadı: Sonuç bulunamadı")
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("OBJECT DETECTION RAPORU\n")
            f.write("=" * 50 + "\n\n")

            total_objects = 0
            all_class_counts = {}

            for i, result in enumerate(results):
                f.write(f"Görüntü/Video {i+1}:\n")
                f.write("-" * 20 + "\n")

                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    object_count = len(boxes)
                    total_objects += object_count

                    f.write(f"Toplam tespit edilen nesne: {object_count}\n")

                    # Sınıf dağılımı
                    class_counts = {}
                    confidence_scores = []

                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        confidence = float(box.conf[0])

                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        confidence_scores.append(confidence)

                        # Genel sınıfa ekle
                        all_class_counts[class_name] = all_class_counts.get(class_name, 0) + 1

                    f.write("\nNesne türleri ve sayıları:\n")
                    for class_name, count in class_counts.items():
                        f.write(f"  - {class_name}: {count}\n")

                    # Güven skorları
                    if confidence_scores:
                        avg_conf = sum(confidence_scores) / len(confidence_scores)
                        max_conf = max(confidence_scores)
                        min_conf = min(confidence_scores)

                        f.write("\nGüven skorları istatistikleri:\n")
                        f.write(f"  - Ortalama: {avg_conf:.3f}\n")
                        f.write(f"  - Maksimum: {max_conf:.3f}\n")
                        f.write(f"  - Minimum: {min_conf:.3f}\n")
                else:
                    f.write("Hiç nesne tespit edilemedi\n")

                f.write("\n")

            # Genel özet
            f.write("GENEL ÖZET\n")
            f.write("=" * 20 + "\n")
            f.write(f"Toplam işlenen görüntü/video: {len(results)}\n")
            f.write(f"Toplam tespit edilen nesne: {total_objects}\n")

            if all_class_counts:
                f.write("\nGenel nesne dağılımı:\n")
                for class_name, count in sorted(all_class_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_objects) * 100
                    f.write(f"  - {class_name}: {count} ({percentage:.1f}%)\n")

            f.write(f"\nRapor oluşturulma tarihi: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"Rapor kaydedildi: {output_file}")

    def save_detailed_results(self, results, output_dir="results"):
        """
        Detaylı sonuçları JSON formatında kaydeder

        Args:
            results: YOLO tespit sonuçları
            output_dir (str): Çıktı klasörü
        """
        import json
        import os

        os.makedirs(output_dir, exist_ok=True)

        detailed_data = {
            "summary": {
                "total_images": len(results),
                "total_objects": 0,
                "processing_date": __import__('datetime').datetime.now().isoformat()
            },
            "results": []
        }

        total_objects = 0
        all_class_counts = {}

        for i, result in enumerate(results):
            image_result = {
                "image_index": i + 1,
                "objects": []
            }

            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                object_count = len(boxes)
                total_objects += object_count

                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                    obj_data = {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": bbox
                    }

                    image_result["objects"].append(obj_data)

                    # Genel sınıfa ekle
                    all_class_counts[class_name] = all_class_counts.get(class_name, 0) + 1

                image_result["object_count"] = object_count
            else:
                image_result["object_count"] = 0

            detailed_data["results"].append(image_result)

        detailed_data["summary"]["total_objects"] = total_objects

        if all_class_counts:
            detailed_data["summary"]["class_distribution"] = all_class_counts

        # JSON dosyası kaydet
        json_file = os.path.join(output_dir, "detailed_results.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)

        print(f"Detaylı sonuçlar JSON olarak kaydedildi: {json_file}")

def main():
    """
    Ana fonksiyon - kullanım örnekleri
    """
    detector = ObjectDetector()

    print("Object Detection Sistemi Hazır!")
    print("Kullanım seçenekleri:")
    print("1. Görüntü dosyası seçin")
    print("2. Video dosyası seçin")
    print("3. Webcam kullanın")

    choice = input("Seçiminizi yapın (1/2/3): ")

    if choice == "1":
        image_path = input("Görüntü dosya yolunu girin: ")
        results = detector.detect_image(image_path)
        detector.show_detection_info(results)

    elif choice == "2":
        video_path = input("Video dosya yolunu girin: ")
        output_path = detector.detect_video(video_path)
        print(f"İşlenmiş video: {output_path}")

    elif choice == "3":
        detector.detect_webcam()

    else:
        print("Geçersiz seçim!")

if __name__ == "__main__":
    main()
