#!/usr/bin/env python3
"""
Multi-Model Webcam Object Detection
4 farklı YOLO modelini aynı anda çalıştıran webcam uygulaması
Ekran 2x2 grid şeklinde bölünür ve her model için ayrı FPS sayacı gösterilir
"""

import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
import os

class MultiModelDetector:
    def __init__(self, model_names=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt']):
        """
        Çoklu model nesne tespit sistemi

        Args:
            model_names (list): Kullanılacak model isimleri
        """
        self.model_names = model_names
        self.models = []
        self.load_models()

        # Webcam ayarları
        self.cap = None

        # FPS hesaplama için değişkenler
        self.fps_counters = [0.0] * len(model_names)
        self.prev_times = [0.0] * len(model_names)
        self.frame_counts = [0] * len(model_names)

        # Model etiketleri
        self.model_labels = [
            f"Model {i+1}: {name.replace('.pt', '').upper()}"
            for i, name in enumerate(model_names)
        ]

    def load_models(self):
        """Modelleri yükler"""
        print("YOLO modelleri yükleniyor...")

        for i, model_name in enumerate(self.model_names):
            try:
                if os.path.exists(model_name):
                    print(f"  {model_name} - Yerelden yükleniyor...")
                    model = YOLO(model_name)
                else:
                    print(f"  {model_name} - İnternetten indiriliyor...")
                    model = YOLO(model_name)

                self.models.append(model)
                print(f"    ✓ Model {i+1} yüklendi: {model_name}")

            except Exception as e:
                print(f"    ✗ Model {i+1} yüklenemedi: {model_name} - {e}")
                # Yerine varsayılan model kullan
                try:
                    print("    → Varsayılan model (yolov8n.pt) kullanılıyor...")
                    model = YOLO('yolov8n.pt')
                    self.models.append(model)
                    self.model_names[i] = 'yolov8n.pt'
                    print(f"    ✓ Varsayılan model yüklendi: yolov8n.pt")
                except Exception as e2:
                    print(f"    ✗ Varsayılan model de yüklenemedi: {e2}")
                    self.models.append(None)

    def create_grid_frame(self, frames):
        """
        Kareleri 2x2 grid şeklinde birleştirir

        Args:
            frames (list): İşlenmiş kareler

        Returns:
            numpy.ndarray: Birleştirilmiş görüntü
        """
        if len(frames) != 4:
            return None

        # Kare boyutlarını kontrol et
        height, width = frames[0].shape[:2]

        # 2x2 grid oluştur
        top_row = np.hstack([frames[0], frames[1]])
        bottom_row = np.hstack([frames[2], frames[3]])
        combined_frame = np.vstack([top_row, bottom_row])

        return combined_frame

    def add_model_info(self, frame, model_index, fps):
        """
        Kareye model bilgilerini ve FPS'i ekler

        Args:
            frame: İşlenmiş kare
            model_index (int): Model indeksi
            fps (float): FPS değeri
        """
        # Model etiketi
        label = self.model_labels[model_index]
        cv2.putText(frame, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # FPS bilgisi
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Kare sayacı
        frame_text = f"Frame: {self.frame_counts[model_index]}"
        cv2.putText(frame, frame_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def process_frame_with_model(self, frame, model_index):
        """
        Belirtilen model ile kareyi işler

        Args:
            frame: Orijinal kare
            model_index (int): Model indeksi

        Returns:
            numpy.ndarray: İşlenmiş kare
        """
        if self.models[model_index] is None:
            # Model yüklenememişse boş kare döndür
            empty_frame = np.zeros_like(frame)
            cv2.putText(empty_frame, f"Model {model_index+1}: Yuklenemedi",
                       (50, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            return empty_frame

        # FPS hesaplama
        current_time = time.time()

        if self.prev_times[model_index] > 0:
            self.fps_counters[model_index] = 1 / (current_time - self.prev_times[model_index])

        self.prev_times[model_index] = current_time
        self.frame_counts[model_index] += 1

        # Nesne tespiti
        results = self.models[model_index](frame, conf=0.5)

        # Sonuçları çiz
        annotated_frame = results[0].plot()

        # Model bilgilerini ekle
        self.add_model_info(annotated_frame, model_index, self.fps_counters[model_index])

        return annotated_frame

    def run_multi_model_detection(self, show_window=True):
        """
        Çoklu model ile webcam tespitini çalıştırır

        Args:
            show_window (bool): Pencere gösterilsin mi
        """
        print("Multi-Model Webcam Nesne Tespiti Başlatılıyor...")
        print("=" * 50)
        print("Kullanılan modeller:")
        for i, name in enumerate(self.model_names):
            print(f"  {i+1}. {name}")
        print("=" * 50)
        print("Çıkmak için 'q' tuşuna basın")

        # Webcam'i aç
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Hata: Webcam açılamadı!")
            return

        # Ana döngü
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Her model için kareyi işle (paralel processing için threading kullanılabilir)
                processed_frames = []

                for i in range(len(self.models)):
                    processed_frame = self.process_frame_with_model(frame.copy(), i)
                    processed_frames.append(processed_frame)

                # Grid oluştur
                if len(processed_frames) == 4:
                    combined_frame = self.create_grid_frame(processed_frames)

                    if show_window:
                        # Sonuçları göster
                        cv2.imshow('Multi-Model Object Detection (2x2 Grid)', combined_frame)

                        # 'q' tuşuna basılırsa çık
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                # Kısa bir bekleme
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nİşlem kullanıcı tarafından durduruldu")

        except Exception as e:
            print(f"Hata oluştu: {e}")

        finally:
            # Temizlik
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("\nMulti-Model Webcam nesne tespiti durduruldu")

    def get_model_info(self):
        """Model bilgileri döndürür"""
        info = []
        for i, model in enumerate(self.models):
            if model is not None:
                info.append({
                    'index': i + 1,
                    'name': self.model_names[i],
                    'loaded': True,
                    'fps': self.fps_counters[i],
                    'frames_processed': self.frame_counts[i]
                })
            else:
                info.append({
                    'index': i + 1,
                    'name': self.model_names[i],
                    'loaded': False,
                    'fps': 0.0,
                    'frames_processed': 0
                })

        return info

def main():
    """Ana fonksiyon"""
    # Kullanılabilir modeller listesi
    available_models = [
        'yolov8n.pt',  # Nano model (en hızlı)
        'yolov8s.pt',  # Small model
        'yolov8m.pt',  # Medium model
        'yolov8l.pt',  # Large model (en doğru ama yavaş)
        'yolov8x.pt',  # Extra large model
        'yolo11n.pt',  # YOLOv11 Nano
        'yolo11s.pt',  # YOLOv11 Small
        'yolo11m.pt',  # YOLOv11 Medium
        'yolo11l.pt',  # YOLOv11 Large
    ]

    print("Multi-Model Webcam Object Detection")
    print("=" * 40)
    print("Mevcut modeller:")
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")

    print("\nVarsayılan modeller kullanılacak: yolov8n, yolov8s, yolov8m, yolov8l")
    print("Başka modeller kullanmak için kodu düzenleyin.")

    # Multi-model detector oluştur
    detector = MultiModelDetector()

    # Model bilgilerini göster
    print("\nModel yükleme durumu:")
    for info in detector.get_model_info():
        status = "✓ Yüklendi" if info['loaded'] else "✗ Yüklenemedi"
        print(f"  Model {info['index']}: {info['name']} - {status}")

    input("\nDevam etmek için Enter'a basın...")

    # Çoklu model tespitini başlat
    detector.run_multi_model_detection()

if __name__ == "__main__":
    main()
