import cv2
import numpy as np
import time
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis
from picamera2 import Picamera2

# --- YAPILANDIRMA ---
YOLO_MODEL_PATH = "face_yolo11_widerface_best_int8_OPTIMIZED.onnx"
DATABASE_PATH = "known_faces_embeddings.npz"
IMG_SIZE = 640

def load_database(path):
    try:
        data = np.load(path, allow_pickle=True)
        return data['embeddings'], data['names']
    except Exception as e:
        print(f"Veritabani yuklenemedi: {e}")
        return None, None

# 1. Modelleri Yukle
print(f"YOLO yukleniyor: {YOLO_MODEL_PATH}")
yolo_model = YOLO(YOLO_MODEL_PATH, task='detect')

print("InsightFace yukleniyor...")
face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(160, 160))

known_embeddings, known_names = load_database(DATABASE_PATH)
if known_embeddings is not None:
    print(f"Veritabani: {len(known_names)} kisi yuklendi.")

# 2. Picamera2 Baslatma (RPi5 Ozel)
print("Kamera baslatiliyor...")
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Değişkenler
running = True
fps_n = 0
fps_t0 = time.time()
total_frames = 0
t0 = time.time()

print("Sistem aktif. Cikis icin Ctrl+C kullanin.")

try:
    while running:
        # Kameradan frame al
        frame = picam2.capture_array()
        
        if frame is None:
            continue

        # YOLO ile Yüz Tespiti
        results = yolo_model.predict(frame, conf=0.25, imgsz=IMG_SIZE, verbose=False)
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # Yüzü kırp ve InsightFace'e gönder
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                
                # Yüz Tanıma (Embedding çıkarma)
                faces = face_app.get(frame) # Genelde tüm frame üzerinden landmark bakar
                
                # Buraya kendi recognition mantığını (cosine similarity vb.) ekleyebilirsin
                # Örnek: cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # FPS Hesapla
        fps_n += 1
        total_frames += 1
        now = time.time()
        if now - fps_t0 >= 1.0:
            print(f"Anlik FPS: {fps_n / (now - fps_t0):.2f}", end="\r")
            fps_n = 0
            fps_t0 = now

except KeyboardInterrupt:
    print("\nDurduruluyor...")

finally:
    # Kaynakları serbest bırak
    picam2.stop()
    print(f"Kapatildi. Sure: {time.time()-t0:.1f}s | Ort. FPS: {total_frames/(time.time()-t0):.2f}")