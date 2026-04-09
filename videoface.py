import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ================== AYARLAR ==================
TFLITE_MODEL_PATH = "newint8.onnx" 
DB_PATH = "known_faces_embeddings.npz"  
YOLO_IMG_SIZE = 640                     
YOLO_DET_THRESHOLD = 0.40               
RECOG_THRESHOLD = 0.50                  
PADDING_RATIO = 0.25                    

# RTSP Stream Ayarları
RTSP_URL = "rtsp://10.69.36.232:8554/test"  # VLC'de belirlediğiniz port ve path'e göre güncelleyin
# Gecikmeyi düşürmek için OpenCV FFMPEG ayarları
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
# ============================================

# 1) Modeli Yükle
print(f"YOLO TFLite modeli yukleniyor ({TFLITE_MODEL_PATH})...")
yolo = YOLO(TFLITE_MODEL_PATH, task="detect")

print("InsightFace 'buffalo_s' yukleniyor...")
app = FaceAnalysis(name="buffalo_s", root=".", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(320, 320))

# 2) Veritabanını Yükle
try:
    db = np.load(DB_PATH, allow_pickle=True)
    DB_EMBS = db["encodings"]          
    DB_NAMES = db["names"]             
    print(f"Veritabani yuklendi. Kayitli kisi sayisi: {len(DB_NAMES)}")
except Exception as e:
    raise SystemExit(f"Veritabani yuklenemedi: {e}")

def cosine_similarity(a, b):
    return np.dot(b, a)

def predict_identity(emb):
    sims = cosine_similarity(emb, DB_EMBS)  
    idx = int(np.argmax(sims))
    best_sim = float(sims[idx])
    if best_sim >= RECOG_THRESHOLD:
        return str(DB_NAMES[idx]), best_sim
    return "Unknown", best_sim

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# 3) RTSP Bağlantısını Başlat
print(f"RTSP Stream'e baglaniliyor: {RTSP_URL}")
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# RTSP stream'lerde buffer birikmesini ve lag oluşmasını engellemek için buffer size 1 yapılır
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("HATA: RTSP stream acilamadi. IP, Port ve VLC yayininin acik oldugundan emin olun.")
    exit()

print("\n--- CANLI TANIMA SISTEMI AKTIF (Terminale Cikti Verilecek) ---")
print("Sistemi durdurmak icin 'Ctrl + C' tuslarina basin.\n")

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Frame okunamadi. Baglanti kopmus veya yayin durmus olabilir. Yeniden deneniyor...")
            time.sleep(1)
            continue
            
        frame_count += 1
        H, W = frame.shape[:2]

        # Her frame'de tespit edilen kişileri bir listeye toplayalım
        detected_people = []

        # YOLO Inference
        results = yolo(frame, imgsz=YOLO_IMG_SIZE, verbose=False)
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < YOLO_DET_THRESHOLD:
                    continue

                # Orijinal Bounding Box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_w = x2 - x1
                box_h = y2 - y1

                # Padding Eklenmesi
                pad_x = int(box_w * PADDING_RATIO)
                pad_y = int(box_h * PADDING_RATIO)

                px1 = clamp(x1 - pad_x, 0, W - 1)
                py1 = clamp(y1 - pad_y, 0, H - 1)
                px2 = clamp(x2 + pad_x, 0, W - 1)
                py2 = clamp(y2 + pad_y, 0, H - 1)

                roi = frame[py1:py2, px1:px2]
                if roi.size == 0: continue

                # InsightFace Analizi
                faces = app.get(roi)
                
                if faces:
                    f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                    emb = f.normed_embedding  
                    name, score = predict_identity(emb)
                    detected_people.append(f"{name} (Score: {score:.2f})")
                else:
                    detected_people.append(f"Yuz Bulunamadi (YOLO Conf: {conf:.2f})")

        # Terminal Çıktısı (Saniyede yaklaşık 1 kez bilgi vermek için)
        current_time = time.time()
        if current_time - start_time >= 1.0: # Her 1 saniyede bir çıktı ver
            if detected_people:
                print(f"[Frame {frame_count}] Tespit Edilenler: {', '.join(detected_people)}")
            else:
                pass # Kimse yoksa terminali boşuna meşgul etmemek için pass geçiyoruz. İsterseniz print("Kimse yok") yazabilirsiniz.
                
            start_time = current_time

except KeyboardInterrupt:
    print("\nKullanici tarafindan durduruldu (Ctrl+C).")

finally:
    cap.release()
    print("Baglanti sonlandirildi.")