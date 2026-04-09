import cv2
import time
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ================== AYARLAR ==================
# Model ismini TFLite olarak guncelledik!
TFLITE_MODEL_PATH = "face_yolo11_widerface_best_int8.tflite" 
DB_PATH = "known_faces_embeddings.npz"  
YOLO_IMG_SIZE = 640                     # Egitim boyutumuz
YOLO_DET_THRESHOLD = 0.40               # Yeni model guclu oldugu icin esigi artirabiliriz
RECOG_THRESHOLD = 0.50                  
CAM_INDEX = 0
PADDING_RATIO = 0.25                    # %25 ekstra genisletme (InsightFace hizalamasi icin sart)
# ============================================

# 1) Modeli Yukle
print(f"YOLO TFLite modeli yukleniyor ({TFLITE_MODEL_PATH})...")
yolo = YOLO(TFLITE_MODEL_PATH, task="detect")

print("InsightFace 'buffalo_s' yukleniyor...")
app = FaceAnalysis(name="buffalo_s", root=".", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(320, 320))

# 2) Veritabanini Yukle
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

# Kamerayi Baslat
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Canli tanima sistemi aktif.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]

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

            # === PADDING EKLENMESI (Kritik Duzeltme) ===
            pad_x = int(box_w * PADDING_RATIO)
            pad_y = int(box_h * PADDING_RATIO)

            px1 = clamp(x1 - pad_x, 0, W - 1)
            py1 = clamp(y1 - pad_y, 0, H - 1)
            px2 = clamp(x2 + pad_x, 0, W - 1)
            py2 = clamp(y2 + pad_y, 0, H - 1)

            roi = frame[py1:py2, px1:px2]
            if roi.size == 0: continue

            # InsightFace artik daha genis bir alan goruyor, landmark'lari (goz, burun) bulabilir
            faces = app.get(roi)
            
            kutu_rengi = (60, 200, 255) # Varsayilan Sari
            etiket = f"Face {conf:.2f}" 

            if faces:
                # ROI icindeki en buyuk yuzu al
                f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                emb = f.normed_embedding  
                name, score = predict_identity(emb)
                
                if name != "Unknown":
                    kutu_rengi = (0, 255, 0) # Yesil
                    etiket = f"{name} {score:.2f}"
                else:
                    kutu_rengi = (0, 0, 255) # Kirmizi
                    etiket = f"Unknown {score:.2f}"

            # Ekrana cizdir (Orijinal x1, y1 kordinatlarina ciziyoruz ki kutu kaymasin)
            cv2.rectangle(frame, (x1, y1), (x2, y2), kutu_rengi, 2)
            cv2.putText(frame, etiket, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, kutu_rengi, 2)

    cv2.imshow("Edge-Optimized Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()