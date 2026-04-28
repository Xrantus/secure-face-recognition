"""
Canli yuz tespiti + tanima — Raspberry Pi 5 (Headless) + Picamera2 + ONNX (YOLO INT8) + InsightFace.

YOLOv11 INT8 ONNX: kuantizasyon ham guven skorlarini dusurur; YOLO_DET_THRESHOLD ve YOLO_PRED_CONF
asagida INT8 telafisiyle ayarlanmistir.
"""
import os
import threading
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from ultralytics import YOLO

# Picamera2 import (RPi5'in resmi kamera kütüphanesi)
from picamera2 import Picamera2

# ========= Tek model (INT8 ONNX — RPi5 ORT + Ultralytics) =========
MODEL_PATH = "face_yolo11_widerface_best_int8_OPTIMIZED.onnx"

# INT8 telafisi: dusuk ham conf; on eleme cok dusuk, son esik 0.15–0.20 bandi
YOLO_IMG_SIZE = 640
YOLO_DET_THRESHOLD = 0.15
YOLO_PRED_CONF = 0.01
YOLO_IOU = 0.45

MAX_DET = 100
YOLO_TARGET_CLASS = None
YOLO_TARGET_CLASS_ID = None

DB_PATH = "known_faces_embeddings.npz"
RECOG_THRESHOLD = 0.50
MIN_FACE_SIZE = 35
LANDMARK_PAD = 0.20
# YOLO ROI'si zaten kucuk; 320 yerine 160 -> RPi CPU'da InsightFace ic ice detector darbogazi azalir
DET_SIZE = (160, 160)
FRAME_SKIP = 8


def _landmarks_list(kpss):
    if kpss is None:
        return []
    if isinstance(kpss, np.ndarray):
        if kpss.size == 0:
            return []
        if kpss.ndim == 2 and kpss.shape == (5, 2):
            return [kpss]
        if kpss.ndim == 3 and kpss.shape[1:] == (5, 2):
            return [kpss[i] for i in range(kpss.shape[0])]
        return []
    return [np.asarray(k) for k in kpss] if kpss else []


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def predict_identity(emb, db_embs, db_names):
    sims = np.dot(db_embs, emb)
    idx = int(np.argmax(sims))
    best_sim = float(sims[idx])
    best_name = str(db_names[idx])
    if best_sim >= RECOG_THRESHOLD:
        return best_name, best_sim
    return "Unknown", best_sim


_int8_only = "face_yolo11_widerface_best_int8.onnx"
if not os.path.isfile(MODEL_PATH):
    raise SystemExit(f"Model bulunamadi: {MODEL_PATH}")

yolo = YOLO(MODEL_PATH, task="detect")

print(
    f"YOLO (RPi5 INT8 ayarlari): {MODEL_PATH} | imgsz={YOLO_IMG_SIZE} | "
    f"predict conf>={YOLO_PRED_CONF} | son esik>={YOLO_DET_THRESHOLD} | NMS iou={YOLO_IOU}"
)

app = FaceAnalysis(name="buffalo_s", root=".", allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1, det_size=DET_SIZE)

try:
    db = np.load(DB_PATH, allow_pickle=True)
    DB_EMBS = db["encodings"]
    DB_NAMES = db["names"]
    DB_EMBS = DB_EMBS / np.linalg.norm(DB_EMBS, axis=1, keepdims=True)
    print(f"Veritabani: {DB_PATH} | {len(DB_NAMES)} kisi | InsightFace det_size={DET_SIZE}")
except Exception as e:
    raise SystemExit(f"Veritabani yuklenemedi: {e}") from e


def _predict_kw():
    return {
        "imgsz": YOLO_IMG_SIZE,
        "verbose": False,
        "conf": YOLO_PRED_CONF,
        "iou": YOLO_IOU,
        "max_det": MAX_DET,
    }


latest_frame = None
frame_lock = threading.Lock()
running = True
debug_names = True
logged_raw_boxes = False


# Picamera2 icin Thread Fonksiyonu
def frame_reader_thread(picam):
    global latest_frame, running
    while running:
        try:
            # Kameradan dogrudan Numpy dizisi (array) olarak RGB formatinda kare aliyoruz
            frame_rgb = picam.capture_array()
            # OpenCV BGR formatiyla calistigi icin renk uzayini ceviriyoruz
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            with frame_lock:
                latest_frame = frame_bgr
        except Exception as e:
            print(f"Kamera okuma hatasi: {e}")
            running = False
            break


# ========= Picamera2 Baslatma =========
try:
    print("Picamera2 baslatiliyor...")
    picam2 = Picamera2()
    # 640x480 cozunurlukte onizleme konfigürasyonu olustur
    config = picam2.create_preview_configuration({"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
except Exception as e:
    raise SystemExit(f"Picamera2 baslatilamadi. Hata: {e}")

reader_t = threading.Thread(target=frame_reader_thread, args=(picam2,), daemon=True)
reader_t.start()

while latest_frame is None and running:
    time.sleep(0.05)
print("Sistem Aktif! (Durdurmak icin terminalde CTRL+C yapin)\n")

frame_counter = 0
last_results = None
fps_t0 = time.time()
fps_n = 0
total_frames = 0
t0 = time.time()

# Headless (Arayüzsüz) sistem icin klavye kesmesi yakalayici (CTRL+C)
try:
    while running:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        frame = cv2.flip(frame, 1)
        fps_n += 1
        total_frames += 1

        if frame_counter % FRAME_SKIP == 0:
            last_results = [yolo.predict(frame, **_predict_kw())[0]]

        if last_results:
            for r in last_results:
                boxes = r.boxes
                if not logged_raw_boxes:
                    n = len(boxes) if boxes is not None else 0
                    best = 0.0
                    if n > 0:
                        try:
                            c = boxes.conf
                            if hasattr(c, "detach"):
                                c = c.detach().cpu().numpy()
                            else:
                                c = np.asarray(c)
                            best = float(np.max(c)) if c.size else 0.0
                        except Exception:
                            pass
                    print(f"[tespit] ham kutu: {n} | en yuksek conf: {best:.3f} (cizim icin >={YOLO_DET_THRESHOLD})")
                    logged_raw_boxes = True

                if boxes is None or len(boxes) == 0:
                    continue

                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])

                    if conf < YOLO_DET_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                    H, W = frame.shape[:2]
                    x1, x2 = clamp(x1, 0, W - 1), clamp(x2, 0, W - 1)
                    y1, y2 = clamp(y1, 0, H - 1), clamp(y2, 0, H - 1)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue

                    face_w, face_h = x2 - x1, y2 - y1
                    if min(face_w, face_h) < MIN_FACE_SIZE:
                        continue

                    pw = int(face_w * LANDMARK_PAD)
                    ph = int(face_h * LANDMARK_PAD)
                    roi = frame[
                        max(0, y1 - ph) : min(H, y2 + ph),
                        max(0, x1 - pw) : min(W, x2 + pw),
                    ]
                    
                    if roi.size == 0:
                        continue

                    _, kpss = app.det_model.detect(roi, max_num=1, metric="default")
                    lm_list = _landmarks_list(kpss)

                    if lm_list:
                        kps = lm_list[0]
                        aligned_face = face_align.norm_crop(roi, landmark=kps)
                        emb = app.models["recognition"].get_feat(aligned_face)[0]
                        emb = emb / np.linalg.norm(emb)
                        name, score = predict_identity(emb, DB_EMBS, DB_NAMES)
                        
                        # Eger taninan kisi veritabanindaysa terminale log bas
                        if name != "Unknown":
                            print(f"[BASARILI] {name} tespit edildi! (Skor: {score:.2f})")

        now = time.time()
        if now - fps_t0 >= 1.0:
            print(f"Anlik FPS: {fps_n / (now - fps_t0):.2f}")
            fps_n = 0
            fps_t0 = now

        frame_counter += 1

except KeyboardInterrupt:
    print("\n[BILGI] CTRL+C algilandi, sistem guvenli sekilde kapatiliyor...")
    running = False

finally:
    # Sistemi guvenli sekilde durdurma (Cleanup)
    running = False
    reader_t.join(timeout=1)
    
    try:
        picam2.stop()
    except:
        pass
        
    elapsed = time.time() - t0
    if elapsed > 0:
        print(f"Sure {elapsed:.1f} s | Ort. FPS {total_frames / elapsed:.2f}")

    print("Kapatildi.")