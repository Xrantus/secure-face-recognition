"""
Canli yuz tespiti + tanima — Windows Icin Uyarlanmis Kod.
Arayuz (GUI) aciktir. Standart OpenCV VideoCapture kullanilir.
"""

import os
import threading
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from ultralytics import YOLO

# ========= Model =========
MODEL_PATH = "face_yolo11_widerface_best_int8_OPTIMIZED.onnx"

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
DET_SIZE = (160, 160)
FRAME_SKIP = 2


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


if not os.path.isfile(MODEL_PATH):
    raise SystemExit(f"Model bulunamadi: {MODEL_PATH}")

# YOLO modelini yukle
yolo = YOLO(MODEL_PATH, task="detect")

print(
    f"YOLO (Windows Ayarlari): {MODEL_PATH} | imgsz={YOLO_IMG_SIZE} | "
    f"predict conf>={YOLO_PRED_CONF} | son esik>={YOLO_DET_THRESHOLD}"
)

# Windows icin provider secimi
# NVIDIA GPU varsa CUDAExecutionProvider deneyebilirsin.
# Yoksa sadece CPUExecutionProvider kullan.
INSIGHTFACE_PROVIDERS = ['CPUExecutionProvider']

# Ornek:
# INSIGHTFACE_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

app = FaceAnalysis(
    name="buffalo_s",
    root=".",
    allowed_modules=["detection", "recognition"],
    providers=INSIGHTFACE_PROVIDERS
)
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


def frame_reader_thread(cap):
    global latest_frame, running
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Kameradan frame okunamadi.")
            running = False
            break
        with frame_lock:
            latest_frame = frame


print("Kamera baslatiliyor...")

# Windows'ta genelde CAP_DSHOW daha stabil olur
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Alternatif:
# cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise SystemExit("Windows kamerasi baslatilamadi.")

reader_t = threading.Thread(target=frame_reader_thread, args=(cap,), daemon=True)
reader_t.start()

while latest_frame is None and running:
    time.sleep(0.05)

print("Sistem Aktif! Penceriyi kapatmak icin 'q' tusuna basin.\n")

frame_counter = 0
last_results = None
fps_t0 = time.time()
fps_n = 0
total_frames = 0
t0 = time.time()

while running:
    with frame_lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()

    # Ayna goruntusu icin
    frame = cv2.flip(frame, 1)

    fps_n += 1
    total_frames += 1

    if frame_counter % FRAME_SKIP == 0:
        last_results = [yolo.predict(frame, **_predict_kw())[0]]

    if last_results:
        for r in last_results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
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

                MIN_RECOG_FACE_SIZE = 45
                if min(face_w, face_h) < MIN_RECOG_FACE_SIZE:
                    continue

                pw = int(face_w * LANDMARK_PAD)
                ph = int(face_h * LANDMARK_PAD)

                roi = frame[
                    max(0, y1 - ph): min(H, y2 + ph),
                    max(0, x1 - pw): min(W, x2 + pw),
                ]

                if roi.size == 0:
                    continue

                _, kpss = app.det_model.detect(roi, max_num=1, metric="default")
                lm_list = _landmarks_list(kpss)

                kutu_rengi = (60, 200, 255)
                etiket = f"Face {conf:.2f}"

                if lm_list:
                    kps = lm_list[0]
                    aligned_face = face_align.norm_crop(roi, landmark=kps)
                    emb = app.models["recognition"].get_feat(aligned_face)[0]
                    emb = emb / np.linalg.norm(emb)

                    name, score = predict_identity(emb, DB_EMBS, DB_NAMES)

                    if name != "Unknown":
                        kutu_rengi = (0, 255, 0)
                        etiket = f"{name} {score:.2f}"
                    else:
                        kutu_rengi = (0, 0, 255)
                        etiket = f"Unknown {score:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), kutu_rengi, 2)
                cv2.putText(
                    frame,
                    etiket,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    kutu_rengi,
                    2,
                )

    now = time.time()
    if now - fps_t0 >= 1.0:
        print(f"Anlik FPS: {fps_n / (now - fps_t0):.2f}")
        fps_n = 0
        fps_t0 = now

    cv2.imshow("Windows Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

    frame_counter += 1

cap.release()
reader_t.join(timeout=1)

elapsed = time.time() - t0
if elapsed > 0:
    print(f"\nSure {elapsed:.1f} s | Ort. FPS {total_frames / elapsed:.2f}")

cv2.destroyAllWindows()
print("Kapatildi.")