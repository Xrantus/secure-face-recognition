"""
Canli yuz tespiti + tanima — tek ONNX modeli (Ultralytics YOLO) + InsightFace.

ONNX Runtime QDQ ile uretilen *_int8.onnx modelleri Ultralytics ONNX backend'i ile
siklikla uyumsuzdur (ham kutu 0, conf ~0). Bu script FP32 ONNX kullanir.

YOLO_IMG_SIZE, export ile sabit giris boyuna esit olmali (ORT hatasi: Expected N Got M).
Bu repodaki face_yolo11_widerface_best.onnx 640x640; INT8 kopya 320x320 olabilir.

Veritabani: known_faces_embeddings.npz (new_db.py / db_create.py).
"""
import os
import threading
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from ultralytics import YOLO

# ========= Tek model (FP32 ONNX — INT8 degil) =========
MODEL_PATH = "face_yolo11_widerface_best_int8_OPTIMIZED.onnx"
# FP32 widerface ONNX bu projede 640 sabit; 320 verilirse ORT "Expected 640" hatasi verir.
YOLO_IMG_SIZE = 640

# ========= Genel =========
FRAME_SKIP = 8
# FP32'de dahi dusuk conf (~0.1) gorulebilir; 0.15 cogu kutuyu eleyebilir.
YOLO_DET_THRESHOLD = 0.08
YOLO_PRED_CONF = 0.01
YOLO_IOU = 0.45
MAX_DET = 100
YOLO_TARGET_CLASS = None
YOLO_TARGET_CLASS_ID = None

DB_PATH = "known_faces_embeddings.npz"
RECOG_THRESHOLD = 0.50
MIN_FACE_SIZE = 35
LANDMARK_PAD = 0.20
DET_SIZE = (320, 320)
CAM_INDEX = 0


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
    if os.path.isfile(_int8_only):
        raise SystemExit(
            f"'{MODEL_PATH}' yok; klasorde sadece '{_int8_only}' var.\n"
            "ORT QDQ INT8 ONNX, Ultralytics ile cogu ortamda 0 tespit verir.\n"
            "Cozum: new_coco.py ile FP32 export (adim 2) veya egittigin .pt ile\n"
            "  yolo = YOLO('face_yolo11_widerface_best.pt'); yolo.export(format='onnx', imgsz=320)\n"
            "ciktisini 'face_yolo11_widerface_best.onnx' olarak bu klasore koy."
        )
    raise SystemExit(f"Model bulunamadi: {MODEL_PATH}")

yolo = YOLO(MODEL_PATH, task="detect")

print(
    f"YOLO: {MODEL_PATH} | imgsz={YOLO_IMG_SIZE} | "
    f"predict conf>={YOLO_PRED_CONF} | son esik>={YOLO_DET_THRESHOLD}"
)

app = FaceAnalysis(name="buffalo_s", root=".", allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1, det_size=DET_SIZE)

try:
    db = np.load(DB_PATH, allow_pickle=True)
    DB_EMBS = db["encodings"]
    DB_NAMES = db["names"]
    DB_EMBS = DB_EMBS / np.linalg.norm(DB_EMBS, axis=1, keepdims=True)
    print(f"Veritabani: {DB_PATH} | {len(DB_NAMES)} kisi")
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
    cap.release()


cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit("Webcam baslatilamadi.")

reader_t = threading.Thread(target=frame_reader_thread, args=(cap,), daemon=True)
reader_t.start()
while latest_frame is None and running:
    time.sleep(0.05)
print("Aktif. Cikis: q")

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

    frame = cv2.flip(frame, 1)
    fps_n += 1
    total_frames += 1

    if frame_counter % FRAME_SKIP == 0:
        # stream=False: tek Results; stream=True ile bazı ortamlarda bos iterator raporlari var.
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
                print(
                    f"[tespit] ham kutu: {n} | en yuksek conf: {best:.3f} "
                    f"(cizim icin >={YOLO_DET_THRESHOLD})"
                )
                logged_raw_boxes = True

            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                try:
                    class_name = yolo.names[cls_id]
                except Exception:
                    class_name = str(cls_id)
                conf = float(boxes.conf[i])

                if YOLO_TARGET_CLASS is not None and class_name != YOLO_TARGET_CLASS:
                    continue
                if YOLO_TARGET_CLASS_ID is not None and cls_id != YOLO_TARGET_CLASS_ID:
                    continue
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

            if debug_names:
                try:
                    print(f"YOLO names: {yolo.names}")
                except Exception:
                    pass
                debug_names = False

    now = time.time()
    if now - fps_t0 >= 1.0:
        print(f"Anlik FPS: {fps_n / (now - fps_t0):.2f}")
        fps_n = 0
        fps_t0 = now

    cv2.imshow("Face live (tek ONNX)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break

    frame_counter += 1

running = False
reader_t.join(timeout=1)
elapsed = time.time() - t0
if elapsed > 0:
    print(f"Sure {elapsed:.1f} s | Ort. FPS {total_frames / elapsed:.2f}")

cv2.destroyAllWindows()
print("Kapatildi.")
