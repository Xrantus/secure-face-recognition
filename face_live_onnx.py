"""
Canli yuz tespiti + tanima — sadece YOLO ONNX (Ultralytics).
TFLite Interpreter kullanilmaz; model yolu her zaman Ultralytics YOLO ile yuklenir.

Oncelik sirasi (ilk bulunan kullanilir):
  face_yolo11_widerface_best_int8.onnx -> face_yolo11_widerface_int8.onnx
  -> face_yolo11_widerface_best.onnx -> face_yolo11_best_int8.onnx
  -> face_yolo11_best.onnx -> (son care) .pt

Veritabani: new_db.py veya db_create.py ile uretilmis known_faces_embeddings.npz

Not: Ultralytics varsayilan predict conf=0.25 oldugu icin dusuk guvenli kutular hic gelmezdi;
YOLO_PRED_CONF (0.01) ile modele gonderilir, asil filtre YOLO_DET_THRESHOLD ile yapilir.

imgsz uyusmazligi: FACE_LIVE_IMGSZ=320 veya 640 ortam degiskeni ile zorlayin.
"""
import os
import threading
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from ultralytics import YOLO

# ================== AYARLAR ==================
FRAME_SKIP = 8
YOLO_DET_THRESHOLD = 0.15
# Ultralytics varsayilan predict conf=0.25; asagidaki YOLO_PRED_CONF ile modele dusuk esik
# gonderilir, filtreleme YOLO_DET_THRESHOLD ile burada yapilir.
YOLO_PRED_CONF = 0.01
YOLO_IOU = 0.70
YOLO_TARGET_CLASS = None
# Tek sinifli yuz modellerinde genelde 0; bazi ONNX ciktilarinda farkli id gelirse None yapin.
YOLO_TARGET_CLASS_ID = None
DEBUG_YOLO = True

DB_PATH = "known_faces_embeddings.npz"
RECOG_THRESHOLD = 0.50
MIN_FACE_SIZE = 35
LANDMARK_PAD = 0.20
DET_SIZE = (320, 320)
CAM_INDEX = 0

# Export script'lerine gore giris cozunurlugu (YOLO imgsz=...)
# WIDERFACE agirliklari cogu zaman 640 ile egitilir/export edilir; new_coco.py 320 da uretebilir.
# Tutarsizlikta: ortam degiskeni FACE_LIVE_IMGSZ=640 veya 320
_IMGSZ_BY_BASENAME = {
    "face_yolo11_widerface_best_int8.onnx": 320,  # new_coco QDQ
    "face_yolo11_widerface_best_int8_OPTIMIZED.onnx": 320,
    "face_yolo11_widerface_best.onnx": 640,  # FP32 export (bu repo)
    "face_yolo11_widerface_int8.onnx": 640,  # int8_export.py
    "face_yolo11_best_int8.onnx": 320,
    "face_yolo11_best.onnx": 320,
    "newint8.onnx": 640,
}


def _yolo_imgsz_for_path(model_path: str) -> int:
    env = os.environ.get("FACE_LIVE_IMGSZ")
    if env and env.isdigit():
        return int(env)
    base = os.path.basename(model_path).lower()
    return _IMGSZ_BY_BASENAME.get(base, 640 if "widerface" in base else 320)


def _yolo_predict_kwargs():
    return {
        "imgsz": YOLO_IMG_SIZE,
        "verbose": False,
        "conf": YOLO_PRED_CONF,
        "iou": YOLO_IOU,
    }


# INT8 basarisiz olursa denenecek FP32 ONNX esleri (aynı aile)
_ONNX_FALLBACK = {
    "face_yolo11_widerface_best_int8.onnx": "face_yolo11_widerface_best.onnx",
    "face_yolo11_widerface_best_int8_OPTIMIZED.onnx": "face_yolo11_widerface_best.onnx",
    "face_yolo11_best_int8.onnx": "face_yolo11_best.onnx",
}
# ============================================


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


def _pick_onnx_model():
    """Oncelikli ONNX yollarini dene; yoksa .pt ile devam (gelistirme kolayligi)."""
    candidates = [
        "face_yolo11_widerface_best_int8.onnx",
        "face_yolo11_widerface_best_int8_OPTIMIZED.onnx",
        "face_yolo11_widerface_int8.onnx",
        "face_yolo11_widerface_best.onnx",
        "face_yolo11_best_int8.onnx",
        "face_yolo11_best.onnx",
    ]
    pt_fallback = [
        "face_yolo11_widerface_best.pt",
        "face_yolo11_best.pt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p, False
    for p in pt_fallback:
        if os.path.exists(p):
            print(f"UYARI: ONNX bulunamadi, PyTorch agirlik kullaniliyor: {p}")
            return p, True
    raise SystemExit(
        "Hic uygun model bulunamadi. En az bir ONNX veya .pt dosyasi ekleyin "
        "(or. face_yolo11_widerface_best.onnx)."
    )


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def cosine_similarity(a, b):
    return np.dot(b, a)


def predict_identity(emb, db_embs, db_names):
    sims = cosine_similarity(emb, db_embs)
    idx = int(np.argmax(sims))
    best_sim = float(sims[idx])
    best_name = str(db_names[idx])
    if best_sim >= RECOG_THRESHOLD:
        return best_name, best_sim
    return "Unknown", best_sim


# --- Modeller ---
selected_path, is_pt = _pick_onnx_model()
YOLO_IMG_SIZE = _yolo_imgsz_for_path(selected_path)
print(
    f"YOLO yukleniyor: {selected_path} (imgsz={YOLO_IMG_SIZE}) [PyTorch={is_pt}] | "
    f"predict conf>={YOLO_PRED_CONF}, sonra esik={YOLO_DET_THRESHOLD}"
)
yolo = YOLO(selected_path, task="detect")
# args.imgsz kullanma: ONNX sabit girisli ise egitim imgsz (or. 640) ile uyusmaz, ORT hata verir.

print("InsightFace 'buffalo_s' yukleniyor...")
app = FaceAnalysis(name="buffalo_s", root=".", allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1, det_size=DET_SIZE)

try:
    db = np.load(DB_PATH, allow_pickle=True)
    DB_EMBS = db["encodings"]
    DB_NAMES = db["names"]
    DB_EMBS = DB_EMBS / np.linalg.norm(DB_EMBS, axis=1, keepdims=True)
    print(f"Veritabani: {DB_PATH} | Kisi sayisi: {len(DB_NAMES)}")
except Exception as e:
    raise SystemExit(f"Embedding veritabani yuklenemedi: {e}. Once new_db.py calistirin.") from e

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
    cap.release()
    print("Kamera okuma thread'i sonlandi.")


cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit("Webcam baslatilamadi.")

reader_t = threading.Thread(target=frame_reader_thread, args=(cap,), daemon=True)
reader_t.start()
while latest_frame is None and running:
    time.sleep(0.05)
print("Canli tanima (ONNX YOLO) aktif. Cikis: q")

frame_counter = 0
last_results = None
fps_start_time = time.time()
fps_frame_count = 0
total_frames_rendered = 0
program_start_time = time.time()

while running:
    with frame_lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()

    frame = cv2.flip(frame, 1)
    fps_frame_count += 1
    total_frames_rendered += 1

    if frame_counter % FRAME_SKIP == 0:
        try:
            yolo_out = yolo(frame, stream=True, **_yolo_predict_kwargs())
            last_results = list(yolo_out)
        except Exception as e:
            base = os.path.basename(selected_path)
            fb = _ONNX_FALLBACK.get(base)
            if fb and os.path.exists(fb) and not is_pt:
                print(f"YOLO INT8/ONNX hatasi: {e}")
                print(f"FP32 ONNX'e geciliyor: {fb}")
                selected_path = fb
                YOLO_IMG_SIZE = _yolo_imgsz_for_path(selected_path)
                yolo = YOLO(selected_path, task="detect")
                yolo_out = yolo(frame, stream=True, **_yolo_predict_kwargs())
                last_results = list(yolo_out)
            else:
                raise

    if last_results:
        for r in last_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                try:
                    class_name = yolo.names[cls_id]
                except Exception:
                    class_name = str(cls_id)
                conf = float(box.conf[0])

                if YOLO_TARGET_CLASS is not None and class_name != YOLO_TARGET_CLASS:
                    continue
                if YOLO_TARGET_CLASS_ID is not None and cls_id != YOLO_TARGET_CLASS_ID:
                    continue
                if conf < YOLO_DET_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
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
                x1e = max(0, x1 - pw)
                y1e = max(0, y1 - ph)
                x2e = min(W, x2 + pw)
                y2e = min(H, y2 + ph)
                roi = frame[y1e:y2e, x1e:x2e]
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

            if DEBUG_YOLO:
                try:
                    print(f"YOLO names: {yolo.names}")
                except Exception:
                    pass
                DEBUG_YOLO = False

    now = time.time()
    if now - fps_start_time >= 1.0:
        fps = fps_frame_count / (now - fps_start_time)
        print(f"Anlik FPS: {fps:.2f}")
        fps_frame_count = 0
        fps_start_time = now

    cv2.imshow("Face live (YOLO ONNX)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break

    frame_counter += 1

running = False
reader_t.join(timeout=1)
elapsed = time.time() - program_start_time
if elapsed > 0:
    print("\n------------------ SISTEM RAPORU ------------------")
    print(f"Sure: {elapsed:.2f} s | Frame: {total_frames_rendered} | Ort. FPS: {total_frames_rendered / elapsed:.2f}")
    print("---------------------------------------------------")

cv2.destroyAllWindows()
print("Kapatildi.")
