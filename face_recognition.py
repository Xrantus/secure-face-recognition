import cv2
import time
import threading
import os
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ================== AYARLAR ==================
FRAME_SKIP = 6                  # Her N frame'de bir detection yap
# YOLO_IMG_SIZE modelden otomatik okunacak; asagida TFLite yuklendikten sonra set edilir
YOLO_IMG_SIZE = None
YOLO_DET_THRESHOLD = 0.25       # YOLO confidence threshold
NMS_THRESHOLD = 0.45            # Non-Maximum Suppression threshold

DB_PATH = "known_faces_embeddings.npz"
RECOG_THRESHOLD = 0.55          # Cosine similarity threshold (pipeline duzeltildi, 0.55 daha saglikli)
MIN_FACE_SIZE = 35              # Minimum yuz boyutu (piksel)
DET_SIZE = (320, 320)           # InsightFace landmark detector input boyutu
LANDMARK_PAD = 0.20             # YOLO bbox'i landmark detection icin genisletme orani
CAM_INDEX = 0
TFLITE_MODEL_PATH = "face_yolo11_widerface_best_int8.tflite"
# ============================================


# =========== 1) TFLite YOLO Yukleme ===========
print(f"TFLite model yukleniyor: {TFLITE_MODEL_PATH}")
try:
    # RPi icin tflite_runtime, masaustu icin tensorflow.lite
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
        print("tflite_runtime kullaniliyor (RPi modu).")
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        print("tensorflow.lite kullaniliyor (masaustu modu).")

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]  # [1, H, W, 3]
    YOLO_IMG_SIZE = int(input_shape[1])
    print(f"TFLite model yuklendi. Input shape: {input_shape} → {YOLO_IMG_SIZE}x{YOLO_IMG_SIZE}")
except Exception as e:
    raise SystemExit(f"TFLite model yuklenemedi: {e}\nModel yolu dogru mu? -> {TFLITE_MODEL_PATH}")


# =========== 2) InsightFace Yukleme ===========
print("InsightFace 'buffalo_s' yukleniyor...")
app = FaceAnalysis(name="buffalo_s", root=".", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=DET_SIZE)
print("InsightFace hazir.")


# =========== 3) Veritabani Yukleme ===========
try:
    db = np.load(DB_PATH, allow_pickle=True)
    DB_EMBS = db["encodings"].astype(np.float32)
    DB_NAMES = db["names"]
    # L2 normalize (zaten normalize edilmis olmali ama garantiye al)
    norms = np.linalg.norm(DB_EMBS, axis=1, keepdims=True)
    DB_EMBS = DB_EMBS / np.where(norms == 0, 1, norms)
    print(f"Veritabani yuklendi: {len(DB_NAMES)} kisi kayitli.")
    for n in DB_NAMES:
        print(f"  - {n}")
except Exception as e:
    raise SystemExit(
        f"Embedding veritabani yuklenemedi: {e}\n"
        f"Lutfen once new_db.py'yi calistirin."
    )


# =========== 4) Yardimci Fonksiyonlar ===========
def predict_identity(emb):
    """Normalize edilmis embedding'i DB ile karsilastirir."""
    sims = np.dot(DB_EMBS, emb)          # Cosine similarity (vektorler normalize)
    idx = int(np.argmax(sims))
    best_sim = float(sims[idx])
    best_name = str(DB_NAMES[idx])

    if best_sim >= RECOG_THRESHOLD:
        return best_name, best_sim
    return "Unknown", best_sim


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def tflite_detect(frame):
    """
    TFLite YOLO ile yuz tespiti yapar.
    Doner: list of [x1, y1, x2, y2, conf]  (frame koordinatlari)
    """
    H, W = frame.shape[:2]

    # Preprocessing — YOLO standardi
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (YOLO_IMG_SIZE, YOLO_IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # [1, YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3]

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Output: Ultralytics TFLite export → [1, 5, num_det] veya [1, num_det, 5]
    raw = interpreter.get_tensor(output_details[0]['index'])

    # Shape normalizasyonu: [1, 5, N] → [N, 5]
    if raw.ndim == 3:
        if raw.shape[1] == 5:      # [1, 5, N] formatı (Ultralytics default)
            raw = raw[0].T         # → [N, 5]
        elif raw.shape[2] == 5:    # [1, N, 5] formatı
            raw = raw[0]           # → [N, 5]
        else:
            return []
    else:
        return []

    detections = []
    for det in raw:
        cx, cy, w, h, conf = det
        if conf < YOLO_DET_THRESHOLD:
            continue

        # YOLO normalized center format → pixel xyxy
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)

        x1, x2 = clamp(x1, 0, W - 1), clamp(x2, 0, W - 1)
        y1, y2 = clamp(y1, 0, H - 1), clamp(y2, 0, H - 1)

        if x2 > x1 and y2 > y1:
            detections.append([x1, y1, x2, y2, conf])

    if not detections:
        return []

    # NMS uygula
    boxes = [[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in detections]
    scores = [d[4] for d in detections]
    indices = cv2.dnn.NMSBoxes(boxes, scores, YOLO_DET_THRESHOLD, NMS_THRESHOLD)

    if len(indices) == 0:
        return []

    return [detections[i] for i in indices.flatten()]


def get_embedding_from_bbox(frame, x1, y1, x2, y2):
    """
    YOLO bbox'tan duzeltilmis pipeline ile embedding cikarir.
    1) Padding ekle
    2) det_model ile landmark al
    3) norm_crop ile hizala
    4) recognition modeli ile embedding uret
    Doner: (embedding, landmark_bulundu_mu)
    """
    H, W = frame.shape[:2]

    # Landmark detection icin bbox'i biraz genislet
    pw = int((x2 - x1) * LANDMARK_PAD)
    ph = int((y2 - y1) * LANDMARK_PAD)
    x1e = clamp(x1 - pw, 0, W - 1)
    y1e = clamp(y1 - ph, 0, H - 1)
    x2e = clamp(x2 + pw, 0, W - 1)
    y2e = clamp(y2 + ph, 0, H - 1)

    roi = frame[y1e:y2e, x1e:x2e]
    if roi.size == 0:
        return None, False

    # Minimum boyut kontrolu
    rh, rw = roi.shape[:2]
    if min(rw, rh) < MIN_FACE_SIZE:
        return None, False

    try:
        # Sadece landmark tespiti (tam detection degil — YOLO zaten yapti)
        bboxes, kpss = app.det_model.detect(roi, max_num=1, metric='default')
    except Exception:
        return None, False

    if kpss is None or len(kpss) == 0:
        return None, False

    kps = kpss[0]  # (5, 2) — 5 landmark noktasi

    # Affine alignment → 112x112 standart ArcFace yuz formati
    aligned_face = face_align.norm_crop(roi, landmark=kps)

    # Sadece recognition modelini calistir (detection YOK)
    emb = app.models['recognition'].get_feat(aligned_face)[0]
    emb = emb / np.linalg.norm(emb)

    return emb, True


# =========== 5) Kamera Thread'i ===========
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
    print("Kamera thread'i sonlandi.")


cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit("Kamera baslatilamadi.")

reader_t = threading.Thread(target=frame_reader_thread, args=(cap,), daemon=True)
reader_t.start()

print("Kamera baslatildi, ilk frame bekleniyor...")
while latest_frame is None and running:
    time.sleep(0.05)
print("Sistem aktif. Cikmak icin 'q' tusuna basin.\n")


# =========== 6) Ana Dongu ===========
frame_counter = 0
last_detections = []   # [x1, y1, x2, y2, conf, name, score, kutu_rengi]
fps_start = time.time()
fps_count = 0
total_frames = 0
program_start = time.time()

while running:
    with frame_lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()

    frame = cv2.flip(frame, 1)
    fps_count += 1
    total_frames += 1

    # Her FRAME_SKIP karede bir tam pipeline calistir
    if frame_counter % FRAME_SKIP == 0:
        raw_dets = tflite_detect(frame)
        last_detections = []

        for det in raw_dets:
            x1, y1, x2, y2, conf = det

            emb, found = get_embedding_from_bbox(frame, x1, y1, x2, y2)

            if not found or emb is None:
                # YOLO buldu ama landmark alinamadi → sari kutu
                last_detections.append((x1, y1, x2, y2, conf, "Face", conf, (60, 200, 255)))
                continue

            name, score = predict_identity(emb)

            if name != "Unknown":
                kutu_rengi = (0, 220, 0)       # Yesil — eslesme basarili
            else:
                kutu_rengi = (0, 0, 220)       # Kirmizi — yabanci

            last_detections.append((x1, y1, x2, y2, conf, name, score, kutu_rengi))

    # Onceki detection sonuclarini her frame'e cizdirme
    for (x1, y1, x2, y2, conf, name, score, renk) in last_detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)

        if name == "Face":
            etiket = f"Face {conf:.2f}"
        else:
            etiket = f"{name} {score:.2f}"

        # Etiket arka plani (okunabilirlik icin)
        (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(0, y1 - 10)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw, ty + 2), renk, -1)
        cv2.putText(frame, etiket, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # FPS gostergesi
    now = time.time()
    if now - fps_start >= 1.0:
        fps = fps_count / (now - fps_start)
        fps_count = 0
        fps_start = now
        print(f"FPS: {fps:.1f} | Aktif tespit: {len(last_detections)}")

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

    frame_counter += 1


# =========== 7) Temizlik ===========
print("\nSistemden cikiliyor...")
running = False
reader_t.join(timeout=2)

elapsed = time.time() - program_start
if elapsed > 0:
    print("\n========== SISTEM RAPORU ==========")
    print(f"Calisma Suresi  : {elapsed:.1f} saniye")
    print(f"Toplam Frame    : {total_frames}")
    print(f"Ortalama FPS    : {total_frames / elapsed:.1f}")
    print("===================================")

cv2.destroyAllWindows()
print("Kapatildi.")
