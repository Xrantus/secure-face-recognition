import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ================== AYARLAR ==================
FRAME_SKIP = 6
YOLO_IMG_SIZE = 160
YOLO_PERSON_THRESHOLD = 0.5

DB_PATH = "known_faces_embeddings.npz"  # create_db.py nin cikti dosyasi
RECOG_THRESHOLD = 0.55                  # cosine esik 
MIN_FACE_SIZE = 80                      # ROI icindeki min yuz boyutu (px)
DET_SIZE = (320, 320)                   # InsightFace detektor girdi boyutu
CAM_INDEX = 0
# ============================================

# 1) Modelleri yukle
yolo = YOLO("yolo11n.pt")

# InsightFace (GPU varsa 0, yoksa CPU -1)
try:
    app = FaceAnalysis(name="buffalo_l", root=".")
    app.prepare(ctx_id=0, det_size=DET_SIZE)
except Exception:
    app = FaceAnalysis(name="buffalo_l", root=".")
    app.prepare(ctx_id=-1, det_size=DET_SIZE)

# 2) Veritabani yukle
try:
    db = np.load(DB_PATH, allow_pickle=True)
    DB_EMBS = db["encodings"]          # sekil: (N, 512), normlanmis olmali
    DB_NAMES = db["names"]             # sekil: (N,)
    # Guvenlik icin tekrar normla
    DB_EMBS = DB_EMBS / np.linalg.norm(DB_EMBS, axis=1, keepdims=True)
    print(f"Loaded DB: {DB_PATH} | persons: {len(DB_NAMES)}")
except Exception as e:
    raise SystemExit(f"Embedding veritabani yuklenemedi: {e}")

# 3) Yardimci fonksiyonlar
def cosine_similarity(a, b):
    # a: (512,), b: (M,512)
    return np.dot(b, a)

def predict_identity(emb):
    # emb: (512,) normlu
    sims = cosine_similarity(emb, DB_EMBS)  # (N,)
    idx = int(np.argmax(sims))
    best_sim = float(sims[idx])
    best_name = str(DB_NAMES[idx])
    if best_sim >= RECOG_THRESHOLD:
        return best_name, best_sim
    return "Unknown", best_sim

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# 4) Kamera okuma thread
latest_frame = None
frame_lock = threading.Lock()
running = True

def frame_reader_thread(cap):
    global latest_frame, running
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Kamera kare okunamadi.")
            running = False
            break
        with frame_lock:
            latest_frame = frame
    cap.release()
    print("Kamera okuma thread bitti.")

# 5) Kamera baslat
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit("Webcam baslatilamadi.")

reader_t = threading.Thread(target=frame_reader_thread, args=(cap,), daemon=True)
reader_t.start()

print("Kamera thread basladi. Ilk kare bekleniyor...")
while latest_frame is None and running:
    time.sleep(0.05)
print("Canli tanima basladi.")

# 6) Dongu metrikleri
frame_counter = 0
last_results = None
fps_start_time = time.time()
fps_frame_count = 0
total_frames_rendered = 0
program_start_time = time.time()

# 7) Ana dongu
while running:
    with frame_lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()

    frame = cv2.flip(frame, 1)

    fps_frame_count += 1
    total_frames_rendered += 1

    # YOLO inference i seyrek calistir
    if frame_counter % FRAME_SKIP == 0:
        yolo_out = yolo(frame, stream=True, imgsz=YOLO_IMG_SIZE, verbose=False)
        last_results = list(yolo_out)

    if last_results:
        for r in last_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = yolo.names[cls_id]
                conf = float(box.conf[0])

                if class_name != "person":
                    continue
                if conf < YOLO_PERSON_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Kirmizi alan kontrolu
                H, W = frame.shape[:2]
                x1 = clamp(x1, 0, W - 1)
                x2 = clamp(x2, 0, W - 1)
                y1 = clamp(y1, 0, H - 1)
                y2 = clamp(y2, 0, H - 1)
                if x2 <= x1 or y2 <= y1:
                    continue

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # ROI icinde yuz tespiti + embedding
                faces = app.get(roi)
                best_label = None
                best_score = -1.0

                if faces:
                    for f in faces:
                        fx1, fy1, fx2, fy2 = map(int, f.bbox)
                        fw = fx2 - fx1
                        fh = fy2 - fy1
                        if min(fw, fh) < MIN_FACE_SIZE:
                            continue

                        emb = f.normed_embedding  # (512,)
                        name, score = predict_identity(emb)
                        if score > best_score:
                            best_score = score
                            best_label = name

                        # Global koordinata cevirip yuz kutusu ciz
                        gx1 = x1 + fx1
                        gy1 = y1 + fy1
                        gx2 = x1 + fx2
                        gy2 = y1 + fy2
                        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{name} {score:.2f}",
                            (gx1, gy1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                # Person kutusu da ciz (etiketi yuz sonuclarina gore)
                label = f"person {conf:.2f}"
                if best_label is not None:
                    label = f"{best_label} {best_score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 200, 255), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (60, 200, 255),
                    2,
                )

    # FPS yazdir
    now = time.time()
    if now - fps_start_time >= 1.0:
        fps = fps_frame_count / (now - fps_start_time)
        print(f"FPS: {fps:.2f}")
        fps_frame_count = 0
        fps_start_time = now

    cv2.imshow("Face Recognition (YOLO + InsightFace)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break

    frame_counter += 1

# 8) Temizlik ve ortalama FPS
print("Cikis yapiliyor...")
running = False
reader_t.join(timeout=1)

program_end_time = time.time()
elapsed = program_end_time - program_start_time
if elapsed > 0:
    avg_fps = total_frames_rendered / elapsed
    print("-------------------------------------------")
    print(f"Toplam sure: {elapsed:.2f} sn")
    print(f"Toplam kare: {total_frames_rendered}")
    print(f"Ortalama FPS: {avg_fps:.2f}")
    print("-------------------------------------------")

cv2.destroyAllWindows()
print("Bitti.")
