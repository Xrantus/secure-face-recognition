import cv2
import time
import threading
import os
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ================== AYARLAR ==================
FRAME_SKIP = 6                          # Inference islemini kac karede bir yapacagiz
YOLO_IMG_SIZE = 320                     # 640 yerine 320 yaparak hizi 2-3 kat artiriyoruz (Edge device optimizasyonu)
YOLO_PERSON_THRESHOLD = 0.5

DB_PATH = "known_faces_embeddings.npz"  # create_db.py'nin cikti dosyasi
RECOG_THRESHOLD = 0.55                  # Cosine similarity esik degeri
MIN_FACE_SIZE = 60                      # ROI icindeki min yuz boyutu (px) - Pi icin biraz dusuruldu
DET_SIZE = (320, 320)                   # InsightFace dedektor girdi boyutu
CAM_INDEX = 0
# ============================================

# 1) Modelleri Yukle

# INT8 ONNX model varsa onu, yoksa ONNX, o da yoksa .pt modeli yukle.
# ONNX olusturmak icin: python coco.py
selected_yolo_model = None
if os.path.exists("yolo11n_int8.onnx"):
    print("YOLO INT8 ONNX modeli yukleniyor...")
    selected_yolo_model = "yolo11n_int8.onnx"
    yolo = YOLO(selected_yolo_model, task="detect")
elif os.path.exists("yolo11n.onnx"):
    print("YOLO ONNX modeli yukleniyor...")
    selected_yolo_model = "yolo11n.onnx"
    yolo = YOLO(selected_yolo_model, task="detect")
else:
    print("yolo11n.onnx bulunamadi, yolo11n.pt yukleniyor...")
    selected_yolo_model = "yolo11n.pt"
    yolo = YOLO(selected_yolo_model)

# InsightFace - Raspberry Pi (Edge Device) icin agir 'buffalo_l' yerine 
# MobileFaceNet mimarisini kullanan 'buffalo_s' (Small) modeline gecildi.
# SADECE detection (tespit) ve recognition (tanima) modullerini ac
print("InsightFace 'buffalo_s' (MobileFaceNet) yukleniyor...")
app = FaceAnalysis(name="buffalo_s", root=".", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=DET_SIZE)

# 2) Veritabanini Yukle
try:
    db = np.load(DB_PATH, allow_pickle=True)
    DB_EMBS = db["encodings"]          # shape: (N, 512), normalize edilmis olmali
    DB_NAMES = db["names"]             # shape: (N,)
    # Guvenlik icin tekrar normalize et
    DB_EMBS = DB_EMBS / np.linalg.norm(DB_EMBS, axis=1, keepdims=True)
    print(f"Veritabani yuklendi: {DB_PATH} | Kayitli kisi sayisi: {len(DB_NAMES)}")
except Exception as e:
    raise SystemExit(f"Embedding veritabani yuklenemedi: {e}. Lutfen create_db.py'yi calistirdigindan emin ol.")

# 3) Yardimci Fonksiyonlar
def cosine_similarity(a, b):
    # a: (512,), b: (M,512)
    return np.dot(b, a)

def predict_identity(emb):
    # emb: (512,) normlu vector
    sims = cosine_similarity(emb, DB_EMBS)  # (N,)
    idx = int(np.argmax(sims))
    best_sim = float(sims[idx])
    best_name = str(DB_NAMES[idx])
    
    if best_sim >= RECOG_THRESHOLD:
        return best_name, best_sim
    return "Unknown", best_sim

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# 4) Kamera Okuma Thread'i (I/O Darbogazini Onlemek Icin)
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
    print("Kamera okuma thread'i sonlandirildi.")

# 5) Kamerayi Baslat
cap = cv2.VideoCapture(CAM_INDEX)
# Cozunurluk Raspberry Pi islemcisine cok yuk bindirmemesi icin VGA seviyesinde tutuluyor
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit("Webcam baslatilamadi. Lutfen baglantiyi kontrol et.")

reader_t = threading.Thread(target=frame_reader_thread, args=(cap,), daemon=True)
reader_t.start()

print("Kamera thread'i basladi. Ilk frame bekleniyor...")
while latest_frame is None and running:
    time.sleep(0.05)
print("Canli tanima sistemi aktif.")

# 6) Dongu Metrikleri
frame_counter = 0
last_results = None
fps_start_time = time.time()
fps_frame_count = 0
total_frames_rendered = 0
program_start_time = time.time()

# 7) Ana Dongu (Pipeline)
while running:
    with frame_lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()

    frame = cv2.flip(frame, 1)

    fps_frame_count += 1
    total_frames_rendered += 1

    # YOLO inference islemini seyrek calistir (FPS artisi icin)
    if frame_counter % FRAME_SKIP == 0:
        try:
            yolo_out = yolo(frame, stream=True, imgsz=YOLO_IMG_SIZE, verbose=False)
            last_results = list(yolo_out)
        except Exception as e:
            # INT8 ONNX bazi ortamlarda ORT operator destegi nedeniyle acilamayabilir.
            if selected_yolo_model == "yolo11n_int8.onnx" and os.path.exists("yolo11n.onnx"):
                print(f"INT8 model hatasi alindi: {e}")
                print("FP32 ONNX modele geciliyor (yolo11n.onnx)...")
                selected_yolo_model = "yolo11n.onnx"
                yolo = YOLO(selected_yolo_model, task="detect")
                yolo_out = yolo(frame, stream=True, imgsz=YOLO_IMG_SIZE, verbose=False)
                last_results = list(yolo_out)
            else:
                raise

    if last_results:
        for r in last_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = yolo.names[cls_id]
                conf = float(box.conf[0])

                # Mimarisi Guncelleme Notu: Eger YOLO'yu dogrudan yuz (face) tespiti 
                # icin egitirsen, buradaki "person" kosulunu "face" olarak degistirebilirsin.
                if class_name != "person":
                    continue
                if conf < YOLO_PERSON_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                H, W = frame.shape[:2]
                x1 = clamp(x1, 0, W - 1)
                x2 = clamp(x2, 0, W - 1)
                y1 = clamp(y1, 0, H - 1)
                y2 = clamp(y2, 0, H - 1)
                
                if x2 <= x1 or y2 <= y1:
                    continue

                # Kisinin (Person) bulundugu bolgeyi kes (Crop)
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Kesilen ROI icinde yuz tespiti ve Feature Extraction (Embedding)
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

                        # MobileFaceNet'in cikardigi vektor (Feature Vektor)
                        emb = f.normed_embedding  
                        name, score = predict_identity(emb)
                        
                        if score > best_score:
                            best_score = score
                            best_label = name

                        # Kucuk ROI koordinatlarini ana (global) frame koordinatlarina cevir
                        gx1 = x1 + fx1
                        gy1 = y1 + fy1
                        gx2 = x1 + fx2
                        gy2 = y1 + fy2
                        
                        # Yuz kutusu ve Kimlik Bilgisi (Yesil renk = Yuz)
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

                # Kisi (Person) kutusu ve Etiketi (Sari renk = Insan)
                label = f"person {conf:.2f}"
                if best_label is not None:
                    label = f"Match: {best_label}"
                    
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

    # Anlik FPS Hesaplama ve Ekrana Yazdirma
    now = time.time()
    if now - fps_start_time >= 1.0:
        fps = fps_frame_count / (now - fps_start_time)
        print(f"Anlik FPS: {fps:.2f}")
        fps_frame_count = 0
        fps_start_time = now

    cv2.imshow("Edge-Optimized Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break

    frame_counter += 1

# 8) Kaynaklarin Temizlenmesi ve Raporlama
print("Sistemden cikis yapiliyor, kaynaklar temizleniyor...")
running = False
reader_t.join(timeout=1)

program_end_time = time.time()
elapsed = program_end_time - program_start_time
if elapsed > 0:
    avg_fps = total_frames_rendered / elapsed
    print("\n------------------ SISTEM RAPORU ------------------")
    print(f"Toplam Calisma Suresi : {elapsed:.2f} saniye")
    print(f"Islenen Toplam Kare   : {total_frames_rendered} frame")
    print(f"Ortalama FPS Degeri   : {avg_fps:.2f} FPS")
    print("---------------------------------------------------")

cv2.destroyAllWindows()
print("Kapatildi.")