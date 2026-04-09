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
YOLO_DET_THRESHOLD = 0.25
# ONNX export'larda class isimleri bazen "face" gelmeyebiliyor. Bu nedenle:
# - once filtreyi kapatip (None) tum tespitleri gor (debug)
# - sonra isme gore ("face") veya id'ye gore (0) filtrele
YOLO_TARGET_CLASS = None                # ornek: "item"
YOLO_TARGET_CLASS_ID = 0                # bu modelde names: {0: 'item'} geldigi icin id=0 filtrele
DEBUG_YOLO = True                       # ilk calistirmada True birak, sonra False yapabilirsin.

DB_PATH = "known_faces_embeddings.npz"  # create_db.py'nin cikti dosyasi
RECOG_THRESHOLD = 0.55                  # Cosine similarity esik degeri
MIN_FACE_SIZE = 60                      # ROI icindeki min yuz boyutu (px) - Pi icin biraz dusuruldu
DET_SIZE = (320, 320)                   # InsightFace dedektor girdi boyutu
CAM_INDEX = 0
# ============================================

# 1) Modelleri Yukle

# INT8 ONNX model varsa onu, yoksa FP32 ONNX, o da yoksa .pt modeli yukle.
# ONNX olusturmak/quantize etmek icin: python new_coco.py
selected_yolo_model = None
if os.path.exists("face_yolo11_best_int8.onnx"):
    print("YOLO INT8 ONNX modeli yukleniyor (face_yolo11_best_int8.onnx)...")
    selected_yolo_model = "face_yolo11_best_int8.onnx"
    yolo = YOLO(selected_yolo_model, task="detect")
elif os.path.exists("face_yolo11_best.onnx"):
    print("YOLO ONNX modeli yukleniyor (face_yolo11_best.onnx)...")
    selected_yolo_model = "face_yolo11_best.onnx"
    yolo = YOLO(selected_yolo_model, task="detect")
else:
    # Eger ONNX dosyalar yoksa, egittigin .pt dosyasini buraya koyabilirsin.
    # (Ultralytics YOLO ile .pt dogrudan calisir.)
    print("ONNX bulunamadi, face_yolo11_best.pt yukleniyor...")
    selected_yolo_model = "face_yolo11_best.pt"
    yolo = YOLO(selected_yolo_model, task="detect")

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
            if selected_yolo_model == "face_yolo11_best_int8.onnx" and os.path.exists("face_yolo11_best.onnx"):
                print(f"INT8 model hatasi alindi: {e}")
                print("FP32 ONNX modele geciliyor (face_yolo11_best.onnx)...")
                selected_yolo_model = "face_yolo11_best.onnx"
                yolo = YOLO(selected_yolo_model, task="detect")
                yolo_out = yolo(frame, stream=True, imgsz=YOLO_IMG_SIZE, verbose=False)
                last_results = list(yolo_out)
            else:
                raise

    if last_results:
        for r in last_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                # ONNX modellerde yolo.names her zaman beklenen stringleri vermeyebilir.
                try:
                    class_name = yolo.names[cls_id]
                except Exception:
                    class_name = str(cls_id)
                conf = float(box.conf[0])

                # Filtre (istege bagli)
                if YOLO_TARGET_CLASS is not None and class_name != YOLO_TARGET_CLASS:
                    continue
                if YOLO_TARGET_CLASS_ID is not None and cls_id != YOLO_TARGET_CLASS_ID:
                    continue
                if conf < YOLO_DET_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                H, W = frame.shape[:2]
                x1 = clamp(x1, 0, W - 1)
                x2 = clamp(x2, 0, W - 1)
                y1 = clamp(y1, 0, H - 1)
                y2 = clamp(y2, 0, H - 1)
                
                if x2 <= x1 or y2 <= y1:
                    continue

                # Tespit edilen bolgeyi kes (Crop)
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Kesilen ROI icinde Feature Extraction (Embedding)
                faces = app.get(roi)
                
                # Varsayilan Kutu (Sari renk ve sadece Face yazisi)
                kutu_rengi = (60, 200, 255) 
                etiket = f"Face {conf:.2f}" 

                if faces:
                    for f in faces:
                        fw = f.bbox[2] - f.bbox[0]
                        fh = f.bbox[3] - f.bbox[1]
                        if min(fw, fh) < MIN_FACE_SIZE:
                            continue

                        # MobileFaceNet'in cikardigi vektor (Feature Vektor)
                        emb = f.normed_embedding  
                        name, score = predict_identity(emb)
                        
                        # Eger tanimlama yapilirsa kutuyu Yesil yap ve Ismi yazdir
                        kutu_rengi = (0, 255, 0)
                        etiket = f"{name} {score:.2f}"
                        break # En net yuzu alip donguden cik

                # ---------------------------------------------------------
                # EKRANA TEK BIR KUTU VE ETIKET CIZDIRME (YOLO Koordinatlari)
                # ---------------------------------------------------------
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