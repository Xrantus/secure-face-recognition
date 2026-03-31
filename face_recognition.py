import cv2
import time
import numpy as np
from ultralytics import YOLO
import threading
import os

# --- AYARLAR ---
FRAME_SKIP = 3
YOLO_IMG_SIZE = 320
PERSON_CONFIDENCE_THRESHOLD = 0.5

# COCO veri setinde 'person' sınıfının index'i 0'dır
PERSON_CLASS_ID = 0

# --- MODEL YÜKLEME ---
model = YOLO('yolo11n.pt')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- YÜZ TANIMA İÇİN ÖZNİTELİK ÇIKARMA FONKSİYONU ---
def extract_face_features(face_image):
    """
    Yüz görüntüsünden öznitelik vektörü çıkarır.
    Basit histogram tabanlı öznitelik çıkarma kullanır.
    """
    if face_image is None or face_image.size == 0:
        return None

    # Gri tonlamaya çevir
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # Histogram eşitleme
    gray = cv2.equalizeHist(gray)

    # Histogram çıkar (8x8 bloklar için)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Ek öznitelikler: ortalama renk değerleri
    mean_color = np.mean(face_image, axis=(0, 1))

    # Vektörü birleştir
    feature_vector = np.concatenate([hist, mean_color])

    return feature_vector

# --- BENZERLİK HESAPLAMA ---
def calculate_similarity(vector1, vector2):
    """
    İki vektör arasındaki kosinüs benzerliğini hesaplar.
    """
    if vector1 is None or vector2 is None:
        return 0.0

    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return similarity

# --- HEDEF VEKTÖR YÜKLEME/KAYDETME ---
def load_or_create_target_vector(target_vector_file='target_face_vector.npy'):
    """
    Önceden kaydedilmiş hedef vektörü yükler veya yeni bir tane oluşturur.
    """
    if os.path.exists(target_vector_file):
        print(f"Hedef vektör dosyası bulundu: {target_vector_file}")
        return np.load(target_vector_file)
    else:
        print(f"Hedef vektör dosyası bulunamadı: {target_vector_file}")
        print("Lütfen önce bir hedef yüz fotoğrafı yükleyin ve 'r' tuşuna basın.")
        return None

# --- KAMERA I/O THREAD ---
latest_frame = None
frame_lock = threading.Lock()
running = True
target_vector = None

def frame_reader_thread(cap):
    global latest_frame, running
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Hata: Kare okunamadı veya kamera bağlantısı kesildi.")
            running = False
            break
        with frame_lock:
            latest_frame = frame
    cap.release()
    print("Kamera okuma thread'i sonlandırıldı.")

# --- WEBCAM BAŞLATMA ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Hata: Webcam başlatılamadı.")
    exit()

reader_thread = threading.Thread(target=frame_reader_thread, args=(cap,))
reader_thread.daemon = True
reader_thread.start()

print("Kamera okuma thread'i başlatıldı. İlk kare bekleniyor...")
while latest_frame is None and running:
    time.sleep(0.1)

# Hedef vektörü yükle
target_vector = load_or_create_target_vector()

print("Yüz Tanıma Sistemi başlatıldı!")

# --- ANA DÖNGÜ DEĞİŞKENLERİ ---
frame_counter = 0
last_results = None
fps_start_time = time.time()
fps_frame_count = 0

# --- Ortalama FPS için eklendi ---
total_frames_rendered = 0
program_start_time = time.time()

# Yüz tanıma sonuçlarını tutmak için
recognition_results = []

while running:
    with frame_lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()

    frame = cv2.flip(frame, 1)

    # FPS hesaplaması için kare sayacını artır
    fps_frame_count += 1
    total_frames_rendered += 1

    # Sadece belirli aralıklarla YOLO çalıştır (sadece person sınıfı için)
    if frame_counter % FRAME_SKIP == 0:
        # YOLO'yu sadece 'person' sınıfı için çalıştır
        results = model(frame, stream=True, imgsz=YOLO_IMG_SIZE, verbose=False,
                       classes=[PERSON_CLASS_ID], conf=PERSON_CONFIDENCE_THRESHOLD)
        last_results = list(results)

    # Sonuçları işle ve yüz tanıma yap
    if last_results:
        for r in last_results:
            boxes = r.boxes
            for box in boxes:
                # Sadece person sınıfı zaten filtrelendi
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]

                # Bounding box'ı çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # Person etiketi
                label = f'Person {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # Yüz tanıma için ROI çıkar
                person_roi = frame[y1:y2, x1:x2]

                if person_roi.size > 0:
                    # Yüz algılama için gri tonlamaya çevir
                    gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

                    # Haar Cascade ile yüzleri tespit et
                    faces = face_cascade.detectMultiScale(
                        gray_roi,
                        scaleFactor=1.3,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    if len(faces) > 0:
                        for (fx, fy, fw, fh) in faces:
                            # Yüz koordinatlarını global kareye göre hesapla
                            face_x1 = x1 + fx
                            face_y1 = y1 + fy
                            face_x2 = x1 + fx + fw
                            face_y2 = y1 + fy + fh

                            # Yüzü kırp
                            face_crop = frame[face_y1:face_y2, face_x1:face_x2]

                            if face_crop.size > 0:
                                # Yüz özniteliklerini çıkar
                                face_features = extract_face_features(face_crop)

                                if face_features is not None and target_vector is not None:
                                    # Benzerlik hesapla
                                    similarity = calculate_similarity(face_features, target_vector)

                                    # Eşik değer kontrolü (ayarlanabilir)
                                    threshold = 0.7
                                    recognized = similarity > threshold

                                    # Sonucu kaydet
                                    recognition_results.append({
                                        'similarity': similarity,
                                        'recognized': recognized,
                                        'timestamp': time.time()
                                    })

                                    # Görsel geri bildirim
                                    color = (0, 255, 0) if recognized else (0, 0, 255)
                                    status_text = f"{'TANINDI' if recognized else 'BILINMIYOR'} ({similarity:.2f})"

                                    cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), color, 2)
                                    cv2.putText(frame, status_text, (face_x1, face_y1 - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                else:
                                    # Hedef vektör yoksa sarı çerçeve
                                    cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 255), 2)
                                    cv2.putText(frame, "HEDEF YOK", (face_x1, face_y1 - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # FPS hesapla ve göster
    if time.time() - fps_start_time >= 1.0:
        fps = fps_frame_count / (time.time() - fps_start_time)
        print(f"Anlık FPS: {fps:.2f}")
        fps_frame_count = 0
        fps_start_time = time.time()

    # Bilgi gösterimi
    info_text = f"FPS: {fps:.1f} | Tespit Edilen Yuz: {len(recognition_results)}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Yuz Tanima Sistemi', frame)

    key = cv2.waitKey(1) & 0xFF

    # Çıkış için 'q' tuşu
    if key == ord('q'):
        running = False
        break

    # Hedef vektör kaydetmek için 'r' tuşu (referans yüz)
    elif key == ord('r') and latest_frame is not None:
        # Mevcut frame'deki yüzleri al ve ilk bulunan yüzü kaydet
        temp_results = model(frame, stream=True, imgsz=YOLO_IMG_SIZE, verbose=False,
                           classes=[PERSON_CLASS_ID], conf=PERSON_CONFIDENCE_THRESHOLD)

        for r in temp_results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_roi = frame[y1:y2, x1:x2]

                if person_roi.size > 0:
                    gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray_roi, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
                    )

                    if len(faces) > 0:
                        (fx, fy, fw, fh) = faces[0]
                        face_x1, face_y1 = x1 + fx, y1 + fy
                        face_x2, face_y2 = x1 + fx + fw, y1 + fy + fh

                        face_crop = frame[face_y1:face_y2, face_x1:face_x2]

                        if face_crop.size > 0:
                            target_vector = extract_face_features(face_crop)
                            if target_vector is not None:
                                np.save('target_face_vector.npy', target_vector)
                                print("Hedef vektör kaydedildi! target_face_vector.npy")
                            break

    frame_counter += 1

# Kaynakları serbest bırak
print("Çıkış yapılıyor...")
running = False
reader_thread.join(timeout=1)

# --- Ortalama FPS'i hesapla ve yazdır ---
program_end_time = time.time()
total_elapsed_time = program_end_time - program_start_time

if total_elapsed_time > 0:
    average_fps = total_frames_rendered / total_elapsed_time
    print("-" * 40)
    print(f"Toplam Süre: {total_elapsed_time:.2f} saniye")
    print(f"Toplam İşlenen Kare: {total_frames_rendered}")
    print(f"Ortalama FPS: {average_fps:.2f}")
    print("-" * 40)

    # Yüz tanıma sonuçlarını özetle
    if recognition_results:
        similarities = [r['similarity'] for r in recognition_results]
        recognized_count = sum(1 for r in recognition_results if r['recognized'])

        print(f"Yüz Tanıma Özeti:")
        print(f"Toplam Tespit: {len(recognition_results)}")
        print(f"Başarılı Tanıma: {recognized_count}")
        print(f"Tanıma Oranı: {recognized_count/len(recognition_results)*100:.1f}%")
        print(f"Ortalama Benzerlik: {np.mean(similarities):.3f}")
        print(f"Max Benzerlik: {np.max(similarities):.3f}")
        print(f"Min Benzerlik: {np.min(similarities):.3f}")
        print("-" * 40)
    else:
        print("Hiç yüz tanıma işlemi gerçekleştirilmedi.")
else:
    print("Program çok hızlı kapatıldı, ortalama FPS hesaplanamadı.")

cv2.destroyAllWindows()
print("Yüz Tanıma Sistemi sonlandırıldı.")
