import cv2
import time
from ultralytics import YOLO
import threading 

# --- AYARLAR ---
FRAME_SKIP = 6 
YOLO_IMG_SIZE = 160 
HAAR_SCALE_FACTOR = 1.3 
HAAR_MIN_NEIGHBORS = 5  
HAAR_MIN_SIZE = (30, 30) 

# --- MODEL YÜKLEME ---
model = YOLO('yolo11n.pt')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- KAMERA I/O THREAD ---
latest_frame = None
frame_lock = threading.Lock()
running = True

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
print("Face Detection başlatıldı!")

# --- ANA DÖNGÜ DEĞİŞKENLERİ ---
frame_counter = 0
last_results = None
fps_start_time = time.time()
fps_frame_count = 0
# --- Ortalama FPS için eklendi ---
total_frames_rendered = 0
program_start_time = time.time() # Programın toplam çalışma süresi için
# --- Eklendi son ---

while running:
    with frame_lock:
        if latest_frame is None:
            continue
        frame = latest_frame.copy() 

    frame = cv2.flip(frame, 1)

    # FPS hesaplaması için kare sayacını artır
    fps_frame_count += 1
    total_frames_rendered += 1 # Ortalama FPS için eklendi
    
    if frame_counter % FRAME_SKIP == 0:
        results = model(frame, stream=True, imgsz=YOLO_IMG_SIZE, verbose=False)
        last_results = list(results) 

    if last_results:
        for r in last_results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                if class_name == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    
                    if confidence > 0.5:
                        person_roi = frame[y1:y2, x1:x2]
                        
                        if person_roi.size == 0:
                            continue

                        gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(
                            gray_roi,
                            scaleFactor=HAAR_SCALE_FACTOR,
                            minNeighbors=HAAR_MIN_NEIGHBORS,
                            minSize=HAAR_MIN_SIZE,
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        
                        if len(faces) > 0:
                            face = max(faces, key=lambda x: x[2] * x[3])
                            fx, fy, fw, fh = face
                            
                            face_x1 = x1 + fx
                            face_y1 = y1 + fy
                            face_x2 = x1 + fx + fw
                            face_y2 = y1 + fy + fh
                            
                            cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 255, 0), 2)
                            
                            label = f'Face {confidence:.2f}'
                            cv2.putText(frame, label, (face_x1, face_y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            face_height = int((y2 - y1) * 0.3)
                            face_y1 = y1
                            face_y2 = y1 + face_height
                            
                            cv2.rectangle(frame, (x1, face_y1), (x2, face_y2), (0, 255, 0), 2)
                            
                            label = f'Face (Est) {confidence:.2f}'
                            cv2.putText(frame, label, (x1, face_y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if time.time() - fps_start_time >= 1.0:
        fps = fps_frame_count / (time.time() - fps_start_time)
        print(f"Anlık FPS: {fps:.2f}")
        fps_frame_count = 0
        fps_start_time = time.time()

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break
    
    frame_counter += 1

# Kaynakları serbest bırak
print("Çıkış yapılıyor...")
running = False
reader_thread.join(timeout=1) 

# --- Ortalama FPS'i hesapla ve yazdır (Eklendi) ---
program_end_time = time.time()
total_elapsed_time = program_end_time - program_start_time

if total_elapsed_time > 0:
    average_fps = total_frames_rendered / total_elapsed_time
    print("-" * 30)
    print(f"Toplam Süre: {total_elapsed_time:.2f} saniye")
    print(f"Toplam İşlenen Kare: {total_frames_rendered}")
    print(f"Ortalama FPS: {average_fps:.2f}")
    print("-" * 30)
else:
    print("Program çok hızlı kapatıldı, ortalama FPS hesaplanamadı.")
# --- Eklendi son ---

cv2.destroyAllWindows()
print("Face Detection sonlandırıldı.")