import cv2
import time
from ultralytics import YOLO

# --- AYARLAR ---
# Her kaç karede bir nesne tespiti yapılacağını belirler.
# Değeri artırmak FPS'i artırır ama tespitlerin güncelliğini azaltır.
FRAME_SKIP = 3 

# Modelimizi yüklüyoruz (nano versiyonu hız için en iyisidir)
model = YOLO('yolo11n.pt')

# Webcam'i başlatıyoruz
cap = cv2.VideoCapture(0)

# Çözünürlüğü düşürerek performansı artırıyoruz
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Hata: Webcam başlatılamadı.")
    exit()

# Değişkenler
frame_counter = 0
last_results = None
fps_start_time = time.time()
fps_frame_count = 0

while True:
    # Kameradan bir kare oku
    ret, frame = cap.read()
    if not ret:
        print("Hata: Kare okunamadı.")
        break

    # FPS hesaplaması için kare sayacını artır
    fps_frame_count += 1
    
    # Sadece belirlenen aralıklarla modeli çalıştır
    if frame_counter % FRAME_SKIP == 0:
        # Modeli daha küçük bir görüntü boyutuyla çalıştırarak hızı artır
        results = model(frame, stream=True, imgsz=320, verbose=False)
        last_results = list(results) # Sonuçları sakla

    # Eğer daha önce bir sonuç elde ettiysek, onu ekrana çiz
    if last_results:
        for r in last_results:
            boxes = r.boxes
            for box in boxes:
                # Sınıf ID'sini al
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                # Sadece 'person' sınıfını çizdirmek isterseniz:
                # if class_name == 'person':
                # Koordinatları al
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Dikdörtgeni çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                
                # Etiketi yaz
                label = f'{class_name} {box.conf[0]:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # FPS'i hesapla ve ekrana yaz
    if time.time() - fps_start_time >= 1.0:
        fps = fps_frame_count / (time.time() - fps_start_time)
        print(f"Anlık FPS: {fps:.2f}")
        fps_frame_count = 0
        fps_start_time = time.time()

    cv2.imshow('Optimize Edilmis Webcam', frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_counter += 1

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()