import cv2
import os

# Gecikmeyi (latency) minimuma indirmek için FFMPEG ayarları
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# PC'nizin IP adresi ve VLC'de belirlediğiniz Port/Path
# Örnek: PC IP adresiniz 192.168.1.100 ise
rtsp_url = "rtsp://10.69.36.232:8554/test"

# Stream'i başlat
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

# Buffer boyutunu 1'e düşürerek eski frame'lerin birikmesini ve lag oluşmasını engelliyoruz
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("RTSP stream açılamadı. IP ve Port'u kontrol edin.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Frame okunamadı. Bağlantı kopmuş olabilir.")
        break

    # --- Face Recognition işlemlerinizi bu "frame" değişkeni üzerinde yapacaksınız ---
    
    # Görüntüyü ekranda göster
    cv2.imshow('Raspberry Pi - RTSP Stream', frame)

    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()