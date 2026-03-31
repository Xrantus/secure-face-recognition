import cv2
import time
from ultralytics import YOLO
import numpy as np

# ================= AYARLAR =================
# Eger indirdigin modelin adi farkliysa asagidan degistir
STANDART_MODEL_PATH = "face_yolo11_best.onnx"
OZEL_MODEL_PATH = "face_yolo11_best_int8.onnx" 
# ===========================================

print("Modeller hafizaya yukleniyor... Lutfen bekleyin.")
# 1. Standart Orijinal Model
model_standart = YOLO(STANDART_MODEL_PATH)
# 2. Senin Egittigin Yuz Modeli
model_ozel = YOLO(OZEL_MODEL_PATH)

cap = cv2.VideoCapture(0)

print("Kamera basladi! Cikmak icin 'q' tusuna basin.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Her iki model icin ayni kareyi kopyaliyoruz
    frame_std = frame.copy()
    frame_ozel = frame.copy()

    # ---------------------------------------------------------
    # 1. STANDART MODEL (Sadece Insan - Class 0 aranir)
    # ---------------------------------------------------------
    start_time = time.time()
    # Orijinal modelde 80 sinif var, adil karsilastirma icin sadece insani (0) filtreliyoruz
    results_std = model_standart(frame, classes=[0], verbose=False) 
    std_sure = (time.time() - start_time) * 1000 # Milisaniye cinsinden sure
    
    for r in results_std:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            # Kirmizi kutu ciz
            cv2.rectangle(frame_std, (x1, y1), (x2, y2), (0, 0, 255), 2) 
            cv2.putText(frame_std, f"Person: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    cv2.putText(frame_std, "STANDART YOLO11n", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame_std, f"Inference: {std_sure:.1f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ---------------------------------------------------------
    # 2. SENIN OZEL MODELIN (Sadece Yuz)
    # ---------------------------------------------------------
    start_time = time.time()
    results_ozel = model_ozel(frame, verbose=False) 
    ozel_sure = (time.time() - start_time) * 1000
    
    for r in results_ozel:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            # Yesil kutu ciz
            cv2.rectangle(frame_ozel, (x1, y1), (x2, y2), (0, 255, 0), 2) 
            cv2.putText(frame_ozel, f"Face: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame_ozel, "TEZ MODELI (Custom)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame_ozel, f"Inference: {ozel_sure:.1f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---------------------------------------------------------
    # GORUNTULERI YAN YANA BIRLESTIR VE GOSTER
    # ---------------------------------------------------------
    karsilastirma = np.hstack((frame_std, frame_ozel))
    cv2.imshow("YOLO Model Karsilastirmasi", karsilastirma)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()