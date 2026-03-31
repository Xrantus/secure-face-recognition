# -*- coding: utf-8 -*-
# --- GEREKLİ KÜTÜPHANELERİ IMPORT ETME ---
import cv2
import numpy as np
from PIL import Image as PILImage # PIL Image'ı farklı isimle import ettim
import io
import time
import os
import pandas as pd
import warnings
from deepface import DeepFace
from ultralytics import YOLO

# Uyarıları bastır (isteğe bağlı)
warnings.filterwarnings("ignore", category=FutureWarning, module='deepface.*')
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.*')
warnings.filterwarnings("ignore", category=UserWarning, module='google.*')

print("Kütüphaneler import edildi.")

# --- YARDIMCI FONKSİYONLAR ---

# YOLO ile 'person' tespiti (Aynı kalabilir)
def detect_person(frame, yolo_model):
    person_boxes = []
    # classes=[0] -> Sadece 'person' sınıfı
    # conf=0.5 -> Güven skoru %50'den yüksek olanları al
    results = yolo_model(frame, classes=[0], conf=0.5, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            person_boxes.append((x1, y1, x2, y2))
    return person_boxes

# Yüzü kırpma (Aynı kalabilir)
def extract_face(frame, person_bbox):
    x1, y1, x2, y2 = person_bbox
    # Kırpma sınırlarını kontrol et
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    face_crop = frame[y1:y2, x1:x2]
    return face_crop

# DeepFace ile yüz tanıma (Aynı kalabilir)
def recognize_face(face_img, db_path, model_name):
    try:
        # enforce_detection=False -> DeepFace'e verilen kırpılmış resimde
        # yüz bulamazsa hata verme, doğrudan karşılaştırmayı dene.
        dfs = DeepFace.find(img_path = face_img,
                            db_path = db_path,
                            model_name=model_name,
                            enforce_detection=False, # Önemli!
                            distance_metric='cosine',
                            threshold=0.45, # Eşik değeri (deneyerek ayarla)
                            silent=True) # Konsolu temiz tut
        return dfs
    except ValueError as ve:
        # print(f"DeepFace ValueError (muhtemelen yüz bulunamadı): {ve}") # Hata ayıklama
        return None
    except Exception as e:
        # print(f"DeepFace recognize_face error: {e}") # Hata ayıklama
        return None # Hata durumunda None döndür

print("Yardımcı fonksiyonlar tanımlandı.")
# --- YARDIMCI FONKSİYONLAR SONU ---

# --- MODEL YÜKLEME VE DB HAZIRLIĞI ---
try:
    # === GÜNCELLEME: Model yolunu yerel yolunla değiştir ===
    yolo_model_path = 'yolo11n.pt' # Eğer dosya kodla aynı dizindeyse
    # yolo_model_path = 'C:/path/to/your/model/yolov8n.pt' # Veya tam yolunu ver
    yolo_model = YOLO(yolo_model_path)
    print(f"YOLO modeli yüklendi: {yolo_model_path}")

    # === GÜNCELLEME: Veritabanı yolunu yerel yolunla değiştir ===
    # ÖNEMLİ: Klasör ayırıcıları işletim sistemine uygun olmalı (Windows: \\ veya /, Linux/Mac: /)
    ahmet_database_path = "YOLO_Egitim/dataset/" # Eğer kodun çalıştığı yerden göreceli bir yolsa
    # ahmet_database_path = "/Users/kullanici/Belgelerim/YOLO_Egitim/dataset/" # Örnek Mac/Linux yolu
    # ahmet_database_path = "C:\\Users\\Kullanici\\Documents\\YOLO_Egitim\\dataset\\" # Örnek Windows yolu
    # Klasörün var olup olmadığını kontrol et
    if not os.path.isdir(ahmet_database_path):
        raise FileNotFoundError(f"Veritabanı klasörü bulunamadı: {ahmet_database_path}")
    print(f"Ahmet'in veritabanı yolu: {ahmet_database_path}")

    PKL_FILE_PATH = os.path.join(ahmet_database_path, "representations_vggface.pkl")

    # DeepFace modelini ve veritabanını hazırla
    RECOGNITION_MODEL = 'VGG-Face'
    print(f"DeepFace veritabanı ('{ahmet_database_path}') hazırlanıyor...")

    DeepFace.build_model(RECOGNITION_MODEL)
    print(f"{RECOGNITION_MODEL} modeli yüklendi.")

    # Temsil dosyalarını oluştur/yükle (actions parametresi olmadan)
    # İlk çalıştırmada biraz zaman alabilir.
    _ = DeepFace.find(img_path = np.zeros([1,1,3], dtype=np.uint8),
                      db_path = ahmet_database_path,
                      model_name=RECOGNITION_MODEL,
                      enforce_detection=False,
                      silent=True)

    # .pkl dosyasını kontrol et
    pkl_found = False
    for file in os.listdir(ahmet_database_path):
        if file.endswith(".pkl"):
             print(f"Veritabanı temsil dosyası bulundu: {file}")
             pkl_found = True
             break
    if not pkl_found:
         print(f"UYARI: Temsil dosyası (.pkl) '{ahmet_database_path}' içinde OLUŞTURULAMADI veya BULUNAMADI.")

    print("Modeller ve DB hazırlığı tamamlandı.")

except FileNotFoundError as fnf_err:
    print(f"HATA: Gerekli dosya veya klasör bulunamadı: {fnf_err}")
    exit() # Hata durumunda programdan çık
except Exception as e:
    print(f"HATA: Hazırlık aşamasında sorun oluştu: {e}")
    exit() # Hata durumunda programdan çık
# --- HAZIRLIK SONU ---

# --- WEBCAM BAŞLATMA ---
cap = cv2.VideoCapture(0) # 0 genellikle varsayılan webcam'dir

if not cap.isOpened():
    print("HATA: Webcam açılamadı. Başka bir programın kullanmadığından emin olun.")
    exit()

# --- VİDEO KAYDETME AYARLARI ---
recording = False
video_writer = None
output_folder = "kayitlar"  # Kayıt klasörü
os.makedirs(output_folder, exist_ok=True)  # Klasör yoksa oluştur

# Video codec ve ayarları
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI formatı için
fps = 20.0  # Saniyedeki kare sayısı
frame_size = (640, 480)  # Çıktı video boyutu

print("Webcam başlatıldı.")
print("Kontroller:")
print("- 'q': Çıkış")
print("- 'r': Video kaydetmeyi başlat/durdur")
print("- 's': Anlık ekran görüntüsü al")
time.sleep(1) # Kameranın başlaması için kısa bir bekleme

# --- VİDEO KAYDETME FONKSİYONU ---
def start_video_recording():
    global video_writer, recording
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_filename = f"video_kayit_{timestamp}.avi"
    video_path = os.path.join(output_folder, video_filename)

    # Video boyutunu kameradan al
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        recording = True
        print(f"Video kaydediliyor: {video_filename}")
        return True
    return False

def stop_video_recording():
    global video_writer, recording
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    recording = False
    print("Video kaydı durduruldu.")

def take_screenshot(frame):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    screenshot_filename = f"ekran_goruntusu_{timestamp}.png"
    screenshot_path = os.path.join(output_folder, screenshot_filename)

    # Görüntüyü kaydet
    cv2.imwrite(screenshot_path, frame)
    print(f"Ekran görüntüsü kaydedildi: {screenshot_filename}")

# --- PERFORMANS OPTİMİZASYONU AYARLARI ---
TARGET_FPS = 30  # Hedef FPS
FRAME_TIME = 1.0 / TARGET_FPS  # Her kare için geçmesi gereken süre (saniye)

PROCESS_INTERVAL = 10  # Her 10 karede bir yüz tanıma yap (daha seyrek)
PERSISTENT_BOXES_DURATION = 60  # Bounding box'ları 60 kare göster (2 saniye @ 30fps)

# --- GERÇEK ZAMANLI TANIMA DÖNGÜSÜ ---
frame_count = 0
last_process_frame = 0
persistent_boxes = []  # Önceki tanıma sonuçlarını sakla
box_frame_counter = {}  # Her bounding box'ın gösterim süresini takip et

window_name = "Ahmet Tanima - Cikis icin 'q'" # Pencere başlığı

# FPS hesaplama için değişkenler
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0.0

try:
    while True:
        loop_start_time = time.time()

        # Kameradan bir kare oku
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Hata: Kameradan kare okunamadı.")
            time.sleep(0.5) # Bekleyip tekrar dene
            continue

        # Görüntüyü yatay olarak çevir (ayna efekti için)
        processed_frame = cv2.flip(frame, 1)
        frame_count += 1
        fps_frame_count += 1

        # FPS hesapla (her saniye)
        if fps_frame_count % 30 == 0:
            elapsed_time = time.time() - fps_start_time
            current_fps = 30 / elapsed_time if elapsed_time > 0 else 0
            fps_start_time = time.time()
            fps_frame_count = 0

        # Sadece belirli aralıklarla yüz tanıma yap (daha seyrek)
        current_boxes = []
        if frame_count - last_process_frame >= PROCESS_INTERVAL:
            # YOLO ile 'person' tespiti
            person_boxes = detect_person(processed_frame, yolo_model)

            # Tespit edilen her kişi için
            for bbox in person_boxes:
                # Yüzü kırp
                face_img = extract_face(processed_frame, bbox)

                # Kırpılan bölge geçerliyse tanıma yap
                if face_img is not None and face_img.shape[0] > 10 and face_img.shape[1] > 10:
                    # DeepFace ile tanıma
                    recognition_results = recognize_face(face_img, ahmet_database_path, RECOGNITION_MODEL)

                    # Sonucu işle ve etiketi hazırla
                    label = "Diger_Kisi"
                    color = (0, 0, 255) # Kırmızı
                    distance = -1

                    if recognition_results is not None and isinstance(recognition_results, list) and len(recognition_results) > 0:
                         if isinstance(recognition_results[0], pd.DataFrame) and not recognition_results[0].empty:
                            result_df = recognition_results[0]
                            best_match = result_df.iloc[result_df['distance'].idxmin()]
                            identity_path = best_match['identity']
                            distance = best_match['distance']

                            # Kişi adını klasör isminden al
                            person_folder = os.path.basename(os.path.dirname(identity_path))
                            person_name = person_folder

                            # Bilinen kişilerin renklerini belirle
                            if person_name == "Ahmet":
                                color = (0, 255, 0) # Yeşil - Ahmet
                                label = f"Ahmet ({distance:.2f})"
                            elif person_name == "Hatice":
                                color = (255, 0, 255) # Magenta - Hatice
                                label = f"Hatice ({distance:.2f})"
                            else:
                                color = (0, 165, 255) # Turuncu - Diğer kişiler
                                label = f"{person_name} ({distance:.2f})"

                    # Bounding box bilgilerini persistent_boxes'a ekle
                    box_info = {
                        'bbox': bbox,
                        'label': label,
                        'color': color,
                        'frame_count': frame_count
                    }
                    current_boxes.append(box_info)

            # İşlem yapıldı olarak işaretle
            last_process_frame = frame_count

        # Önceki tanıma sonuçlarını güncelle ve çiz
        if current_boxes:
            # Yeni sonuçları persistent_boxes'a ekle
            persistent_boxes.extend(current_boxes)
            # Her bounding box için sayaç başlat
            for box_info in current_boxes:
                box_key = f"{box_info['bbox']}_{frame_count}"
                box_frame_counter[box_key] = 0

        # Süresi dolan bounding box'ları temizle
        persistent_boxes = [
            box for box in persistent_boxes
            if frame_count - box['frame_count'] < PERSISTENT_BOXES_DURATION
        ]

        # Tüm persistent bounding box'ları çiz
        for box_info in persistent_boxes:
            x1, y1, x2, y2 = box_info['bbox']
            color = box_info['color']
            label = box_info['label']

            # Bounding box'ı çiz
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(processed_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Video kaydediliyorsa kareyi kaydet
        if recording and video_writer is not None:
            video_writer.write(processed_frame)

        # Ekranda bilgi göster
        status_text = "KAYIT: AKTIF" if recording else "KAYIT: DURDURULDU"
        cv2.putText(processed_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if recording else (0, 0, 255), 2)

        # FPS ve performans bilgilerini göster
        fps_text = f"FPS: {current_fps:.1f}"
        process_text = f"Son islem: {frame_count - last_process_frame} kare once"
        cv2.putText(processed_frame, fps_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(processed_frame, process_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # İşlenmiş kareyi ekranda göster
        cv2.imshow(window_name, processed_frame)

        # FPS kontrolü için bekleme
        elapsed_time = time.time() - loop_start_time
        remaining_time = max(0, FRAME_TIME - elapsed_time)
        time.sleep(remaining_time)

        # Tuş kontrolleri
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n'q' tuşuna basıldı, çıkılıyor...")
            break
        elif key == ord('r'):
            if not recording:
                if start_video_recording():
                    print("Video kaydı başlatıldı.")
                else:
                    print("Video kaydı başlatılamadı.")
            else:
                stop_video_recording()
        elif key == ord('s'):
            take_screenshot(processed_frame)

except KeyboardInterrupt:
    print("\nKullanıcı tarafından durduruldu (Ctrl+C).")
except Exception as e:
    print(f"\nBeklenmedik bir hata oluştu: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Video kaydediliyorsa durdur
    if recording:
        stop_video_recording()

    # Webcam'i serbest bırak ve pencereleri kapat
    if 'cap' in locals() and cap.isOpened():
        cap.release()
        print("Webcam serbest bırakıldı.")
    cv2.destroyAllWindows()
    print("Tüm pencereler kapatıldı.")
    print("Tanıma işlemi bitti.")