import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ================= AYARLAR =================
IMAGE_FOLDER = "db-images" 
DB_PATH = "known_faces_embeddings.npz"
# ===========================================

print("Yeni veritabani icin 'buffalo_s' yukleniyor...")
app = FaceAnalysis(name="buffalo_s", root=".", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(320, 320))


def _landmarks_list(kpss):
    """det_model.detect kpss ciktisi list veya ndarray olabilir; (5,2) landmark listesine cevir."""
    if kpss is None:
        return []
    if isinstance(kpss, np.ndarray):
        if kpss.size == 0:
            return []
        if kpss.ndim == 2 and kpss.shape == (5, 2):
            return [kpss]
        if kpss.ndim == 3 and kpss.shape[1:] == (5, 2):
            return [kpss[i] for i in range(kpss.shape[0])]
        return []
    return [np.asarray(k) for k in kpss] if kpss else []


global_embeddings = []
global_names = []

if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
    print(f"Lutfen '{IMAGE_FOLDER}' klasorunun icine kisilerin isimleriyle klasorler olusturup fotograflarini koyun.")
    exit()

print(f"\n--- DEDEKTIF MODU: '{IMAGE_FOLDER}' klasoru taraniyor ---")
# Sadece klasorleri bul (icerideki alakasiz tekil dosyalari atla)
kisi_klasorleri = [f for f in os.listdir(IMAGE_FOLDER) if os.path.isdir(os.path.join(IMAGE_FOLDER, f))]

if len(kisi_klasorleri) == 0:
    print("HATA: Ana klasorde hic kisi klasoru bulunamadi!")
    print("Ornek yapi: db-images/Cenk/foto1.jpg, db-images/Cenk/foto2.jpg")
    exit()

for kisi_adi in kisi_klasorleri:
    print(f"\n-> İŞLENİYOR: '{kisi_adi}' klasoru")
    kisi_klasor_yolu = os.path.join(IMAGE_FOLDER, kisi_adi)
    dosyalar = os.listdir(kisi_klasor_yolu)
    
    kisi_embeddings = []
    
    for filename in dosyalar:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
            img_path = os.path.join(kisi_klasor_yolu, filename)
            
            try:
                # Turkce karakterli yollari sorunsuz okumak icin
                img_array = np.fromfile(img_path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"   [!] HATA: {filename} okuma basarisiz ({e})")
                continue
            
            if img is None:
                print(f"   [!] HATA: {filename} bozuk veya okunamiyor.")
                continue

            bboxes, kpss = app.det_model.detect(img, max_num=0, metric="default")
            lm_list = _landmarks_list(kpss)
            if not lm_list:
                print(f"   [-] HATA: {filename} (Yuz bulunamadi)")
                continue

            b = np.asarray(bboxes)
            if b.size and b.ndim == 1:
                b = b.reshape(1, -1)
            if b.size and b.shape[0] == len(lm_list):
                areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                best_i = int(np.argmax(areas))
            else:
                best_i = 0
            aligned = face_align.norm_crop(img, landmark=lm_list[best_i])
            emb = app.models["recognition"].get_feat(aligned)[0]
            emb = emb / np.linalg.norm(emb)
            kisi_embeddings.append(emb)
            print(f"   [+] BAŞARILI: {filename} (Yuz bulundu)")
                
    # Kisiye ait tum fotograflar islendi, simdi ortalama (mean) alalim
    if len(kisi_embeddings) > 0:
        # 1. Adim: Vektorlerin matematiksel ortalamasini al
        ortalama_emb = np.mean(kisi_embeddings, axis=0)
        
        # 2. Adim: L2 Normalizasyonu (Cosine Similarity icin sart!)
        ortalama_emb = ortalama_emb / np.linalg.norm(ortalama_emb)
        
        global_embeddings.append(ortalama_emb)
        global_names.append(kisi_adi) # Klasor adi = Kisi adi
        
        print(f"=== SONUC: '{kisi_adi}' icin {len(kisi_embeddings)} fotografin ortalamasi basariyla kaydedildi! ===")
    else:
        print(f"=== SONUC: '{kisi_adi}' klasorunde gecerli hic yuz bulunamadi! ===")

if len(global_embeddings) > 0:
    np.savez(DB_PATH, encodings=np.array(global_embeddings), names=np.array(global_names))
    print(f"\nVeritabani basariyla guncellendi! Toplam {len(global_names)} kisi kaydedildi.")
else:
    print("\nSonuc: Hic yuz kaydedilemedi.")