import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

# ================= AYARLAR =================
IMAGE_FOLDER = "db-images" 
DB_PATH = "known_faces_embeddings.npz"
# ===========================================

print("Yeni veritabani icin 'buffalo_s' yukleniyor...")
app = FaceAnalysis(name="buffalo_s", root=".", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(320, 320))

embeddings = []
names = []

if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
    print(f"Lutfen '{IMAGE_FOLDER}' klasorunun icine yuz fotograflarini koyun ve kodu tekrar calistirin.")
    exit()

print(f"\n--- DEDEKTIF MODU: '{IMAGE_FOLDER}' klasoru taraniyor ---")
dosyalar = os.listdir(IMAGE_FOLDER)
print(f"Klasor icinde bulunan dosyalar: {dosyalar}\n")

if len(dosyalar) == 0:
    print("HATA: Klasor BOMBOŞ! Lütfen fotoğrafları doğru klasörün içine taşıdığından emin ol.")

for filename in dosyalar:
    print(f"-> İnceleniyor: {filename}")
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
        img_path = os.path.join(IMAGE_FOLDER, filename)
        
        try:
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"   [!] HATA: Okuma basarisiz ({e})")
            continue
        
        if img is None:
            print(f"   [!] HATA: Dosya bozuk veya Turkce karakter sorunu.")
            continue
            
        faces = app.get(img)
        if len(faces) > 0:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            emb = face.normed_embedding
            name = os.path.splitext(filename)[0]
            
            embeddings.append(emb)
            names.append(name)
            print(f"   [+] BAŞARILI: {name} eklendi.")
        else:
            print(f"   [-] HATA: Bu fotografta yuz bulunamadi.")
    else:
        print(f"   [!] ATLANDI: Gecerli bir resim formati degil.")

if len(embeddings) > 0:
    np.savez(DB_PATH, encodings=np.array(embeddings), names=np.array(names))
    print(f"\nVeritabani basariyla guncellendi! Toplam {len(names)} kisi kaydedildi.")
else:
    print("\nSonuc: Hic yuz kaydedilemedi.")