import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ================= AYARLAR =================
IMAGE_FOLDER = "db-images"
DB_PATH = "known_faces_embeddings.npz"
DET_SIZE = (320, 320)
# ===========================================

print("Veritabani olusturma basliyor...")
print("'buffalo_s' yukleniyor...")
app = FaceAnalysis(name="buffalo_s", root=".", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=DET_SIZE)

global_embeddings = []
global_names = []

if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
    print(f"'{IMAGE_FOLDER}' klasoru olusturuldu.")
    print("Lutfen icine kisi isimli klasorler olusturup fotograflari ekleyin.")
    print("Ornek: db-images/Ali/foto1.jpg")
    exit()

kisi_klasorleri = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if os.path.isdir(os.path.join(IMAGE_FOLDER, f))
])

if len(kisi_klasorleri) == 0:
    print("HATA: Hic kisi klasoru bulunamadi!")
    print("Ornek yapi: db-images/Ali/foto1.jpg, db-images/Veli/foto1.jpg")
    exit()

print(f"\nToplam {len(kisi_klasorleri)} kisi klasoru bulundu.")
print("=" * 50)

for kisi_adi in kisi_klasorleri:
    print(f"\n[{kisi_adi}] isleniyor...")
    kisi_klasor_yolu = os.path.join(IMAGE_FOLDER, kisi_adi)
    dosyalar = sorted(os.listdir(kisi_klasor_yolu))

    kisi_embeddings = []

    for filename in dosyalar:
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
            continue

        img_path = os.path.join(kisi_klasor_yolu, filename)

        try:
            # Turkce karakter destekli okuma
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"  [!] {filename} okunamadi: {e}")
            continue

        if img is None:
            print(f"  [!] {filename} bozuk veya desteklenmiyor.")
            continue

        # --- DUZELTILMIS PIPELINE ---
        # Inference ile ayni yoldan gec:
        # det_model → landmark → norm_crop → recognition
        try:
            bboxes, kpss = app.det_model.detect(img, max_num=1, metric='default')
        except Exception as e:
            print(f"  [!] {filename} detection hatasi: {e}")
            continue

        if kpss is None or len(kpss) == 0:
            print(f"  [-] {filename}: Yuz veya landmark bulunamadi.")
            continue

        # En buyuk yuzu al (max_num=1 zaten en iyisini veriyor)
        kps = kpss[0]  # shape: (5, 2)

        # Affine alignment → standart 112x112 ArcFace formati
        aligned_face = face_align.norm_crop(img, landmark=kps)

        # Sadece recognition modelini calistir
        emb = app.models['recognition'].get_feat(aligned_face)[0]
        emb = emb / np.linalg.norm(emb)

        kisi_embeddings.append(emb)
        print(f"  [+] {filename}: OK")

    # Kisi icin ortalama embedding hesapla
    if len(kisi_embeddings) > 0:
        ortalama_emb = np.mean(kisi_embeddings, axis=0)
        ortalama_emb = ortalama_emb / np.linalg.norm(ortalama_emb)  # L2 normalize

        global_embeddings.append(ortalama_emb)
        global_names.append(kisi_adi)

        print(f"  => '{kisi_adi}': {len(kisi_embeddings)} fotograf islendi, ortalama embedding kaydedildi.")
    else:
        print(f"  => '{kisi_adi}': Gecerli hic yuz bulunamadi, atlandi.")

print("\n" + "=" * 50)

if len(global_embeddings) > 0:
    np.savez(
        DB_PATH,
        encodings=np.array(global_embeddings),
        names=np.array(global_names)
    )
    print(f"Veritabani kaydedildi: '{DB_PATH}'")
    print(f"Toplam kayitli kisi: {len(global_names)}")
    for n in global_names:
        print(f"  - {n}")
else:
    print("SONUC: Hic embedding kaydedilemedi.")
