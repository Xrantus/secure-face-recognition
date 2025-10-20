import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ====================== AYARLAR ======================
DB_FOLDER = 'db-images'                      # db-images/<isim>/<fotoğraflar>
OUTPUT_FILE = 'known_faces_embeddings.npz'   # Çıkış dosyası
EMBEDDING_THRESHOLD = 0.5                    # Minimum güven skoru
MIN_FACE_SIZE = 100                          # Çok küçük yüzleri ele (px)
# =====================================================

print("InsightFace modeli yükleniyor...")

# InsightFace modelini yükle (GPU varsa 0, yoksa CPU -1)
try:
    app = FaceAnalysis(name='buffalo_l', root='.')
    app.prepare(ctx_id=0, det_size=(640, 640))
except Exception:
    app = FaceAnalysis(name='buffalo_l', root='.')
    app.prepare(ctx_id=-1, det_size=(640, 640))

print("InsightFace modeli başarıyla yüklendi.\n")

known_face_encodings = []
known_face_names = []

# =====================================================
# VERITABANI OLUSTURMA
# =====================================================
if not os.path.exists(DB_FOLDER):
    print(f"Hata: '{DB_FOLDER}' klasörü bulunamadı. Lütfen önce bu klasörü oluşturun.")
    exit()

for person_name in sorted(os.listdir(DB_FOLDER)):
    person_dir = os.path.join(DB_FOLDER, person_name)
    if not os.path.isdir(person_dir):
        continue

    print(f"\n{person_name} için embedding oluşturuluyor...")

    person_embeddings = []

    for filename in sorted(os.listdir(person_dir)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        file_path = os.path.join(person_dir, filename)
        img = cv2.imread(file_path)

        if img is None:
            print(f"Uyari: Goruntu okunamadi: {file_path}")
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"Uyari: Yuz bulunamadi: {filename}")
            continue

        for face in faces:
            w = face.bbox[2] - face.bbox[0]
            h = face.bbox[3] - face.bbox[1]
            if min(w, h) < MIN_FACE_SIZE:
                print(f"Yuz cok kucuk ({min(w,h):.0f}px), atlandi: {filename}")
                continue

            if face.det_score < EMBEDDING_THRESHOLD:
                print(f"Dusuk guven ({face.det_score:.2f}), atlandi: {filename}")
                continue

            emb = face.normed_embedding  # normalize edilmiş 512-dim vektör
            person_embeddings.append(emb)

    if len(person_embeddings) == 0:
        print(f"{person_name} icin gecerli yuz bulunamadi, atlandi.")
        continue

    mean_emb = np.mean(np.vstack(person_embeddings), axis=0)
    mean_emb = mean_emb / np.linalg.norm(mean_emb)
    known_face_encodings.append(mean_emb)
    known_face_names.append(person_name)

    print(f"{person_name}: {len(person_embeddings)} yuz -> tek vektor olusturuldu.")

# =====================================================
# KAYDETME
# =====================================================
if len(known_face_encodings) == 0:
    print("Hic embedding uretilmedi. Fotoğrafları kontrol et.")
    exit()

known_face_encodings = np.vstack(known_face_encodings)
known_face_names = np.array(known_face_names)

np.savez_compressed(
    OUTPUT_FILE,
    encodings=known_face_encodings,
    names=known_face_names
)

print("\n-----------------------------------------------")
print(f"Basarili! {len(known_face_names)} kisi icin embedding kaydedildi.")
print(f"Dosya: {OUTPUT_FILE}")
print("-----------------------------------------------")
