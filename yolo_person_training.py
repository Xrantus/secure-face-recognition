"""
YOLOv11 Nano Modeli ile Özel Kişi Tanıma Eğitimi
Google Colab için Hazırlanmış Eğitim Scripti

Bu script aşağıdaki adımları gerçekleştirir:
1. Özel veri setini hazırlar (kişi isimleri ile etiketlenmiş)
2. YOLOv11 Nano modelini eğitir
3. Eğitilmiş modeli test eder
4. Modeli export eder

Kullanım:
1. Bu dosyayı Google Colab'a yükleyin
2. Gerekli kütüphaneleri yükleyin
3. Veri setinizi hazırlayın
4. Scripti çalıştırın
"""

import os
import yaml
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import glob

# --- AYARLAR ---
DATASET_PATH = "person_dataset"  # Veri seti klasörü
MODEL_NAME = "yolo11n.pt"       # Başlangıç modeli
EPOCHS = 50                     # Eğitim epoch sayısı
IMG_SIZE = 640                  # Eğitim görüntü boyutu
BATCH_SIZE = 16                 # Batch boyutu (GPU belleğine göre ayarlayın)

# Kişi sınıfları (örnek isimler - kendi isimlerinizle değiştirin)
PERSON_CLASSES = [
    'Ahmet',      # Sınıf 0
    'Buket',      # Sınıf 1
    'Cem',        # Sınıf 2
    'Deniz'       # Sınıf 3
]

# --- VERİ SETİ HAZIRLAMA FONKSİYONLARI ---

def create_dataset_yaml():
    """Dataset YAML dosyasını oluşturur"""
    data = {
        'train': f'{DATASET_PATH}/train/images',
        'val': f'{DATASET_PATH}/valid/images',
        'test': f'{DATASET_PATH}/test/images',
        'nc': len(PERSON_CLASSES),
        'names': PERSON_CLASSES
    }

    with open(f'{DATASET_PATH}/dataset.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Dataset YAML oluşturuldu: {DATASET_PATH}/dataset.yaml")
    print(f"Sınıf sayısı: {len(PERSON_CLASSES)}")
    print(f"Sınıflar: {PERSON_CLASSES}")

def augment_person_images(image_path, save_dir, num_augmentations=5):
    """Veri artırımı ile daha fazla eğitim verisi oluşturur"""
    img = cv2.imread(image_path)
    if img is None:
        return

    filename = os.path.basename(image_path)
    name_without_ext = filename.replace('.jpg', '').replace('.png', '')

    # Orijinal görüntüyü kaydet
    cv2.imwrite(f"{save_dir}/{name_without_ext}_aug0.jpg", img)

    for i in range(num_augmentations):
        augmented = img.copy()

        # Rastgele döndürme
        angle = np.random.randint(-30, 30)
        h, w = augmented.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        augmented = cv2.warpAffine(augmented, M, (w, h))

        # Rastgele parlaklık ayarı
        brightness = np.random.uniform(0.5, 1.5)
        augmented = cv2.convertScaleAbs(augmented, alpha=brightness, beta=0)

        # Rastgele gürültü ekleme
        noise = np.random.normal(0, 25, augmented.shape).astype(np.uint8)
        augmented = cv2.add(augmented, noise)

        # Artırılmış görüntüyü kaydet
        cv2.imwrite(f"{save_dir}/{name_without_ext}_aug{i+1}.jpg", augmented)

def prepare_training_data():
    """Eğitim verilerini hazırlar"""
    print("Veri seti hazırlanıyor...")

    # Klasörleri oluştur
    for split in ['train', 'valid', 'test']:
        os.makedirs(f'{DATASET_PATH}/{split}/images', exist_ok=True)
        os.makedirs(f'{DATASET_PATH}/{split}/labels', exist_ok=True)

        # Mevcut görüntüleri artırılmış versiyonlarla çoğalt
        source_dir = f'{DATASET_PATH}/raw/{split}'
        if os.path.exists(source_dir):
            for img_path in glob.glob(f'{source_dir}/*.jpg') + glob.glob(f'{source_dir}/*.png'):
                augment_person_images(img_path, f'{DATASET_PATH}/{split}/images')

    create_dataset_yaml()
    print("Veri seti hazırlama tamamlandı!")

def train_person_model():
    """YOLOv11 Nano modelini özel kişi sınıfları ile eğitir"""
    print("Model eğitimi başlıyor...")

    # Veri seti YAML dosyasını kontrol et
    yaml_path = f'{DATASET_PATH}/dataset.yaml'
    if not os.path.exists(yaml_path):
        print(f"HATA: {yaml_path} bulunamadı!")
        print("Önce prepare_training_data() fonksiyonunu çalıştırın.")
        return

    # Model yükleme ve eğitim
    model = YOLO(MODEL_NAME)

    # Eğitim parametreleri
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='person_recognition_model',
        device=0 if torch.cuda.is_available() else 'cpu',
        patience=10,  # Early stopping
        save=True,
        save_period=10,
        project='person_recognition_runs',
        exist_ok=True,
        # Özel eğitim parametreleri
        lr0=0.01,  # Başlangıç öğrenme oranı
        lrf=0.1,   # Final öğrenme oranı faktörü
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Veri artırımı
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    print("Model eğitimi tamamlandı!")
    return model

def test_trained_model(model_path=None):
    """Eğitilmiş modeli test eder"""
    if model_path is None:
        # En son eğitilmiş modeli bul
        model_path = 'person_recognition_runs/person_recognition_model/weights/best.pt'

    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return

    print(f"Model test ediliyor: {model_path}")

    # Model yükleme
    model = YOLO(model_path)

    # Test seti ile değerlendirme
    test_results = model.val(
        data=f'{DATASET_PATH}/dataset.yaml',
        split='test',
        save=True,
        save_txt=True,
        save_conf=True,
        project='person_recognition_runs',
        name='test_results'
    )

    print("Test sonuçları:")
    print(f"mAP@0.5: {test_results.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {test_results.box.map:.3f}")
    print(f"Precision: {test_results.box.mp:.3f}")
    print(f"Recall: {test_results.box.mr:.3f}")

    return test_results

def export_model(model_path=None, export_format='onnx'):
    """Eğitilmiş modeli dışa aktarır"""
    if model_path is None:
        model_path = 'person_recognition_runs/person_recognition_model/weights/best.pt'

    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return

    print(f"Model dışa aktarılıyor: {model_path} -> {export_format}")

    # Model yükleme ve dışa aktarma
    model = YOLO(model_path)

    # Farklı formatlara export
    if export_format.lower() == 'onnx':
        model.export(format='onnx', opset=11, simplify=True)
    elif export_format.lower() == 'torchscript':
        model.export(format='torchscript')
    elif export_format.lower() == 'tflite':
        model.export(format='tflite')
    elif export_format.lower() == 'openvino':
        model.export(format='openvino')

    print(f"Model {export_format} formatında dışa aktarıldı!")

# --- ANA ÇALIŞTIRMA FONKSİYONU ---

def main():
    """Ana çalıştırma fonksiyonu"""
    print("=" * 60)
    print("YOLOv11 Nano Kişi Tanıma Eğitim Scripti")
    print("=" * 60)

    # 1. Adım: Veri setini hazırla
    print("\n1. Veri seti hazırlanıyor...")
    prepare_training_data()

    # 2. Adım: Modeli eğit
    print("\n2. Model eğitimi başlıyor...")
    trained_model = train_person_model()

    # 3. Adım: Modeli test et
    print("\n3. Model test ediliyor...")
    test_results = test_trained_model()

    # 4. Adım: Modeli dışa aktar
    print("\n4. Model dışa aktarılıyor...")
    export_model(export_format='onnx')

    print("\n" + "=" * 60)
    print("Eğitim tamamlandı! Artık modeli kullanabilirsiniz.")
    print("=" * 60)

# --- KULLANIM ÖRNEKLERİ ---

def example_usage():
    """Örnek kullanım fonksiyonu"""
    print("Örnek Kullanım:")

    # Sadece belirli adımları çalıştırmak için:
    # prepare_training_data()  # Veri seti hazırlama
    # train_person_model()     # Model eğitimi
    # test_trained_model()     # Model testi
    # export_model()           # Model export

    # Veya tüm adımları otomatik çalıştır:
    main()

if __name__ == "__main__":
    # GPU kontrolü
    if torch.cuda.is_available():
        print(f"GPU kullanılabilir: {torch.cuda.get_device_name(0)}")
        print(f"GPU belleği: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("GPU bulunamadı, CPU kullanılacak.")

    # Ana fonksiyonu çalıştır
    example_usage()
