"""Database creation script (refactored).

Walks through `db-images/<PersonName>/*` images, detects a face, extracts an embedding,
and stores the per-person mean embedding into an `.npz` file with keys:
`encodings` and `names`.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from . import config
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer


def read_image_unicode(path: str) -> np.ndarray | None:
    """Read an image with Unicode/Turkish characters in the file path."""

    try:
        arr = np.fromfile(path, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None
    return img


def clamp(v: int, lo: int, hi: int) -> int:
    """Clamp integer between lo and hi."""

    return max(lo, min(hi, v))


def crop_with_padding(
    img_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad_ratio: float,
) -> np.ndarray | None:
    """Crop ROI from bbox and pad by ratio (relative to bbox size)."""

    x1, y1, x2, y2 = bbox
    h, w = img_bgr.shape[:2]
    x1 = clamp(x1, 0, w - 1)
    x2 = clamp(x2, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    y2 = clamp(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None

    bw, bh = x2 - x1, y2 - y1
    pw = int(bw * pad_ratio)
    ph = int(bh * pad_ratio)
    roi = img_bgr[max(0, y1 - ph) : min(h, y2 + ph), max(0, x1 - pw) : min(w, x2 + pw)]
    if roi.size == 0:
        return None
    return roi


def create_db(
    image_folder: str,
    output_npz_path: str,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
) -> None:
    """Create an embeddings DB from a folder structure `image_folder/<name>/*.jpg`."""

    image_root = Path(image_folder)
    image_root.mkdir(parents=True, exist_ok=True)

    person_dirs = sorted([p for p in image_root.iterdir() if p.is_dir()])
    if not person_dirs:
        raise SystemExit(
            f"HATA: Hic kisi klasoru bulunamadi! Ornek yapi: {image_folder}/Ali/foto1.jpg"
        )

    global_embeddings: list[np.ndarray] = []
    global_names: list[str] = []

    print(f"Toplam {len(person_dirs)} kisi klasoru bulundu.")
    print("=" * 50)

    for person_dir in person_dirs:
        name = person_dir.name
        print(f"\n[{name}] isleniyor...")

        per_embeddings: list[np.ndarray] = []
        for filename in sorted(os.listdir(person_dir)):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                continue

            img_path = str(person_dir / filename)
            img = read_image_unicode(img_path)
            if img is None:
                print(f"  [!] {filename} okunamadi veya bozuk.")
                continue

            dets = detector.detect(img)
            best = FaceDetector.best_by_conf(dets)
            if best is None:
                print(f"  [-] {filename}: yuz bulunamadi.")
                continue

            roi = crop_with_padding(img, best.bbox, config.MODEL_CONFIG.landmark_pad)
            if roi is None:
                print(f"  [-] {filename}: ROI gecersiz.")
                continue

            emb = recognizer.embed_from_roi(roi)
            if emb is None:
                print(f"  [-] {filename}: landmark/embedding bulunamadi.")
                continue

            per_embeddings.append(emb)
            print(f"  [+] {filename}: OK")

        if per_embeddings:
            mean_emb = np.mean(np.stack(per_embeddings, axis=0), axis=0)
            mean_emb = FaceRecognizer.l2_normalize(mean_emb)
            global_embeddings.append(mean_emb)
            global_names.append(name)
            print(f"  => '{name}': {len(per_embeddings)} fotograf islendi, ortalama embedding kaydedildi.")
        else:
            print(f"  => '{name}': Gecerli hic yuz bulunamadi, atlandi.")

    print("\n" + "=" * 50)
    if not global_embeddings:
        print("SONUC: Hic embedding kaydedilemedi.")
        return

    np.savez(output_npz_path, encodings=np.array(global_embeddings), names=np.array(global_names))
    print(f"Veritabani kaydedildi: '{output_npz_path}'")
    print(f"Toplam kayitli kisi: {len(global_names)}")
    for n in global_names:
        print(f"  - {n}")


def main() -> None:
    """CLI entry-point for DB creation."""

    project_root = Path(__file__).resolve().parents[1]

    detector = FaceDetector(
        model_path=config.MODEL_CONFIG.yolo_model_path,
        img_size=config.MODEL_CONFIG.yolo_img_size,
        pred_conf=config.MODEL_CONFIG.yolo_pred_conf,
        iou=config.MODEL_CONFIG.yolo_iou,
        max_det=config.MODEL_CONFIG.max_det,
        det_threshold=config.MODEL_CONFIG.yolo_det_threshold,
    )

    recognizer = FaceRecognizer(
        det_size=config.MODEL_CONFIG.det_size,
        model_name=config.MODEL_CONFIG.recognizer_model_name,
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
        if config.HARDWARE_ENV == "MAC"
        else None,
    )

    image_folder = str(project_root / "db-images")
    output_npz = str(project_root / config.MODEL_CONFIG.db_path)

    print("Veritabani olusturma basliyor...")
    create_db(image_folder=image_folder, output_npz_path=output_npz, detector=detector, recognizer=recognizer)


if __name__ == "__main__":
    main()

