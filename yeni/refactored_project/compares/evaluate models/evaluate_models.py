"""
Yuz Tanima Modelleri Kapsamli Karsilastirma Scripti
=====================================================
Test edilen modeller:
  InsightFace Ailesi : buffalo_s, buffalo_l, antelopev2
  Edge / Hafif       : sface (OpenCV), ghostfacenet
  Baseline / Literat.: facenet512

Ciktiler:
  - Konsol raporu (TAR, FAR, FPS, ms)
  - compares/raw_scores.csv  -> threshold deneyleri icin
  - compares/fps_comparison.png
  - compares/accuracy_comparison.png
"""

import sys
import os
import csv
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Yol ayarlari -----------------------------------------------------------
refactored_root = Path(__file__).resolve().parent.parent   # …/refactored_project
workspace_root  = refactored_root.parent                   # …/yeni
if str(refactored_root) not in sys.path:
    sys.path.append(str(refactored_root))

from face_detector   import FaceDetector
from face_recognizer import FaceRecognizer


# ============================================================================
# YARDIMCI FONKSIYONLAR
# ============================================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def crop_with_padding(img_bgr, bbox, pad_ratio=0.20):
    x1, y1, x2, y2 = bbox
    h, w = img_bgr.shape[:2]
    x1, x2 = clamp(x1, 0, w-1), clamp(x2, 0, w-1)
    y1, y2 = clamp(y1, 0, h-1), clamp(y2, 0, h-1)
    if x2 <= x1 or y2 <= y1:
        return None
    bw, bh = x2 - x1, y2 - y1
    if min(bw, bh) < 35:
        return None
    pw, ph = int(bw * pad_ratio), int(bh * pad_ratio)
    roi = img_bgr[max(0, y1-ph):min(h, y2+ph),
                  max(0, x1-pw):min(w, x2+pw)]
    return roi if roi.size > 0 else None


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ============================================================================
# DB OLUSTURMA
# ============================================================================

def create_db_in_memory(image_folder, detector, embed_fn):
    image_root = Path(image_folder)
    if not image_root.exists():
        print(f"  [!] db klasoru bulunamadi: {image_root}")
        return np.array([]), np.array([])

    person_dirs = sorted([p for p in image_root.iterdir() if p.is_dir()])
    embeddings, names = [], []

    for pdir in person_dirs:
        per_embs = []
        for fname in sorted(os.listdir(pdir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            try:
                arr = np.fromfile(str(pdir / fname), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                continue
            if img is None:
                continue
            dets = detector.detect(img)
            best = FaceDetector.best_by_conf(dets)
            if best is None:
                continue
            roi = crop_with_padding(img, best.bbox)
            if roi is None:
                continue
            emb = embed_fn(roi)
            if emb is not None:
                per_embs.append(emb)

        if per_embs:
            mean_emb = l2_normalize(np.mean(np.stack(per_embs), axis=0))
            embeddings.append(mean_emb)
            names.append(pdir.name)

    return np.array(embeddings), np.array(names)


# ============================================================================
# VIDEO ISLEME
# ============================================================================

def process_video(video_path, detector, embed_fn, db_embs,
                  threshold, video_label, model_name, csv_rows):
    """
    Her tespit icin csv_rows'a bir satir ekler.
    Dondurur: (toplam_yuz, kabul_edilen, ort_ms)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"  [!] Video bulunamadi: {video_path}")
        return 0, 0, 0.0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Video acilamadi: {video_path}")
        return 0, 0, 0.0

    times, total, accepted = [], 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        for d in dets:
            roi = crop_with_padding(frame, d.bbox)
            if roi is None:
                continue
            t0 = time.time()
            emb = embed_fn(roi)
            elapsed_ms = (time.time() - t0) * 1000
            if emb is None:
                continue

            times.append(elapsed_ms)
            total += 1

            score = 0.0
            if db_embs.size > 0:
                sims  = np.dot(db_embs, emb.astype(np.float32))
                score = float(np.max(sims))

            acc = 1 if score >= threshold else 0
            accepted += acc

            csv_rows.append({
                "model":    model_name,
                "video":    video_label,
                "cos_score": round(score, 6),
                "accepted": acc,
                "infer_ms": round(elapsed_ms, 3),
            })

    cap.release()
    avg_ms = float(np.mean(times)) if times else 0.0
    return total, accepted, avg_ms


# ============================================================================
# MODEL ADAPTÖRLERI
# ============================================================================

def make_insightface_embedder(model_name: str):
    """buffalo_s, buffalo_l, antelopev2"""
    rec = FaceRecognizer(det_size=(160, 160), model_name=model_name)
    return rec.embed_from_roi


def make_sface_embedder():
    """
    OpenCV SFace (128d).
    Model dosyasi ilk calistirmada workspace_root/models/ altina indirilir.
    Manuel indirme:
      https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
    """
    model_path = workspace_root / "models" / "face_recognition_sface_2021dec.onnx"
    model_path.parent.mkdir(exist_ok=True)

    if not model_path.exists():
        url = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
               "face_recognition_sface/face_recognition_sface_2021dec.onnx")
        print(f"  SFace model indiriliyor: {url}")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(model_path))
            print("  SFace model indirildi.")
        except Exception as e:
            print(f"  [!] SFace indirme hatasi: {e}")
            return None

    try:
        sface = cv2.FaceRecognizerSF.create(str(model_path), "")
    except Exception as e:
        print(f"  [!] SFace yuklenemedi: {e}")
        return None

    def embed(roi_bgr):
        try:
            resized = cv2.resize(roi_bgr, (112, 112))
            feat = sface.feature(resized)
            return l2_normalize(feat.flatten().astype(np.float32))
        except Exception:
            return None

    return embed


def make_deepface_embedder(model_name: str):
    """
    GhostFaceNet ve Facenet512 icin ortak adaptör (deepface uzerinden).
    pip install deepface
    Modeller ilk calistirmada ~/.deepface/weights/ altina otomatik indirilir.
    """
    try:
        from deepface import DeepFace  # type: ignore
        # Modeli onceden bellegee yukle (ilk frame'de gecikme olmasin)
        print(f"  DeepFace modeli on-yuklemesi yapiliyor: {model_name} ...")
        try:
            from deepface.commons import functions  # type: ignore
            functions.initialize_detector(detector_backend="skip")
        except Exception:
            pass
    except ImportError:
        print("  [!] deepface kurulu degil. Kurmak icin: pip install deepface")
        return None

    def embed(roi_bgr):
        try:
            from deepface import DeepFace  # type: ignore
            rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            res = DeepFace.represent(
                rgb,
                model_name=model_name,
                enforce_detection=False,
                detector_backend="skip",
            )
            v = np.array(res[0]["embedding"], dtype=np.float32)
            return l2_normalize(v)
        except Exception:
            return None

    return embed


# ============================================================================
# GRAFIK
# ============================================================================

COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#00BCD4", "#F44336"]


def plot_results(results: dict, save_dir: Path, threshold: float):
    save_dir.mkdir(exist_ok=True)
    models   = list(results.keys())
    fps_vals = [results[m]["fps"]   for m in models]
    tar_vals = [results[m]["tar"]   for m in models]
    far_vals = [results[m]["far"]   for m in models]
    colors   = (COLORS * 5)[:len(models)]

    # FPS
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 6))
    bars = ax.bar(models, fps_vals, color=colors, edgecolor="white")
    ax.set_title("Embedding Cikarim Hizi Karsilastirmasi")
    ax.set_ylabel("FPS  (yuksek = daha iyi)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out = save_dir / "fps_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Kaydedildi: {out}")

    # TAR / FAR
    x, w = np.arange(len(models)), 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.8), 6))
    b1 = ax.bar(x - w/2, tar_vals, w, label="TAR – Dogru Kabul (%)",  color="#43A047")
    b2 = ax.bar(x + w/2, far_vals, w, label="FAR – Yanlis Kabul (%)", color="#E53935")
    ax.set_title(f"TAR / FAR Karsilastirmasi  (Esik = {threshold})")
    ax.set_ylabel("Yuzde (%)")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"%{h:.1f}", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"%{h:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out = save_dir / "accuracy_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Kaydedildi: {out}")


# ============================================================================
# CSV
# ============================================================================

def save_csv(rows: list, save_dir: Path):
    path = save_dir / "raw_scores.csv"
    fieldnames = ["model", "video", "cos_score", "accepted", "infer_ms"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Ham skorlar kaydedildi: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Yuz Tanima Modeli Benchmark")
    parser.add_argument("--known_video",   default="test-videos/tar1.h264")
    parser.add_argument("--unknown_video", default="test-videos/tar2.h264")
    parser.add_argument("--yolo_model",    default="yolo11n_filtered_int8.onnx")
    parser.add_argument("--db_path",       default="db-images")
    parser.add_argument("--threshold",     type=float, default=0.40)
    args = parser.parse_args()

    yolo_path = workspace_root / "yolo11-modes" / args.yolo_model
    if not yolo_path.exists():
        print(f"[!] YOLO modeli bulunamadi: {yolo_path}")
        return

    resolve = lambda p: workspace_root / p if not Path(p).is_absolute() else Path(p)
    known_path   = resolve(args.known_video)
    unknown_path = resolve(args.unknown_video)
    db_root      = resolve(args.db_path)

    print("=" * 65)
    print("  YUZ TANIMA MODELI BENCHMARK".center(65))
    print("=" * 65)
    print(f"  YOLO Model    : {yolo_path.name}")
    print(f"  Bilinen Video : {known_path.name}")
    print(f"  Yabanci Video : {unknown_path.name}")
    print(f"  DB Klasoru    : {db_root.name}")
    print(f"  Esik Degeri   : {args.threshold}")
    print("=" * 65)

    print("\nYOLO Detector yukleniyor...")
    detector = FaceDetector(
        model_path=str(yolo_path),
        img_size=640, pred_conf=0.5,
        iou=0.4, max_det=10, det_threshold=0.5
    )

    # --- Model katalogu ---
    # (gosterim_adi, factory_type, factory_arg)
    MODEL_REGISTRY = [
        ("buffalo_s",    "insightface", "buffalo_s"),
        ("buffalo_l",    "insightface", "buffalo_l"),
        ("antelopev2",   "insightface", "antelopev2"),
        ("sface",        "sface",       None),
        ("ghostfacenet", "deepface",    "GhostFaceNet"),
        ("facenet512",   "deepface",    "Facenet512"),
    ]

    results:  dict = {}
    csv_rows: list = []

    for display_name, factory_type, factory_arg in MODEL_REGISTRY:
        print(f"\n{'='*65}")
        print(f"  Model: {display_name.upper()}")
        print(f"{'='*65}")

        try:
            if factory_type == "insightface":
                embed_fn = make_insightface_embedder(factory_arg)
            elif factory_type == "sface":
                embed_fn = make_sface_embedder()
            elif factory_type == "deepface":
                embed_fn = make_deepface_embedder(factory_arg)
            else:
                embed_fn = None
        except Exception as e:
            print(f"  [!] Embedder olusturulamadi: {e}")
            embed_fn = None

        if embed_fn is None:
            print(f"  [!] '{display_name}' atlanıyor.")
            continue

        print("  -> DB embedding'leri cikariliyor...")
        db_embs, db_names = create_db_in_memory(str(db_root), detector, embed_fn)
        if db_embs.size == 0:
            print("  [!] Uyari: DB bos veya olusturulamadi.")

        print("  -> Bilinen video isleniyor (TAR & FPS)...")
        tot_k, acc_k, ms_k = process_video(
            known_path, detector, embed_fn, db_embs,
            args.threshold, "known", display_name, csv_rows
        )

        print("  -> Yabanci video isleniyor (FAR)...")
        tot_u, acc_u, ms_u = process_video(
            unknown_path, detector, embed_fn, db_embs,
            args.threshold, "unknown", display_name, csv_rows
        )

        tar    = (acc_k / tot_k * 100) if tot_k > 0 else 0.0
        far    = (acc_u / tot_u * 100) if tot_u > 0 else 0.0
        avg_ms = ((ms_k + ms_u) / 2 if ms_k > 0 and ms_u > 0
                  else max(ms_k, ms_u))
        fps    = 1000.0 / avg_ms if avg_ms > 0 else 0.0

        results[display_name] = {"fps": fps, "avg_ms": avg_ms, "tar": tar, "far": far}

        print(f"  Sonuc -> Bilinen: {tot_k} yuz / {acc_k} kabul  |  "
              f"Yabanci: {tot_u} yuz / {acc_u} kabul")

    # --- Ozet Tablo ---
    print("\n")
    print("#" * 70)
    print("  KARSILASTIRMA SONUCLARI OZETI".center(70))
    print("#" * 70)
    print(f"  {'MODEL':<15} | {'FPS':>8} | {'ms/yuz':>8} | {'TAR (%)':>9} | {'FAR (%)':>9}")
    print("  " + "-" * 66)
    for name, r in results.items():
        print(f"  {name.upper():<15} | {r['fps']:>8.1f} | {r['avg_ms']:>8.2f} |"
              f" {r['tar']:>9.2f} | {r['far']:>9.2f}")
    print("#" * 70)
    print(f"\n  Kullanilan Esik: {args.threshold}")
    print("  Not: TAR yuksek (>%90), FAR dusuk (<%%5) olmasi idealdir.")

    # --- Kaydet ---
    save_dir = refactored_root / "compares"
    print(f"\nHam skorlar CSV'ye yaziliyor...")
    save_csv(csv_rows, save_dir)

    if results:
        print("Grafikler ciziliyor...")
        plot_results(results, save_dir, args.threshold)

    print("\nTum islemler tamamlandi.")


if __name__ == "__main__":
    main()
