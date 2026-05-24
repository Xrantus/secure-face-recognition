"""
Model Comparison for Face Recognition Pipeline
================================================
Compares buffalo_s, buffalo_l, antelopev2, and OpenCV SFace.

Metrics
-------
  FPS                  : frames processed per second
  Latency (ms)         : embed extraction latency per face
  TAR / FAR            : true acceptance and false acceptance rates
  TAR/FAR Heatmaps     : heatmaps sweeping key thresholds per model

Outputs
-------
  compares/model_comparison/   PNG plots + summary_report.txt
  compares/model_raw.csv       per-face raw scoring
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths and Imports Setup
# ---------------------------------------------------------------------------
COMPARES_DIR   = Path(__file__).resolve().parent
REFACTORED_DIR = COMPARES_DIR.parent
WORKSPACE_ROOT = REFACTORED_DIR.parent

if str(REFACTORED_DIR) not in sys.path:
    sys.path.insert(0, str(REFACTORED_DIR))

from face_detector import FaceDetector
from face_recognizer import FaceRecognizer

DB_ROOT       = WORKSPACE_ROOT / "db-images"
KNOWN_VIDEO   = WORKSPACE_ROOT / "test-videos" / "tar1.h264"
UNKNOWN_VIDEO = WORKSPACE_ROOT / "test-videos" / "tar2.h264"
OUT_DIR       = COMPARES_DIR / "model_comparison"
CSV_PATH      = COMPARES_DIR / "model_raw.csv"

# Sweep parameters
FRAME_SKIP   = 1
SWEEP_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

PALETTE = {
    "buffalo_s":    "#1E88E5",
    "buffalo_l":    "#9C27B0",
    "antelopev2":   "#43A047",
    "sface":        "#FF9800",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ===========================================================================
# Model Adaptors
# ===========================================================================
def make_insightface_embedder(model_name: str) -> FaceRecognizer:
    return FaceRecognizer(det_size=(160, 160), model_name=model_name)


def make_sface_embedder() -> Optional[object]:
    """OpenCV SFace (128d) model loader and wrapper."""
    model_path = WORKSPACE_ROOT / "models" / "face_recognition_sface_2021dec.onnx"
    model_path.parent.mkdir(exist_ok=True)

    if not model_path.exists():
        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        print(f"  Downloading SFace model: {url}")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(model_path))
            print("  SFace model downloaded.")
        except Exception as e:
            print(f"  [!] SFace download error: {e}")
            return None

    try:
        sface = cv2.FaceRecognizerSF.create(str(model_path), "")
        return sface
    except Exception as e:
        print(f"  [!] SFace load error: {e}")
        return None


# ===========================================================================
# Utilities
# ===========================================================================
def crop_with_padding(frame: np.ndarray, bbox, pad: float = 0.20) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    bw, bh = x2 - x1, y2 - y1
    if min(bw, bh) < 35:
        return None
    pw, ph = int(bw * pad), int(bh * pad)
    roi = frame[max(0, y1 - ph):min(h, y2 + ph), max(0, x1 - pw):min(w, x2 + pw)]
    return roi if roi.size > 0 else None


def l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ===========================================================================
# Database Generation in Memory per Model
# ===========================================================================
def build_db_in_memory(
    detector: FaceDetector,
    embedder: object,
    is_sface: bool,
) -> tuple[np.ndarray, np.ndarray]:
    person_dirs = sorted(p for p in DB_ROOT.iterdir() if p.is_dir())
    embs, names = [], []

    for pdir in person_dirs:
        per = []
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

            if is_sface:
                try:
                    resized = cv2.resize(roi, (112, 112))
                    feat = embedder.feature(resized)
                    v = l2_norm(feat.flatten().astype(np.float32))
                    per.append(v)
                except Exception:
                    pass
            else:
                # InsightFace
                v = embedder.embed_from_roi(roi)
                if v is not None:
                    per.append(v)

        if per:
            mean_v = l2_norm(np.mean(np.stack(per), axis=0))
            embs.append(mean_v)
            names.append(pdir.name)
            print(f"    {pdir.name:15s}: {len(per)} images -> 1 mean embedding")

    return np.array(embs, dtype=np.float32), np.array(names)


# ===========================================================================
# Video processing loop per Model
# ===========================================================================
def process_video(
    video_path: Path,
    video_label: str,
    detector: FaceDetector,
    embedder: object,
    is_sface: bool,
    db_embs: np.ndarray,
    model_name: str,
    csv_rows: list[dict],
) -> tuple[int, int, float, list[float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open: {video_path}")
        return 0, 0, 0.0, []

    times = []
    scores = []
    frame_id = 0
    face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        dets = detector.detect(frame)
        for d in dets:
            roi = crop_with_padding(frame, d.bbox)
            if roi is None:
                continue

            t0 = time.perf_counter()
            if is_sface:
                try:
                    resized = cv2.resize(roi, (112, 112))
                    feat = embedder.feature(resized)
                    emb = l2_norm(feat.flatten().astype(np.float32))
                except Exception:
                    emb = None
            else:
                emb = embedder.embed_from_roi(roi)
            ms = (time.perf_counter() - t0) * 1000

            if emb is None:
                continue

            times.append(ms)
            face_count += 1

            best_score = 0.0
            if db_embs.size > 0:
                sims = db_embs @ emb
                best_score = float(np.max(sims))

            scores.append(best_score)
            csv_rows.append({
                "model":      model_name,
                "video":      video_label,
                "frame_id":   frame_id,
                "score":      round(best_score, 6),
                "latency_ms": round(ms, 3),
            })

    cap.release()
    avg_ms = float(np.mean(times)) if times else 0.0
    return frame_id, face_count, avg_ms, scores


# ===========================================================================
# Visualizations
# ===========================================================================
def _save(fig: plt.Figure, name: str) -> None:
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {p.name}")


def plot_fps_latency(stats: dict) -> None:
    models = list(stats.keys())
    fps_vals = [stats[m]["fps"] for m in models]
    lat_vals = [stats[m]["avg_ms"] for m in models]
    colors = [PALETTE[m] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # FPS Bar Plot
    bars = axes[0].bar(models, fps_vals, color=colors, edgecolor="white", width=0.5)
    axes[0].set_ylabel("FPS (Higher is better) ↑")
    axes[0].set_title("Model Throughput (FPS)", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{bar.get_height():.1f}", ha="center", va="bottom", fontweight="bold")

    # Latency Bar Plot
    bars2 = axes[1].bar(models, lat_vals, color=colors, edgecolor="white", width=0.5)
    axes[1].set_ylabel("Latency (ms per face) ↓")
    axes[1].set_title("Embedding Extraction Latency (ms)", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{bar.get_height():.1f} ms", ha="center", va="bottom", fontweight="bold")

    fig.suptitle("Performance Comparison between Face Recognition Models", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "01_fps_latency.png")


def plot_accuracy(stats: dict, default_threshold: float = 0.50) -> None:
    models = list(stats.keys())
    tar_vals = [stats[m]["tar"] * 100 for m in models]
    far_vals = [stats[m]["far"] * 100 for m in models]

    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, tar_vals, w, label="TAR – Correctly Recognized (%)", color="#43A047", edgecolor="white")
    b2 = ax.bar(x + w/2, far_vals, w, label="FAR – Falsely Recognized (%)", color="#E53935", edgecolor="white")

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"Accuracy Comparison TAR / FAR (Threshold = {default_threshold})", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(-2, 110)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "02_accuracy_comparison.png")


def plot_model_heatmaps(stats: dict) -> None:
    """TAR and FAR Heatmaps across key thresholds for each face recognition model."""
    models = list(stats.keys())
    
    tar_matrix = np.zeros((len(models), len(SWEEP_THRESHOLDS)))
    far_matrix = np.zeros((len(models), len(SWEEP_THRESHOLDS)))

    for i, m in enumerate(models):
        k_scores = np.array(stats[m]["known_scores"])
        u_scores = np.array(stats[m]["unknown_scores"])

        for j, th in enumerate(SWEEP_THRESHOLDS):
            tar = np.mean(k_scores >= th) * 100 if len(k_scores) > 0 else 0.0
            far = np.mean(u_scores >= th) * 100 if len(u_scores) > 0 else 0.0
            tar_matrix[i, j] = tar
            far_matrix[i, j] = far

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. TAR Heatmap
    ax = axes[0]
    ax.imshow(tar_matrix, cmap="Greens", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(SWEEP_THRESHOLDS)))
    ax.set_xticklabels([f"Th={t:.1f}" for t in SWEEP_THRESHOLDS])
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("True Acceptance Rate (TAR %) Heatmap ↑", fontweight="bold")

    for i, m in enumerate(models):
        for j in range(len(SWEEP_THRESHOLDS)):
            val = tar_matrix[i, j]
            text = f"{val:.1f}%"
            color = "white" if val > 65 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    # 2. FAR Heatmap
    ax = axes[1]
    ax.imshow(far_matrix, cmap="Reds", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(SWEEP_THRESHOLDS)))
    ax.set_xticklabels([f"Th={t:.1f}" for t in SWEEP_THRESHOLDS])
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("False Acceptance Rate (FAR %) Heatmap ↓", fontweight="bold")

    for i, m in enumerate(models):
        for j in range(len(SWEEP_THRESHOLDS)):
            val = far_matrix[i, j]
            text = f"{val:.1f}%"
            color = "white" if val > 65 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    fig.suptitle("TAR and FAR across Face Recognition Models & Cosine Thresholds", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "03_tar_far_heatmaps.png")


# ===========================================================================
# Report
# ===========================================================================
def print_summary(stats: dict, default_threshold: float) -> str:
    W = 85
    lines = []
    lines.append("=" * W)
    lines.append("  MODEL COMPARISON — Face Recognition Pipeline".center(W))
    lines.append(f"  Decision Threshold: {default_threshold}".center(W))
    lines.append("=" * W)

    hdr = f"  {'Model':<15} | {'FPS':>8} | {'Latency(ms)':>12} | {'TAR (%)':>10} | {'FAR (%)':>10}"
    lines.append(hdr)
    lines.append("  " + "-" * (W - 2))

    for name, r in stats.items():
        lines.append(
            f"  {name:<15} | {r['fps']:>8.1f} | {r['avg_ms']:>12.2f} | "
            f"{r['tar']*100:>9.2f}% | {r['far']*100:>9.2f}%"
        )

    lines.append("=" * W)
    lines.append("  Metrics: FPS↑  Latency↓  TAR↑  FAR↓")
    lines.append("=" * W)

    report = "\n".join(lines)
    print("\n" + report)
    return report


# ===========================================================================
# Main Runner
# ===========================================================================
def main() -> None:
    for path, label in [
        (KNOWN_VIDEO,   "known video"),
        (UNKNOWN_VIDEO, "unknown video"),
    ]:
        if not Path(path).exists():
            print(f"[!] {label} not found: {path}")
            sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  MODEL COMPARISON  —  Benchmarking Face Models".center(65))
    print("=" * 65)

    # Load YOLO Detector
    print("\n[1/4] Loading modular YOLO detector …")
    detector = FaceDetector(
        model_path=str(WORKSPACE_ROOT / "yolo11-modes" / "yolo11n_filtered_int8.onnx"),
        img_size=640, pred_conf=0.5,
        iou=0.4, max_det=10, det_threshold=0.5,
    )

    models_to_test = ["buffalo_s", "buffalo_l", "antelopev2", "sface"]
    stats: dict = {}
    csv_rows: list[dict] = []

    for model_name in models_to_test:
        print(f"\n[2/4] Testing Model: {model_name.upper()} …")
        is_sface = (model_name == "sface")

        # Load appropriate embedder
        if is_sface:
            embedder = make_sface_embedder()
        else:
            embedder = make_insightface_embedder(model_name)

        if embedder is None:
            print(f"  [!] Skipping {model_name} due to initialization error.")
            continue

        # Build DB in-memory for this specific model configuration
        print(f"  Building in-memory DB for {model_name} …")
        db_embs, db_names = build_db_in_memory(detector, embedder, is_sface)
        if db_embs.size == 0:
            print(f"  [!] DB is empty for {model_name} — skipping.")
            continue

        print(f"  Processing Known Video …")
        frames_k, faces_k, ms_k, scores_k = process_video(
            KNOWN_VIDEO, "known", detector, embedder, is_sface, db_embs, model_name, csv_rows
        )

        print(f"  Processing Unknown Video …")
        frames_u, faces_u, ms_u, scores_u = process_video(
            UNKNOWN_VIDEO, "unknown", detector, embedder, is_sface, db_embs, model_name, csv_rows
        )

        # Performance aggregation
        avg_ms = (ms_k + ms_u) / 2 if ms_k > 0 and ms_u > 0 else max(ms_k, ms_u)
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

        # TAR/FAR at default threshold 0.50
        tar = np.mean(np.array(scores_k) >= 0.50) if scores_k else 0.0
        far = np.mean(np.array(scores_u) >= 0.50) if scores_u else 0.0

        stats[model_name] = {
            "fps":            fps,
            "avg_ms":         avg_ms,
            "tar":            tar,
            "far":            far,
            "known_scores":   scores_k,
            "unknown_scores": scores_u,
        }

    # Save CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "video", "frame_id", "score", "latency_ms"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[3/4] CSV saved: {CSV_PATH}  ({len(csv_rows)} rows)")

    # Visualizations
    print("[4/4] Generating comparison plots …")
    plot_fps_latency(stats)
    plot_accuracy(stats, 0.50)
    plot_model_heatmaps(stats)

    report = print_summary(stats, 0.50)
    report_path = OUT_DIR / "summary_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved: {report_path.name}")
    print(f"\nAll outputs successfully saved to -> {OUT_DIR}\n")


if __name__ == "__main__":
    main()
