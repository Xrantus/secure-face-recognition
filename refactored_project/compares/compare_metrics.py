"""
Similarity Metric Comparison for Face Recognition
==================================================
Metrics  : Cosine | Euclidean (L2) | Manhattan (L1) | Mahalanobis | Pearson
Detector : yolo11n_filtered_int8.onnx
Embedder : buffalo_s  –  w600k_mbf.onnx  (512-d, L2-normalised)
Known    : tar1.h264   (enrolled persons: buket, kerem, meric, nedret)
Unknown  : tar2.h264   (persons NOT in DB)
DB       : db-images/  — one mean-embedding per person

Outputs
-------
  compares/metrics_raw.csv          raw per-face scores for every metric
  compares/metric_comparison/       PNG plots + summary report
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
from sklearn.metrics import roc_curve, auc as sk_auc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COMPARES_DIR   = Path(__file__).resolve().parent        # .../compares
REFACTORED_DIR = COMPARES_DIR.parent                    # .../refactored_project
YENI_DIR       = REFACTORED_DIR.parent                  # .../yeni

if str(REFACTORED_DIR) not in sys.path:
    sys.path.insert(0, str(REFACTORED_DIR))

from face_detector import FaceDetector  # noqa: E402 — needs sys.path patch above

YOLO_PATH     = YENI_DIR / "yolo11-modes" / "yolo11n_filtered_int8.onnx"
BUFFALO_S_REC = COMPARES_DIR / "models" / "buffalo_s" / "w600k_mbf.onnx"
DB_ROOT       = YENI_DIR / "db-images"
KNOWN_VIDEO   = YENI_DIR / "test-videos" / "tar1.h264"
UNKNOWN_VIDEO = YENI_DIR / "test-videos" / "tar2.h264"
OUT_DIR       = COMPARES_DIR / "metric_comparison"
CSV_PATH      = COMPARES_DIR / "metrics_raw.csv"

MAHAL_LAMBDA  = 0.1   # Ridge λ for covariance regularisation
FRAME_SKIP    = 1     # process every N-th frame  (1 = all frames)
N_SWEEP       = 200   # threshold sweep resolution

METRIC_NAMES  = ["cosine", "l2", "l1", "mahalanobis", "pearson"]
METRIC_LABELS = {
    "cosine":      "Cosine Similarity",
    "l2":          "Euclidean Distance (L2)",
    "l1":          "Manhattan Distance (L1)",
    "mahalanobis": "Mahalanobis Distance",
    "pearson":     "Pearson Correlation",
}
# True  → higher score means more similar  (decision: score >= threshold)
# False → lower  score means more similar  (decision: score <= threshold)
IS_SIMILARITY = {
    "cosine":      True,
    "l2":          False,
    "l1":          False,
    "mahalanobis": False,
    "pearson":     True,
}
PALETTE = {
    "cosine":      "#4C8BF5",
    "l2":          "#E84393",
    "l1":          "#FF9800",
    "mahalanobis": "#9C27B0",
    "pearson":     "#00C896",
}
CSV_FIELDS = ["frame_id", "video", "cosine", "l2", "l1",
              "mahalanobis", "pearson", "embed_ms"]

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ===========================================================================
# Buffalo-S embedder (direct ONNX — no full InsightFace stack needed)
# ===========================================================================
class BuffaloSEmbedder:
    """Wraps w600k_mbf.onnx for 112×112 ArcFace inference."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        self._sess  = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._iname = self._sess.get_inputs()[0].name

    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            img  = cv2.resize(face_bgr, (112, 112))
            img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img  = (img.astype(np.float32) - 127.5) / 128.0
            blob = img.transpose(2, 0, 1)[np.newaxis]           # NCHW
            out  = self._sess.run(None, {self._iname: blob})[0]
            v    = out.flatten().astype(np.float32)
            n    = np.linalg.norm(v)
            return v / n if n > 1e-12 else v
        except Exception:
            return None


# ===========================================================================
# Utility helpers
# ===========================================================================
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def crop_with_padding(img: np.ndarray, bbox, pad: float = 0.20) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1, x2 = _clamp(x1, 0, w-1), _clamp(x2, 0, w-1)
    y1, y2 = _clamp(y1, 0, h-1), _clamp(y2, 0, h-1)
    if x2 <= x1 or y2 <= y1:
        return None
    bw, bh = x2-x1, y2-y1
    if min(bw, bh) < 35:
        return None
    pw, ph = int(bw*pad), int(bh*pad)
    roi = img[max(0, y1-ph):min(h, y2+ph), max(0, x1-pw):min(w, x2+pw)]
    return roi if roi.size > 0 else None


def l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    return float(np.dot(a_c, b_c) / (denom + 1e-12))


# ===========================================================================
# DB building
# ===========================================================================
def build_db(db_root: Path, detector: FaceDetector,
             embedder: BuffaloSEmbedder):
    """Return (db_embs [N×512] float32, db_names [N] str)."""
    person_dirs = sorted(p for p in db_root.iterdir() if p.is_dir())
    embs, names = [], []
    for pdir in person_dirs:
        per = []
        for fname in sorted(os.listdir(pdir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            raw = np.fromfile(str(pdir / fname), np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is None:
                continue
            dets = detector.detect(img)
            best = FaceDetector.best_by_conf(dets)
            if best is None:
                continue
            roi = crop_with_padding(img, best.bbox)
            if roi is None:
                continue
            v = embedder.embed(roi)
            if v is not None:
                per.append(v)
        if per:
            mean_v = l2_norm(np.mean(np.stack(per), axis=0))
            embs.append(mean_v)
            names.append(pdir.name)
            print(f"    {pdir.name:15s}: {len(per)} images → 1 mean embedding")
    return np.array(embs, dtype=np.float32), np.array(names)


# ===========================================================================
# Mahalanobis precomputation
# ===========================================================================
def build_inv_cov(db_embs: np.ndarray, lam: float = MAHAL_LAMBDA) -> np.ndarray:
    """Ridge-regularised inverse covariance matrix from DB embeddings."""
    if db_embs.shape[0] < 2:
        return np.eye(db_embs.shape[1], dtype=np.float32)
    cov  = np.cov(db_embs.T).astype(np.float64)
    cov += lam * np.eye(cov.shape[0])
    inv  = np.linalg.pinv(cov)
    return inv.astype(np.float32)


def mah_dist(probe: np.ndarray, ref: np.ndarray,
             inv_cov: np.ndarray) -> float:
    d = (probe - ref).astype(np.float64)
    return float(np.sqrt(np.maximum(0.0, d @ inv_cov.astype(np.float64) @ d)))


# ===========================================================================
# Score computation  (all 5 metrics vs DB → best per metric)
# ===========================================================================
def compute_scores(probe: np.ndarray, db_embs: np.ndarray,
                   inv_cov: np.ndarray) -> dict:
    cosines  = db_embs @ probe                                    # (N,)
    diffs    = db_embs - probe                                    # (N,512)
    l2s      = np.linalg.norm(diffs, axis=1)
    l1s      = np.sum(np.abs(diffs), axis=1)
    mahs     = np.array([mah_dist(probe, ref, inv_cov)
                         for ref in db_embs])
    pearsons = np.array([pearson(probe, ref) for ref in db_embs])
    return {
        "cosine":      float(np.max(cosines)),
        "l2":          float(np.min(l2s)),
        "l1":          float(np.min(l1s)),
        "mahalanobis": float(np.min(mahs)),
        "pearson":     float(np.max(pearsons)),
    }


# ===========================================================================
# Video processing
# ===========================================================================
def process_video(video_path: Path, label: str,
                  detector: FaceDetector, embedder: BuffaloSEmbedder,
                  db_embs: np.ndarray, inv_cov: np.ndarray,
                  frame_skip: int = FRAME_SKIP) -> list[dict]:
    rows: list[dict] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open: {video_path}")
        return rows

    frame_id = 0
    face_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % frame_skip != 0:
            continue
        dets = detector.detect(frame)
        for d in dets:
            roi = crop_with_padding(frame, d.bbox)
            if roi is None:
                continue
            t0  = time.perf_counter()
            emb = embedder.embed(roi)
            ms  = (time.perf_counter() - t0) * 1000
            if emb is None:
                continue
            scores = compute_scores(emb, db_embs, inv_cov)
            row = {"frame_id": frame_id, "video": label, "embed_ms": round(ms, 3)}
            row.update({k: round(v, 6) for k, v in scores.items()})
            rows.append(row)
            face_count += 1

    cap.release()
    print(f"  [{label:7s}] {frame_id} frames processed  →  {face_count} face detections")
    return rows


# ===========================================================================
# CSV save
# ===========================================================================
def save_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV saved: {path}  ({len(rows)} rows)")


# ===========================================================================
# Analysis helpers
# ===========================================================================
def split(rows: list[dict], metric: str):
    k = np.array([r[metric] for r in rows if r["video"] == "known"])
    u = np.array([r[metric] for r in rows if r["video"] == "unknown"])
    return k, u


def compute_roc(rows: list[dict], metric: str):
    y_true = np.array([1 if r["video"] == "known" else 0 for r in rows])
    raw    = np.array([r[metric] for r in rows])
    scores = raw if IS_SIMILARITY[metric] else -raw
    fpr, tpr, ths = roc_curve(y_true, scores)
    auc_val = sk_auc(fpr, tpr)
    fnr = 1 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer     = float(fpr[eer_idx])
    eer_th  = float(ths[eer_idx])
    if not IS_SIMILARITY[metric]:
        eer_th = -eer_th
    return fpr, tpr, auc_val, eer, eer_th


def tar_far_sweep(rows: list[dict], metric: str, n: int = N_SWEEP):
    k, u = split(rows, metric)
    lo = min(k.min(), u.min()) * 0.95
    hi = max(k.max(), u.max()) * 1.05
    ths = np.linspace(lo, hi, n)
    tars, fars = [], []
    for th in ths:
        if IS_SIMILARITY[metric]:
            tars.append(np.mean(k >= th) * 100)
            fars.append(np.mean(u >= th) * 100)
        else:
            tars.append(np.mean(k <= th) * 100)
            fars.append(np.mean(u <= th) * 100)
    return ths, np.array(tars), np.array(fars)


def fisher_ratio(k: np.ndarray, u: np.ndarray) -> float:
    within = (k.var() + u.var()) / 2 + 1e-12
    return float((k.mean() - u.mean()) ** 2 / within)


# ===========================================================================
# Plots
# ===========================================================================
def _save(fig: plt.Figure, name: str) -> None:
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {p.name}")


def plot_roc_curves(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in METRIC_NAMES:
        fpr, tpr, auc_val, eer, _ = compute_roc(rows, m)
        ax.plot(fpr, tpr, color=PALETTE[m], lw=2,
                label=f"{METRIC_LABELS[m]}  (AUC={auc_val:.3f}, EER={eer:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (TAR)")
    ax.set_title("ROC Curves — Similarity Metric Comparison")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, "01_roc_curves.png")


def plot_score_distributions(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, m in zip(axes, METRIC_NAMES):
        k, u = split(rows, m)
        data   = [k, u]
        vp = ax.violinplot(data, positions=[0, 1], showmedians=True,
                           widths=0.7)
        for body in vp["bodies"]:
            body.set_facecolor(PALETTE[m])
            body.set_alpha(0.55)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            vp[part].set_color(PALETTE[m])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Known", "Unknown"], fontsize=10)
        ax.set_title(METRIC_LABELS[m], fontsize=10, fontweight="bold")
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Score Distributions — Known vs Unknown", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "02_score_distributions.png")


def plot_histograms(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, m in zip(axes, METRIC_NAMES):
        k, u = split(rows, m)
        bins = np.linspace(min(k.min(), u.min()), max(k.max(), u.max()), 50)
        ax.hist(k, bins=bins, color=PALETTE[m], alpha=0.6, label="Known",   density=True)
        ax.hist(u, bins=bins, color="#888888", alpha=0.5, label="Unknown", density=True)
        ax.set_title(METRIC_LABELS[m], fontsize=10, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Score Histograms — Known vs Unknown (density)", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "03_score_histograms.png")


def plot_threshold_sweep(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=False)
    for ax, m in zip(axes, METRIC_NAMES):
        ths, tars, fars = tar_far_sweep(rows, m)
        ax.plot(ths, tars, color=PALETTE[m], lw=2, label="TAR")
        ax.plot(ths, fars, color="#E53935",   lw=2, linestyle="--", label="FAR")
        # Mark EER-equivalent: where TAR+FAR ≈ 100 and they cross
        cross = np.argmin(np.abs(tars - (100 - fars)))
        ax.axvline(ths[cross], color="gray", lw=1, linestyle=":")
        ax.set_title(METRIC_LABELS[m], fontsize=10, fontweight="bold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Rate (%)")
        ax.set_ylim(-2, 105)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("TAR & FAR vs Threshold — per Metric", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "04_threshold_sweep.png")


def plot_summary_bars(stats: dict) -> None:
    """stats[metric] = {auc, eer, fisher}"""
    labels = [METRIC_LABELS[m] for m in METRIC_NAMES]
    colors = [PALETTE[m] for m in METRIC_NAMES]
    x = np.arange(len(METRIC_NAMES))
    w = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, key, title, ylabel, better in zip(
        axes,
        ["auc",   "eer",   "fisher"],
        ["AUC",   "EER",   "Fisher Discriminant Ratio"],
        ["AUC ↑", "EER ↓ (lower is better)", "FDR ↑ (higher is better)"],
        ["high",  "low",   "high"],
    ):
        vals = [stats[m][key] for m in METRIC_NAMES]
        bars = ax.bar(x, vals, color=colors, edgecolor="white", width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 * ax.get_ylim()[1],
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold")

    fig.suptitle("Metric Comparison Summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_summary_bars.png")


def plot_tar_far_at_thresholds(rows: list[dict]) -> None:
    """Heat-map style: TAR and FAR for each metric at several fixed thresholds."""
    # Compute relative thresholds per metric (0%, 25%, 50%, 75%, 100% of range)
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for col, m in enumerate(METRIC_NAMES):
        ths, tars, fars = tar_far_sweep(rows, m)
        ax_tar = axes[0][col]
        ax_far = axes[1][col]
        ax_tar.plot(ths, tars, color=PALETTE[m], lw=2)
        ax_tar.fill_between(ths, tars, alpha=0.15, color=PALETTE[m])
        ax_tar.set_title(METRIC_LABELS[m], fontsize=9, fontweight="bold")
        ax_tar.set_ylabel("TAR (%)" if col == 0 else "")
        ax_tar.set_ylim(-2, 105)
        ax_tar.grid(alpha=0.3)

        ax_far.plot(ths, fars, color="#E53935", lw=2)
        ax_far.fill_between(ths, fars, alpha=0.15, color="#E53935")
        ax_far.set_xlabel("Threshold")
        ax_far.set_ylabel("FAR (%)" if col == 0 else "")
        ax_far.set_ylim(-2, 105)
        ax_far.grid(alpha=0.3)

    axes[0][0].annotate("TAR vs Threshold", xy=(0, 0.5),
                        xycoords="axes fraction", rotation=90,
                        va="center", ha="right", fontsize=10)
    axes[1][0].annotate("FAR vs Threshold", xy=(0, 0.5),
                        xycoords="axes fraction", rotation=90,
                        va="center", ha="right", fontsize=10)
    fig.suptitle("TAR and FAR Across Threshold Range — All Metrics",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_tar_far_grid.png")


# ===========================================================================
# Console summary
# ===========================================================================
def print_summary(stats: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  SIMILARITY METRIC COMPARISON — SUMMARY".center(70))
    lines.append("=" * 70)
    header = f"  {'Metric':<22} | {'AUC':>6} | {'EER':>6} | {'EER-Threshold':>14} | {'FDR':>7}"
    lines.append(header)
    lines.append("  " + "-" * 66)
    for m in METRIC_NAMES:
        s = stats[m]
        lines.append(
            f"  {METRIC_LABELS[m]:<22} | {s['auc']:>6.4f} | {s['eer']:>6.4f} |"
            f" {s['eer_th']:>14.4f} | {s['fisher']:>7.3f}"
        )
    lines.append("=" * 70)
    lines.append("  AUC↑  EER↓  FDR↑  (higher AUC/FDR and lower EER = better metric)")
    lines.append("  Note: Mahalanobis uses Ridge-regularised covariance (λ=0.1)")
    lines.append("=" * 70)
    report = "\n".join(lines)
    print("\n" + report)
    return report


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    # --- Sanity checks ---
    for path, label in [
        (YOLO_PATH,     "YOLO model"),
        (BUFFALO_S_REC, "buffalo_s recognition model"),
        (DB_ROOT,       "DB images folder"),
        (KNOWN_VIDEO,   "known video"),
        (UNKNOWN_VIDEO, "unknown video"),
    ]:
        if not Path(path).exists():
            print(f"[!] {label} not found: {path}")
            sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  SIMILARITY METRIC COMPARISON  —  buffalo_s + YOLO11n".center(65))
    print("=" * 65)

    # --- Load detector ---
    print("\n[1/5] Loading YOLO detector …")
    detector = FaceDetector(
        model_path=str(YOLO_PATH),
        img_size=640, pred_conf=0.5,
        iou=0.4, max_det=10, det_threshold=0.5,
    )

    # --- Load embedder ---
    print("[2/5] Loading buffalo_s embedder …")
    embedder = BuffaloSEmbedder(BUFFALO_S_REC)

    # --- Build DB ---
    print("[3/5] Building DB …")
    db_embs, db_names = build_db(DB_ROOT, detector, embedder)
    if db_embs.size == 0:
        print("[!] DB is empty — aborting.")
        sys.exit(1)
    print(f"  DB ready: {len(db_names)} persons  →  {db_embs.shape}")

    # --- Mahalanobis covariance ---
    print("  Computing ridge-regularised covariance (λ=0.1) …")
    inv_cov = build_inv_cov(db_embs, MAHAL_LAMBDA)

    # --- Process videos ---
    print(f"\n[4/5] Processing videos (frame_skip={FRAME_SKIP}) …")
    rows_k = process_video(KNOWN_VIDEO,   "known",   detector, embedder, db_embs, inv_cov)
    rows_u = process_video(UNKNOWN_VIDEO, "unknown", detector, embedder, db_embs, inv_cov)
    all_rows = rows_k + rows_u

    if len(all_rows) == 0:
        print("[!] No faces detected — aborting.")
        sys.exit(1)

    save_csv(all_rows, CSV_PATH)

    # --- Analysis & plots ---
    print(f"\n[5/5] Computing stats & generating plots …")

    stats: dict = {}
    for m in METRIC_NAMES:
        fpr, tpr, auc_val, eer, eer_th = compute_roc(all_rows, m)
        k, u = split(all_rows, m)
        fdr  = fisher_ratio(k, u)
        stats[m] = {"auc": auc_val, "eer": eer, "eer_th": eer_th, "fisher": fdr}

    plot_roc_curves(all_rows)
    plot_score_distributions(all_rows)
    plot_histograms(all_rows)
    plot_threshold_sweep(all_rows)
    plot_summary_bars(stats)
    plot_tar_far_at_thresholds(all_rows)

    report = print_summary(stats)
    report_path = OUT_DIR / "summary_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved: {report_path.name}")

    print(f"\nAll outputs → {OUT_DIR}\n")


if __name__ == "__main__":
    main()
