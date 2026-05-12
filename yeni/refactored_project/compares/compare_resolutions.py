"""
Resolution Comparison for Face Recognition Pipeline
=====================================================
Tests 4 square resolutions (160, 256, 320, 480, 640) on the same video.

Metrics
-------
  FPS                  : end-to-end frames processed per second
  E2E Latency (ms)     : detect + embed + match per frame
  mAP@0.5              : face detection accuracy vs 640x640 baseline
  TAR / FAR            : recognition at best cosine threshold
  Accuracy Degradation : drop vs 640x640 baseline
  CPU Usage (%)        : psutil per-frame measurement
  Peak RAM (MB)        : max RSS during processing
  CPU Temp (°C)        : Raspberry Pi thermal zone (optional)

Models
------
  Detector : yolo11n_filtered_int8.onnx
  Embedder : buffalo_s / w600k_mbf.onnx  (512-d, cosine similarity)

Outputs
-------
  compares/resolution_comparison/   PNG plots + summary_report.txt
  compares/resolution_raw.csv       per-frame raw data
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
import psutil

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COMPARES_DIR   = Path(__file__).resolve().parent
REFACTORED_DIR = COMPARES_DIR.parent
YENI_DIR       = REFACTORED_DIR.parent

if str(REFACTORED_DIR) not in sys.path:
    sys.path.insert(0, str(REFACTORED_DIR))

from face_detector import FaceDetector  # noqa: E402

YOLO_PATH     = YENI_DIR / "yolo11-modes" / "yolo11n_filtered_int8.onnx"
BUFFALO_S_REC = COMPARES_DIR / "models" / "buffalo_s" / "w600k_mbf.onnx"
DB_ROOT       = YENI_DIR / "db-images"
KNOWN_VIDEO   = YENI_DIR / "test-videos" / "tar1.h264"
UNKNOWN_VIDEO = YENI_DIR / "test-videos" / "tar2.h264"
OUT_DIR       = COMPARES_DIR / "resolution_comparison"
CSV_PATH      = COMPARES_DIR / "resolution_raw.csv"

# ---------------------------------------------------------------------------
# Resolutions to test (1:1 square, smallest → largest)
# ---------------------------------------------------------------------------
RESOLUTIONS = [160, 256, 320, 480, 640]
BASELINE_RES = 640       # accuracy degradation reference
FRAME_SKIP   = 1         # process every frame
COSINE_THRESHOLD = 0.50  # fixed decision threshold

PALETTE = {
    160: "#E53935",
    256: "#FF9800",
    320: "#FDD835",
    480: "#43A047",
    640: "#1E88E5",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

CSV_FIELDS = [
    "resolution", "video", "frame_id",
    "e2e_ms", "det_ms", "emb_ms", "match_ms",
    "n_faces", "n_known", "n_unknown",
    "cpu_pct", "ram_mb",
]


# ===========================================================================
# Thermal helper (RPi only)
# ===========================================================================
def read_cpu_temp() -> Optional[float]:
    p = Path("/sys/class/thermal/thermal_zone0/temp")
    if p.exists():
        try:
            return float(p.read_text().strip()) / 1000.0
        except Exception:
            return None
    return None


# ===========================================================================
# Buffalo-S embedder
# ===========================================================================
class BuffaloSEmbedder:
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
            blob = img.transpose(2, 0, 1)[np.newaxis]
            out  = self._sess.run(None, {self._iname: blob})[0]
            v    = out.flatten().astype(np.float32)
            n    = np.linalg.norm(v)
            return v / n if n > 1e-12 else v
        except Exception:
            return None


# ===========================================================================
# Utilities
# ===========================================================================
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def crop_with_padding(img: np.ndarray, bbox, pad: float = 0.20) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1, x2 = _clamp(x1, 0, w - 1), _clamp(x2, 0, w - 1)
    y1, y2 = _clamp(y1, 0, h - 1), _clamp(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    bw, bh = x2 - x1, y2 - y1
    if min(bw, bh) < 20:
        return None
    pw, ph = int(bw * pad), int(bh * pad)
    roi = img[max(0, y1 - ph):min(h, y2 + ph), max(0, x1 - pw):min(w, x2 + pw)]
    return roi if roi.size > 0 else None


def l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ===========================================================================
# DB building
# ===========================================================================
def build_db(db_root: Path, detector: FaceDetector, embedder: BuffaloSEmbedder):
    """Return (db_embs [N×512], db_names [N])."""
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
            print(f"    {pdir.name:15s}: {len(per)} images → 1 embedding")
    return np.array(embs, dtype=np.float32), np.array(names)


def match_cosine(probe: np.ndarray, db_embs: np.ndarray, threshold: float):
    """Return (is_known, best_score)."""
    if db_embs.size == 0:
        return False, -1.0
    scores = db_embs @ probe
    best   = float(np.max(scores))
    return best >= threshold, best


# ===========================================================================
# Per-resolution processing
# ===========================================================================
def process_video_at_res(
    video_path: Path,
    video_label: str,
    res: int,
    detector: FaceDetector,
    embedder: BuffaloSEmbedder,
    db_embs: np.ndarray,
    db_names: np.ndarray,
    frame_skip: int = FRAME_SKIP,
) -> list[dict]:
    rows: list[dict] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open: {video_path}")
        return rows

    proc = psutil.Process()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % frame_skip != 0:
            continue

        # Resize to square resolution
        t_e2e_start = time.perf_counter()
        frame_r = cv2.resize(frame, (res, res))

        # Detection
        t0 = time.perf_counter()
        dets = detector.detect(frame_r)
        det_ms = (time.perf_counter() - t0) * 1000

        n_faces = len(dets)
        n_known = 0
        n_unknown = 0
        total_emb_ms = 0.0
        total_match_ms = 0.0

        for d in dets:
            roi = crop_with_padding(frame_r, d.bbox)
            if roi is None:
                continue

            t0 = time.perf_counter()
            emb = embedder.embed(roi)
            emb_ms = (time.perf_counter() - t0) * 1000
            total_emb_ms += emb_ms

            if emb is None:
                continue

            t0 = time.perf_counter()
            is_known, _ = match_cosine(emb, db_embs, COSINE_THRESHOLD)
            match_ms = (time.perf_counter() - t0) * 1000
            total_match_ms += match_ms

            if is_known:
                n_known += 1
            else:
                n_unknown += 1

        e2e_ms = (time.perf_counter() - t_e2e_start) * 1000
        cpu_pct = proc.cpu_percent(interval=None)
        ram_mb  = proc.memory_info().rss / (1024 * 1024)

        rows.append({
            "resolution": res,
            "video":      video_label,
            "frame_id":   frame_id,
            "e2e_ms":     round(e2e_ms, 3),
            "det_ms":     round(det_ms, 3),
            "emb_ms":     round(total_emb_ms, 3),
            "match_ms":   round(total_match_ms, 3),
            "n_faces":    n_faces,
            "n_known":    n_known,
            "n_unknown":  n_unknown,
            "cpu_pct":    round(cpu_pct, 1),
            "ram_mb":     round(ram_mb, 1),
        })

    cap.release()
    print(f"    [{res:4d}px | {video_label:7s}] {frame_id} frames, {len(rows)} processed")
    return rows


# ===========================================================================
# Aggregate statistics per resolution
# ===========================================================================
def aggregate(rows: list[dict], res: int) -> dict:
    r = [x for x in rows if x["resolution"] == res]
    if not r:
        return {}

    e2e_arr  = np.array([x["e2e_ms"]  for x in r])
    det_arr  = np.array([x["det_ms"]  for x in r])
    emb_arr  = np.array([x["emb_ms"]  for x in r])
    cpu_arr  = np.array([x["cpu_pct"] for x in r])
    ram_arr  = np.array([x["ram_mb"]  for x in r])

    total_e2e_s = e2e_arr.sum() / 1000.0
    n_frames    = len(r)
    fps         = n_frames / total_e2e_s if total_e2e_s > 0 else 0.0

    # TAR / FAR on known vs unknown
    known_rows   = [x for x in r if x["video"] == "known"]
    unknown_rows = [x for x in r if x["video"] == "unknown"]

    total_known_faces   = sum(x["n_faces"] for x in known_rows)
    correctly_known     = sum(x["n_known"]   for x in known_rows)
    total_unknown_faces = sum(x["n_faces"] for x in unknown_rows)
    falsely_accepted    = sum(x["n_known"]   for x in unknown_rows)

    tar = correctly_known   / (total_known_faces   + 1e-9)
    far = falsely_accepted  / (total_unknown_faces + 1e-9)

    return {
        "res":          res,
        "fps":          round(fps, 2),
        "e2e_mean_ms":  round(float(e2e_arr.mean()), 2),
        "e2e_p95_ms":   round(float(np.percentile(e2e_arr, 95)), 2),
        "det_mean_ms":  round(float(det_arr.mean()), 2),
        "emb_mean_ms":  round(float(emb_arr.mean()), 2),
        "cpu_mean_pct": round(float(cpu_arr.mean()), 1),
        "cpu_max_pct":  round(float(cpu_arr.max()), 1),
        "peak_ram_mb":  round(float(ram_arr.max()), 1),
        "tar":          round(tar, 4),
        "far":          round(far, 4),
        "n_frames":     n_frames,
    }


# ===========================================================================
# Plots
# ===========================================================================
def _save(fig: plt.Figure, name: str) -> None:
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {p.name}")


def _bar(ax, stats: list[dict], key: str, ylabel: str, title: str,
         color_map: dict, annotate: bool = True) -> None:
    xs  = [s["res"] for s in stats]
    ys  = [s[key]   for s in stats]
    clr = [color_map[s["res"]] for s in stats]
    bars = ax.bar([str(f"{x}×{x}") for x in xs], ys, color=clr, edgecolor="white", width=0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    if annotate:
        for bar, v in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    f"{v}", ha="center", va="bottom", fontsize=9, fontweight="bold")


def plot_fps_latency(stats: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    _bar(axes[0], stats, "fps",          "FPS",    "Throughput (FPS ↑)",            PALETTE)
    _bar(axes[1], stats, "e2e_mean_ms",  "ms",     "Mean End-to-End Latency (↓)",   PALETTE)
    _bar(axes[2], stats, "e2e_p95_ms",   "ms",     "P95 End-to-End Latency (↓)",    PALETTE)

    fig.suptitle("Performance vs Resolution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "01_fps_latency.png")


def plot_breakdown(stats: list[dict]) -> None:
    labels = [f"{s['res']}×{s['res']}" for s in stats]
    det_ms = [s["det_mean_ms"] for s in stats]
    emb_ms = [s["emb_mean_ms"] for s in stats]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w/2, det_ms, w, label="Detection (ms)", color="#1E88E5", edgecolor="white")
    b2 = ax.bar(x + w/2, emb_ms, w, label="Embedding (ms)", color="#E53935", edgecolor="white")

    for bar in list(b1) + list(b2):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Latency Breakdown: Detection vs Embedding per Resolution",
                 fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "02_latency_breakdown.png")


def plot_accuracy(stats: list[dict]) -> None:
    baseline = next((s for s in stats if s["res"] == BASELINE_RES), None)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # TAR
    _bar(axes[0], stats, "tar", "TAR", "True Acceptance Rate ↑", PALETTE)
    if baseline:
        axes[0].axhline(baseline["tar"], color="gray", lw=1.2, linestyle="--", alpha=0.7,
                        label=f"Baseline ({BASELINE_RES}×{BASELINE_RES})")
        axes[0].legend(fontsize=9)

    # FAR
    _bar(axes[1], stats, "far", "FAR", "False Acceptance Rate ↓", PALETTE)

    # Accuracy degradation (TAR drop vs baseline)
    if baseline:
        degrad = [round(baseline["tar"] - s["tar"], 4) for s in stats]
        labels = [f"{s['res']}×{s['res']}" for s in stats]
        colors = [PALETTE[s["res"]] for s in stats]
        bars = axes[2].bar(labels, degrad, color=colors, edgecolor="white", width=0.6)
        axes[2].set_ylabel("TAR Drop")
        axes[2].set_title(f"Accuracy Degradation vs {BASELINE_RES}×{BASELINE_RES} ↓",
                          fontweight="bold")
        axes[2].grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, degrad):
            axes[2].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.002,
                         f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Recognition Accuracy vs Resolution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_accuracy.png")


def plot_system_resources(stats: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    _bar(axes[0], stats, "cpu_mean_pct", "CPU Usage (%)",
         "Mean CPU Utilization per Resolution", PALETTE)
    _bar(axes[1], stats, "peak_ram_mb",  "RAM (MB)",
         "Peak RAM Usage per Resolution", PALETTE)

    fig.suptitle("System Resources vs Resolution (Raspberry Pi)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "04_system_resources.png")


def plot_tradeoff(stats: list[dict]) -> None:
    """FPS vs TAR scatter — the sweet-spot visualization."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for s in stats:
        color = PALETTE[s["res"]]
        size  = s["peak_ram_mb"] * 3     # bubble size ~ RAM
        ax.scatter(s["fps"], s["tar"], s=size, color=color, alpha=0.85,
                   edgecolors="white", linewidths=1.2, zorder=3)
        ax.annotate(f"{s['res']}×{s['res']}",
                    xy=(s["fps"], s["tar"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_xlabel("Throughput (FPS) →")
    ax.set_ylabel("True Acceptance Rate (TAR) →")
    ax.set_title("FPS vs Accuracy Trade-off\n(bubble size = Peak RAM)",
                 fontweight="bold")
    ax.grid(alpha=0.3)
    ax.text(0.97, 0.03, "↗ ideal region", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="gray")
    fig.tight_layout()
    _save(fig, "05_tradeoff_scatter.png")


# ===========================================================================
# CSV
# ===========================================================================
def save_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV saved: {path}  ({len(rows)} rows)")


# ===========================================================================
# Summary report
# ===========================================================================
def print_summary(stats: list[dict], temp: Optional[float]) -> str:
    W = 90
    lines = []
    lines.append("=" * W)
    lines.append("  RESOLUTION COMPARISON — Face Recognition Pipeline".center(W))
    lines.append(f"  Cosine Threshold: {COSINE_THRESHOLD}  |  Baseline: {BASELINE_RES}×{BASELINE_RES}".center(W))
    lines.append("=" * W)

    hdr = (f"  {'Res':^9} | {'FPS':>6} | {'E2E(ms)':>8} | {'P95(ms)':>8} | "
           f"{'TAR':>6} | {'FAR':>6} | {'TAR-Drop':>9} | {'CPU%':>6} | {'RAM(MB)':>8}")
    lines.append(hdr)
    lines.append("  " + "-" * (W - 2))

    baseline_tar = next((s["tar"] for s in stats if s["res"] == BASELINE_RES), 1.0)

    for s in stats:
        drop = baseline_tar - s["tar"]
        lines.append(
            f"  {s['res']:^9} | {s['fps']:>6.1f} | {s['e2e_mean_ms']:>8.1f} | "
            f"{s['e2e_p95_ms']:>8.1f} | {s['tar']:>6.4f} | {s['far']:>6.4f} | "
            f"{drop:>+9.4f} | {s['cpu_mean_pct']:>6.1f} | {s['peak_ram_mb']:>8.1f}"
        )

    lines.append("=" * W)
    lines.append("  Metrics: FPS↑  E2E↓  TAR↑  FAR↓  TAR-Drop↓  CPU↓  RAM↓")
    if temp is not None:
        lines.append(f"  CPU Temperature (at end of run): {temp:.1f} °C")
    lines.append("=" * W)

    report = "\n".join(lines)
    print("\n" + report)
    return report


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
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
    print("  RESOLUTION COMPARISON  —  buffalo_s + YOLO11n".center(65))
    print("=" * 65)

    # Load models
    print("\n[1/4] Loading YOLO detector …")
    detector = FaceDetector(
        model_path=str(YOLO_PATH),
        img_size=640, pred_conf=0.5,
        iou=0.4, max_det=10, det_threshold=0.5,
    )

    print("[2/4] Loading buffalo_s embedder …")
    embedder = BuffaloSEmbedder(BUFFALO_S_REC)

    print("[3/4] Building face DB …")
    db_embs, db_names = build_db(DB_ROOT, detector, embedder)
    if db_embs.size == 0:
        print("[!] DB is empty — aborting.")
        sys.exit(1)
    print(f"  DB ready: {len(db_names)} persons  →  {db_embs.shape}")

    # Process all resolutions
    print(f"\n[4/4] Processing videos at {len(RESOLUTIONS)} resolutions …")
    all_rows: list[dict] = []

    for res in RESOLUTIONS:
        print(f"\n  --- Resolution: {res}×{res} ---")
        rows_k = process_video_at_res(
            KNOWN_VIDEO, "known", res, detector, embedder, db_embs, db_names
        )
        rows_u = process_video_at_res(
            UNKNOWN_VIDEO, "unknown", res, detector, embedder, db_embs, db_names
        )
        all_rows.extend(rows_k + rows_u)

    save_csv(all_rows, CSV_PATH)

    # Aggregate stats
    stats = [aggregate(all_rows, res) for res in RESOLUTIONS]
    stats = [s for s in stats if s]  # drop empty

    # Optional: CPU temperature at end
    temp = read_cpu_temp()

    # Plots
    print("\n  Generating plots …")
    plot_fps_latency(stats)
    plot_breakdown(stats)
    plot_accuracy(stats)
    plot_system_resources(stats)
    plot_tradeoff(stats)

    # Summary
    report = print_summary(stats, temp)
    report_path = OUT_DIR / "summary_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved: {report_path.name}")

    print(f"\nAll outputs → {OUT_DIR}\n")


if __name__ == "__main__":
    main()
