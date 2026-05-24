"""
Optimal Threshold Finder per Similarity Metric
===============================================
Reads metrics_raw.csv and computes, for each metric:
  - EER threshold        (where FAR == FNR)
  - Youden's J threshold (maximises TAR - FAR)
  - F1-score threshold   (maximises F1)
Then produces a combined TAR/FAR/F1 sweep plot with all 3 thresholds marked.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, f1_score

# ---------------------------------------------------------------------------
CSV_PATH = Path(__file__).parent / "metrics_raw.csv"
OUT_DIR  = Path(__file__).parent / "metric_comparison"
OUT_DIR.mkdir(exist_ok=True)

METRICS = ["cosine", "l2", "l1", "mahalanobis", "pearson"]
LABELS  = {
    "cosine":      "Cosine Similarity",
    "l2":          "Euclidean Distance (L2)",
    "l1":          "Manhattan Distance (L1)",
    "mahalanobis": "Mahalanobis Distance",
    "pearson":     "Pearson Correlation",
}
IS_SIM = {"cosine": True, "l2": False, "l1": False,
          "mahalanobis": False, "pearson": True}
PALETTE = {
    "cosine": "#4C8BF5", "l2": "#E84393", "l1": "#FF9800",
    "mahalanobis": "#9C27B0", "pearson": "#00C896",
}

# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------
if not CSV_PATH.exists():
    import sys
    print(f"[!] Ham veri dosyası bulunamadı: {CSV_PATH}")
    print("Lütfen önce metrikleri karşılaştıran betiği çalıştırın:")
    print("  python -m refactored_project.compares.compare_metrics")
    sys.exit(1)

rows = []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)

y_true = np.array([1 if r["video"] == "known" else 0 for r in rows])

# ---------------------------------------------------------------------------
# Compute best thresholds per metric
# ---------------------------------------------------------------------------
N_SWEEP = 500
results = {}

for m in METRICS:
    raw = np.array([float(r[m]) for r in rows])
    # For ROC: higher score = "more likely known"
    scores = raw if IS_SIM[m] else -raw

    fpr, tpr, roc_ths = roc_curve(y_true, scores)
    fnr = 1 - tpr

    # --- 1. EER threshold ---
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer_th_roc = float(roc_ths[eer_idx])
    eer_th = eer_th_roc if IS_SIM[m] else -eer_th_roc
    eer_tar = float(tpr[eer_idx]) * 100
    eer_far = float(fpr[eer_idx]) * 100

    # --- 2. Youden's J threshold (max TAR - FAR) ---
    j_idx = int(np.argmax(tpr - fpr))
    j_th_roc = float(roc_ths[j_idx])
    j_th = j_th_roc if IS_SIM[m] else -j_th_roc
    j_tar = float(tpr[j_idx]) * 100
    j_far = float(fpr[j_idx]) * 100

    # --- 3. F1-score threshold (sweep) ---
    lo = raw.min() * 0.95
    hi = raw.max() * 1.05
    sweep_ths = np.linspace(lo, hi, N_SWEEP)
    best_f1, best_f1_th, best_f1_tar, best_f1_far = -1, 0, 0, 0
    for th in sweep_ths:
        y_pred = (raw >= th).astype(int) if IS_SIM[m] else (raw <= th).astype(int)
        if y_pred.sum() == 0:
            continue
        f1  = f1_score(y_true, y_pred, zero_division=0)
        tar = np.mean((raw[y_true == 1] >= th) if IS_SIM[m] else (raw[y_true == 1] <= th)) * 100
        far = np.mean((raw[y_true == 0] >= th) if IS_SIM[m] else (raw[y_true == 0] <= th)) * 100
        if f1 > best_f1:
            best_f1, best_f1_th = f1, th
            best_f1_tar, best_f1_far = tar, far

    results[m] = {
        "eer_th":   eer_th,   "eer_tar":   eer_tar,   "eer_far":   eer_far,
        "j_th":     j_th,     "j_tar":     j_tar,     "j_far":     j_far,
        "f1_th":    best_f1_th,"f1_tar":   best_f1_tar,"f1_far":   best_f1_far,
        "f1_val":   best_f1,
        # sweep arrays for plot
        "sweep_ths": sweep_ths,
        "sweep_tar": np.array(
            [np.mean((raw[y_true==1]>=t) if IS_SIM[m] else (raw[y_true==1]<=t))*100
             for t in sweep_ths]),
        "sweep_far": np.array(
            [np.mean((raw[y_true==0]>=t) if IS_SIM[m] else (raw[y_true==0]<=t))*100
             for t in sweep_ths]),
    }

# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------
SEP = "=" * 90
print("\n" + SEP)
print("  OPTIMAL THRESHOLD ANALYSIS PER METRIC".center(90))
print(SEP)
print(f"  {'Metric':<26} | {'Method':<12} | {'Threshold':>12} | {'TAR (%)':>8} | {'FAR (%)':>8}")
print("  " + "-" * 86)
for m in METRICS:
    r = results[m]
    print(f"  {LABELS[m]:<26} | {'EER':12} | {r['eer_th']:12.4f} | {r['eer_tar']:8.2f} | {r['eer_far']:8.2f}")
    youden_label = "Youden's J"
    print(f"  {'':<26} | {youden_label:<12} | {r['j_th']:12.4f} | {r['j_tar']:8.2f} | {r['j_far']:8.2f}")
    print(f"  {'':<26} | {'Best F1':12} | {r['f1_th']:12.4f} | {r['f1_tar']:8.2f} | {r['f1_far']:8.2f}  (F1={r['f1_val']:.4f})")
    print("  " + "-" * 86)
print(SEP)
print("  EER    : threshold where FAR = FNR (theoretically balanced)")
print("  Youden : maximises TAR - FAR  (best separation, slightly aggressive)")
print("  Best F1: maximises F1-score   (balances precision and recall)")
print(SEP + "\n")

# ---------------------------------------------------------------------------
# Plot — one subplot per metric, 3 threshold lines
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(24, 5))
plt.rcParams.update({"font.size": 10})

for ax, m in zip(axes, METRICS):
    r   = results[m]
    ths = r["sweep_ths"]

    ax.plot(ths, r["sweep_tar"], color=PALETTE[m], lw=2.5, label="TAR")
    ax.plot(ths, r["sweep_far"], color="#E53935",   lw=2,   ls="--", label="FAR")

    ax.axvline(r["eer_th"], color="#607D8B", lw=1.5, ls=":",
               label=f"EER  th={r['eer_th']:.3f}")
    ax.axvline(r["j_th"],   color="#FF6F00", lw=1.5, ls="-.",
               label=f"Youden th={r['j_th']:.3f}")
    ax.axvline(r["f1_th"],  color="#2E7D32", lw=1.5, ls=(0,(5,2)),
               label=f"F1   th={r['f1_th']:.3f}")

    ax.set_title(LABELS[m], fontsize=10, fontweight="bold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate (%)" if m == "cosine" else "")
    ax.set_ylim(-2, 105)
    ax.legend(fontsize=7.5, loc="center right")
    ax.grid(alpha=0.3)

fig.suptitle("TAR & FAR vs Threshold — Optimal Points (EER / Youden's J / Best F1)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
out = OUT_DIR / "07_optimal_thresholds.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved: {out}")
