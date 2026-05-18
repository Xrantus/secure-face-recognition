"""
CSV'den Farkli Threshold Degerlerine Gore Grafik Olusturucu
===========================================================
Kullanim:
  python compares/plot_thresholds.py
  python compares/plot_thresholds.py --csv compares/raw_scores.csv
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ----------------------------------------------------------------------------
# ARGPARSE
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Threshold bazli grafik olusturucu")
    p.add_argument(
        "--csv",
        default=str(Path(__file__).parent / "raw_scores.csv"),
        help="raw_scores.csv dosyasinin yolu",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.55, 0.60, 0.65],
        help="Test edilecek threshold degerleri (ornek: 0.50 0.55 0.60 0.65)",
    )
    p.add_argument(
        "--out_dir",
        default=str(Path(__file__).parent),
        help="Grafiklerin kaydedilecegi klasor",
    )
    return p.parse_args()


# ----------------------------------------------------------------------------
# CSV OKUMA
# ----------------------------------------------------------------------------

def load_csv(csv_path: str) -> dict:
    """
    Dondurur:
        data[model][video] = [cos_score, ...]
        video : "known" veya "unknown"
    """
    data: dict = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model  = row["model"]
            video  = row["video"]
            score  = float(row["cos_score"])
            data[model][video].append(score)
    return data


# ----------------------------------------------------------------------------
# METRIK HESAPLAMA
# ----------------------------------------------------------------------------

def compute_metrics(data: dict, threshold: float) -> dict:
    """
    Her model icin verilen threshold'da TAR ve FAR hesapla.
    TAR = known videodaki kabul sayisi / toplam known tespiti
    FAR = unknown videodaki kabul sayisi / toplam unknown tespiti
    """
    metrics = {}
    for model, videos in data.items():
        known_scores   = videos.get("known",   [])
        unknown_scores = videos.get("unknown", [])

        tar = (sum(1 for s in known_scores   if s >= threshold) / len(known_scores)   * 100
               if known_scores else 0.0)
        far = (sum(1 for s in unknown_scores if s >= threshold) / len(unknown_scores) * 100
               if unknown_scores else 0.0)

        metrics[model] = {"tar": tar, "far": far}
    return metrics


# ----------------------------------------------------------------------------
# GRAFIK 1: Her threshold icin ayri TAR/FAR grouped bar chart
# ----------------------------------------------------------------------------

PALETTE = {
    "tar": "#43A047",   # yesil
    "far": "#E53935",   # kirmizi
}
COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#00BCD4", "#F44336", "#795548"]


def plot_per_threshold(data: dict, thresholds: list, out_dir: Path):
    """
    Her threshold icin ayri bir PNG dosyasi olusturur:
        tar_far_th050.png, tar_far_th055.png, ...
    """
    for th in thresholds:
        metrics = compute_metrics(data, th)
        models  = list(metrics.keys())
        tar_vals = [metrics[m]["tar"] for m in models]
        far_vals = [metrics[m]["far"] for m in models]

        x, w = np.arange(len(models)), 0.35
        fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.8), 6))

        b1 = ax.bar(x - w/2, tar_vals, w,
                    label="TAR – Dogru Kabul (%)", color=PALETTE["tar"], edgecolor="white")
        b2 = ax.bar(x + w/2, far_vals, w,
                    label="FAR – Yanlis Kabul (%)", color=PALETTE["far"], edgecolor="white")

        ax.set_title(f"TAR / FAR Karsilastirmasi  |  Esik = {th:.2f}",
                     fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel("Yuzde (%)", fontsize=11)
        ax.set_ylim(0, 115)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models], rotation=20, ha="right", fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.2,
                    f"%{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.2,
                    f"%{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.tight_layout()
        th_str  = f"{int(th*100):03d}"   # 0.50 -> "050"
        out_file = out_dir / f"tar_far_th{th_str}.png"
        plt.savefig(out_file, dpi=150)
        plt.close()
        print(f"  Kaydedildi: {out_file}")


# ----------------------------------------------------------------------------
# GRAFIK 2: TAR vs Threshold — tum modeller tek grafikte (cizgi)
# ----------------------------------------------------------------------------

def plot_tar_vs_threshold(data: dict, thresholds: list, out_dir: Path):
    models = list(data.keys())
    colors = (COLORS * 5)[:len(models)]

    fig, ax = plt.subplots(figsize=(9, 6))
    for model, color in zip(models, colors):
        tars = []
        for th in thresholds:
            scores = data[model].get("known", [])
            tar = (sum(1 for s in scores if s >= th) / len(scores) * 100
                   if scores else 0.0)
            tars.append(tar)
        ax.plot([t * 100 for t in thresholds], tars,
                marker="o", linewidth=2, markersize=7,
                label=model.upper(), color=color)
        # Son noktanin yanina deger yaz
        ax.annotate(f"%{tars[-1]:.1f}",
                    xy=(thresholds[-1]*100, tars[-1]),
                    xytext=(4, 2), textcoords="offset points",
                    fontsize=8, color=color)

    ax.set_title("TAR (Dogru Kabul Orani) vs Threshold", fontsize=13, fontweight="bold")
    ax.set_xlabel("Threshold (%)"); ax.set_ylabel("TAR (%)")
    ax.set_xticks([int(t*100) for t in thresholds])
    ax.set_xticklabels([f"{int(t*100)}" for t in thresholds])
    ax.grid(linestyle="--", alpha=0.45)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 110)

    plt.tight_layout()
    out_file = out_dir / "tar_vs_threshold.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Kaydedildi: {out_file}")


# ----------------------------------------------------------------------------
# GRAFIK 3: FAR vs Threshold — tum modeller tek grafikte (cizgi)
# ----------------------------------------------------------------------------

def plot_far_vs_threshold(data: dict, thresholds: list, out_dir: Path):
    models = list(data.keys())
    colors = (COLORS * 5)[:len(models)]

    fig, ax = plt.subplots(figsize=(9, 6))
    for model, color in zip(models, colors):
        fars = []
        for th in thresholds:
            scores = data[model].get("unknown", [])
            far = (sum(1 for s in scores if s >= th) / len(scores) * 100
                   if scores else 0.0)
            fars.append(far)
        ax.plot([t * 100 for t in thresholds], fars,
                marker="s", linewidth=2, markersize=7,
                label=model.upper(), color=color)

    ax.set_title("FAR (Yanlis Kabul Orani) vs Threshold", fontsize=13, fontweight="bold")
    ax.set_xlabel("Threshold (%)"); ax.set_ylabel("FAR (%)")
    ax.set_xticks([int(t*100) for t in thresholds])
    ax.set_xticklabels([f"{int(t*100)}" for t in thresholds])
    ax.grid(linestyle="--", alpha=0.45)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 50)

    plt.tight_layout()
    out_file = out_dir / "far_vs_threshold.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Kaydedildi: {out_file}")


# ----------------------------------------------------------------------------
# GRAFIK 4: Ozet tablo heatmap (modeller x thresholdlar, TAR degerleri)
# ----------------------------------------------------------------------------

def plot_heatmap(data: dict, thresholds: list, out_dir: Path):
    models   = list(data.keys())
    th_labels = [f"th={int(t*100)}" for t in thresholds]

    tar_matrix = np.zeros((len(models), len(thresholds)))
    far_matrix = np.zeros((len(models), len(thresholds)))

    for i, model in enumerate(models):
        for j, th in enumerate(thresholds):
            known_scores   = data[model].get("known",   [])
            unknown_scores = data[model].get("unknown", [])
            tar_matrix[i, j] = (sum(1 for s in known_scores   if s >= th) /
                                 max(1, len(known_scores)) * 100)
            far_matrix[i, j] = (sum(1 for s in unknown_scores if s >= th) /
                                 max(1, len(unknown_scores)) * 100)

    for matrix, title, fname, cmap in [
        (tar_matrix, "TAR (%) — Model x Threshold", "heatmap_tar.png", "Greens"),
        (far_matrix, "FAR (%) — Model x Threshold", "heatmap_far.png", "Reds"),
    ]:
        fig, ax = plt.subplots(figsize=(max(6, len(thresholds) * 1.5),
                                         max(4, len(models) * 0.8)))
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, label="%")

        ax.set_xticks(range(len(thresholds))); ax.set_xticklabels(th_labels)
        ax.set_yticks(range(len(models)));     ax.set_yticklabels([m.upper() for m in models])
        ax.set_title(title, fontsize=12, fontweight="bold")

        for i in range(len(models)):
            for j in range(len(thresholds)):
                ax.text(j, i, f"{matrix[i, j]:.1f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if matrix[i, j] > 60 else "black")

        plt.tight_layout()
        out_file = out_dir / fname
        plt.savefig(out_file, dpi=150)
        plt.close()
        print(f"  Kaydedildi: {out_file}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  THRESHOLD ANALIZI".center(60))
    print("=" * 60)
    print(f"  CSV Dosyasi   : {args.csv}")
    print(f"  Thresholds    : {[int(t*100) for t in args.thresholds]}")
    print(f"  Cikti Klasoru : {out_dir}")
    print("=" * 60)

    data = load_csv(args.csv)
    print(f"\n  Yuklenen modeller: {', '.join(data.keys())}")

    # Ozet tablo konsola bas
    print("\n")
    print("#" * 68)
    print("  THRESHOLD BAZLI TAR / FAR OZETI".center(68))
    print("#" * 68)

    col_w = 10
    header = f"  {'MODEL':<15}"
    for th in args.thresholds:
        header += f"  {'th='+str(int(th*100)):^{col_w*2+3}}"
    print(header)

    sub   = f"  {'':<15}"
    for _ in args.thresholds:
        sub += f"  {'TAR%':^{col_w}}  {'FAR%':^{col_w}}"
    print(sub)
    print("  " + "-" * 64)

    for model in data.keys():
        row = f"  {model.upper():<15}"
        for th in args.thresholds:
            known   = data[model].get("known",   [])
            unknown = data[model].get("unknown", [])
            tar = (sum(1 for s in known   if s >= th) / max(1,len(known))   * 100)
            far = (sum(1 for s in unknown if s >= th) / max(1,len(unknown)) * 100)
            row += f"  {tar:>{col_w}.2f}  {far:>{col_w}.2f}"
        print(row)

    print("#" * 68)

    # Grafikleri olustur
    print("\nGrafikler olusturuluyor...")
    plot_per_threshold(data, args.thresholds, out_dir)   # 4 ayri bar chart
    plot_tar_vs_threshold(data, args.thresholds, out_dir)  # cizgi: TAR vs th
    plot_far_vs_threshold(data, args.thresholds, out_dir)  # cizgi: FAR vs th
    plot_heatmap(data, args.thresholds, out_dir)            # 2 heatmap

    print("\nTum grafikler tamamlandi.")


if __name__ == "__main__":
    main()
