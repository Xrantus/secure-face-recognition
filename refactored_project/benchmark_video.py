"""
Benchmark Script for YOLO Models
Calculates Precision, Recall, F1-Score and AP@0.5 using Pseudo-Labeling.
Normal Model (Ground Truth) vs INT8 Model (Predictions).
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def calculate_ap(recalls, precisions):
    """
    Calculate Average Precision (AP) using the all-point interpolation method.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def run_benchmark(video_path: str, teacher_model_path: str, student_model_path: str, iou_threshold: float = 0.5):
    print(f"\n--- Benchmark Baslatiliyor ---")
    print(f"Video: {video_path}")
    print(f"Referans Model (Normal): {teacher_model_path}")
    print(f"Test Modeli (INT8): {student_model_path}")
    print("Modeller yukleniyor...")

    teacher_model = YOLO(teacher_model_path, task='detect')
    student_model = YOLO(student_model_path, task='detect')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"HATA: Video acilamadi -> {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Toplam Kare (Frame) Sayisi: {frame_count}")

    # Istatistikler
    all_preds = [] # list of dicts: {'conf': float, 'matched': bool}
    total_truths = 0

    fps_list_teacher = []
    fps_list_student = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Isleniyor: {frame_idx} / {frame_count} kare tamamlandi.")

        # 1. Teacher (Referans) Model ile Tahmin (Ground Truth olarak kabul edilecek)
        t0 = time.time()
        res_teacher = teacher_model.predict(frame, imgsz=640, verbose=False, conf=0.5)[0]
        t1 = time.time()
        fps_list_teacher.append(1 / (t1 - t0 + 1e-6))

        truths = []
        if res_teacher.boxes is not None:
            for box in res_teacher.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                truths.append({"bbox": (x1, y1, x2, y2), "matched": False})
        
        total_truths += len(truths)

        # 2. Student (INT8) Model ile Tahmin
        t0 = time.time()
        res_student = student_model.predict(frame, imgsz=640, verbose=False, conf=0.01)[0]
        t1 = time.time()
        fps_list_student.append(1 / (t1 - t0 + 1e-6))

        preds = []
        if res_student.boxes is not None:
            for box in res_student.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                preds.append({"bbox": (x1, y1, x2, y2), "conf": conf, "matched": False})

        # Preds'leri confidence'a gore sirala
        preds = sorted(preds, key=lambda x: x["conf"], reverse=True)

        # 3. IoU Eslestirmesi
        for p in preds:
            best_iou = 0
            best_truth_idx = -1
            
            for i, t in enumerate(truths):
                if not t["matched"]:
                    iou = calculate_iou(p["bbox"], t["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_truth_idx = i
            
            if best_iou >= iou_threshold:
                truths[best_truth_idx]["matched"] = True
                p["matched"] = True
            
            all_preds.append({"conf": p["conf"], "matched": p["matched"]})

    cap.release()

    if total_truths == 0:
        print("\nDIKKAT: Videoda hic yuz bulunamadi!")
        return

    print("\nVideo islendi. Sonuclar hesaplaniyor...")

    # mAP Hesaplama
    all_preds = sorted(all_preds, key=lambda x: x["conf"], reverse=True)
    
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    for i, p in enumerate(all_preds):
        if p["matched"]:
            tp[i] = 1
        else:
            fp[i] = 1
    
    cumsum_tp = np.cumsum(tp)
    cumsum_fp = np.cumsum(fp)

    recalls = cumsum_tp / (total_truths + 1e-6)
    precisions = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-6)

    ap_05 = calculate_ap(recalls, precisions)

    # Spesifik bir confidence eşiği (örn 0.25) için skorları göster
    conf_threshold = 0.25
    filtered_preds = [p for p in all_preds if p["conf"] >= conf_threshold]
    
    final_tp = sum(1 for p in filtered_preds if p["matched"])
    final_fp = sum(1 for p in filtered_preds if not p["matched"])
    final_fn = total_truths - final_tp

    precision = final_tp / (final_tp + final_fp + 1e-6)
    recall = final_tp / (final_tp + final_fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    avg_fps_teacher = sum(fps_list_teacher) / len(fps_list_teacher)
    avg_fps_student = sum(fps_list_student) / len(fps_list_student)

    print("\n" + "="*50)
    print("                 🎯 BENCHMARK SONUCLARI")
    print("="*50)
    print(f"  Toplam Gercek Yuz Sayisi (Referans): {total_truths}")
    print(f"  Toplam Tahmin Sayisi (INT8):         {len(filtered_preds)}")
    print(f"  Dogru Tahmin (True Positive):        {final_tp}")
    print(f"  Yanlis Tahmin (False Positive):      {final_fp}")
    print(f"  Kacirilan Yuz (False Negative):      {final_fn}")
    print("-" * 50)
    print(f"  Precision (Kesinlik): % {precision * 100:.2f}")
    print(f"  Recall (Duyarlilik):  % {recall * 100:.2f}")
    print(f"  F1-Score:             % {f1_score * 100:.2f}")
    print(f"  mAP@0.5:              % {ap_05 * 100:.2f}")
    print("-" * 50)
    print("                 ⚡ HIZ KARSILASTIRMASI")
    print("-" * 50)
    print(f"  Normal Model Ortalama FPS: {avg_fps_teacher:.1f} FPS")
    print(f"  INT8 Model Ortalama FPS:   {avg_fps_student:.1f} FPS")
    print(f"  Hiz Artisi:                {(avg_fps_student/avg_fps_teacher):.1f}x Kat Daha Hizli")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Modelleri Benchmark Testi")
    parser.add_argument("--video", required=True, help="Test edilecek video dosyasi yolu (ornek: test-videos/ornek.mp4)")
    parser.add_argument("--teacher", default="yolo11-modes/face_yolo11n.onnx", help="Referans (Normal) model yolu")
    parser.add_argument("--student", default="yolo11-modes/face_yolo11n_int8.onnx", help="Test edilecek (INT8) model yolu")
    
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    
    video_path = project_root / args.video if not Path(args.video).is_absolute() else Path(args.video)
    teacher_path = project_root / args.teacher if not Path(args.teacher).is_absolute() else Path(args.teacher)
    student_path = project_root / args.student if not Path(args.student).is_absolute() else Path(args.student)

    run_benchmark(str(video_path), str(teacher_path), str(student_path))
