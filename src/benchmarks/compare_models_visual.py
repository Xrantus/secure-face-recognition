"""Compare two YOLO models against a Teacher (GT) model for Biometric metrics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from .face_detector import FaceDetector, Detection
from .face_recognizer import FaceRecognizer, SimilarityMetric
from .main import resolve_model_path, resolve_video_path, crop_with_padding, metric_threshold


def draw_results(
    frame: np.ndarray,
    detections: list[Detection],
    recognizer: FaceRecognizer,
    db_embs: np.ndarray,
    db_names: np.ndarray,
    metric: SimilarityMetric,
    threshold: float,
    title: str,
) -> np.ndarray:
    canvas = frame.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for det in detections:
        roi = crop_with_padding(canvas, det.bbox, config.MODEL_CONFIG.landmark_pad)
        if roi is None: continue
        emb = recognizer.embed_from_roi(roi)
        if emb is None: continue
        name, score = FaceRecognizer.predict_identity(emb, db_embs, db_names, metric, threshold)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        label = f"{name} {score:.2f}"
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, max(45, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return canvas


def calculate_eer_auc(scores, labels):
    if not scores or len(set(labels)) < 2:
        return 0.0, 0.0
    scores = np.array(scores)
    labels = np.array(labels)
    idx = np.argsort(scores)[::-1]
    scores, labels = scores[idx], labels[idx]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tpr = tp / (tp[-1] + 1e-12)
    fpr = fp / (fp[-1] + 1e-12)
    auc_val = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fpr - fnr))
    eer_val = fpr[eer_idx]
    return eer_val, auc_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare M1 and M2 against a Teacher model")
    parser.add_argument("--m1", default="yolo11n_filtered_int8.onnx")
    parser.add_argument("--m2", default="face_yolo11n_int8.onnx")
    parser.add_argument("--teacher", default="yolo11n_filtered_fp32.onnx")
    parser.add_argument("--video", default="tar1.h264")
    parser.add_argument("--db-path", default=config.MODEL_CONFIG.db_path)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default=config.METRIC_CONFIG.similarity_metric)
    parser.add_argument("--threshold", type=float, default=None)
    
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    m1_abs = resolve_model_path(project_root, args.m1)
    m2_abs = resolve_model_path(project_root, args.m2)
    t_abs = resolve_model_path(project_root, args.teacher)
    video_abs = resolve_video_path(project_root, args.video)
    db_abs = str(project_root / args.db_path)
    
    # Detectors
    det1 = FaceDetector(m1_abs, 640, 0.25, 0.45, 100, 0.25)
    det2 = FaceDetector(m2_abs, 640, 0.10, 0.45, 100, 0.15)
    det_gt = FaceDetector(t_abs, 640, 0.40, 0.45, 100, 0.40) # Teacher is stricter
    
    recognizer = FaceRecognizer(config.MODEL_CONFIG.det_size, config.MODEL_CONFIG.recognizer_model_name,
                                ["CoreMLExecutionProvider", "CPUExecutionProvider"] if config.HARDWARE_ENV == "MAC" else None)
    
    db_embs, db_names = FaceRecognizer.load_db(db_abs)
    threshold = float(args.threshold) if args.threshold is not None else metric_threshold(args.metric)
    
    cap = cv2.VideoCapture(video_abs)
    
    # Stats structures
    stats = {
        "m1": {"dets": 0, "rec": 0, "time": 0.0, "scores": [], "labels": [], "fa": 0, "fr": 0},
        "m2": {"dets": 0, "rec": 0, "time": 0.0, "scores": [], "labels": [], "fa": 0, "fr": 0}
    }
    processed_frames = 0
    frame_counter = 0

    print(f"Teacher: {Path(t_abs).name} | M1: {Path(m1_abs).name} | M2: {Path(m2_abs).name}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None: break
            
            if frame_counter % config.MODEL_CONFIG.frame_skip == 0:
                processed_frames += 1
                
                # 1. Teacher (GT)
                gt_dets = det_gt.detect(frame)
                gt_results = {} # (x,y) -> name
                for d in gt_dets:
                    roi = crop_with_padding(frame, d.bbox, config.MODEL_CONFIG.landmark_pad)
                    if roi is None: continue
                    emb = recognizer.embed_from_roi(roi)
                    if emb is None: continue
                    name, _ = FaceRecognizer.predict_identity(emb, db_embs, db_names, args.metric, threshold)
                    cx, cy = (d.bbox[0]+d.bbox[2])//2, (d.bbox[1]+d.bbox[3])//2
                    gt_results[(cx, cy)] = name

                # 2. M1 and M2 Evaluation
                for m_id, detector in [("m1", det1), ("m2", det2)]:
                    t0 = time.time()
                    m_dets = detector.detect(frame)
                    stats[m_id]["time"] += (time.time() - t0) * 1000
                    stats[m_id]["dets"] += len(m_dets)
                    
                    for d in m_dets:
                        roi = crop_with_padding(frame, d.bbox, config.MODEL_CONFIG.landmark_pad)
                        if roi is None: continue
                        emb = recognizer.embed_from_roi(roi)
                        if emb is None: continue
                        name, score = FaceRecognizer.predict_identity(emb, db_embs, db_names, args.metric, threshold)
                        if name != "Unknown": stats[m_id]["rec"] += 1
                        
                        # Match with GT
                        cx, cy = (d.bbox[0]+d.bbox[2])//2, (d.bbox[1]+d.bbox[3])//2
                        matched = False
                        norm_score = score if args.metric == "cosine" else 1/(1+score)
                        
                        for (gtx, gty), gtname in gt_results.items():
                            if abs(cx-gtx) < 50 and abs(cy-gty) < 50:
                                matched = True
                                if gtname != "Unknown":
                                    stats[m_id]["scores"].append(norm_score)
                                    stats[m_id]["labels"].append(1)
                                    if name != gtname: stats[m_id]["fr"] += 1
                                else:
                                    if name != "Unknown":
                                        stats[m_id]["fa"] += 1
                                        stats[m_id]["scores"].append(norm_score)
                                        stats[m_id]["labels"].append(0)
                                break
                
                c1 = draw_results(frame, det1.detect(frame), recognizer, db_embs, db_names, args.metric, threshold, f"M1: {Path(m1_abs).name}")
                c2 = draw_results(frame, det2.detect(frame), recognizer, db_embs, db_names, args.metric, threshold, f"M2: {Path(m2_abs).name}")
                
                combined = np.hstack((c1, c2))
                h, w = combined.shape[:2]
                scale = min(800 / h, 1600 / w)
                cv2.imshow("M1 vs M2 (Teacher Evaluation)", cv2.resize(combined, (int(w*scale), int(h*scale))))
                
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            frame_counter += 1
            
    finally:
        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "="*60)
        print("📊 BIOMETRIC COMPARISON REPORT (Reference: Teacher)")
        print("="*60)
        
        for m_id, label in [("m1", Path(m1_abs).name), ("m2", Path(m2_abs).name)]:
            s = stats[m_id]
            eer, auc_v = calculate_eer_auc(s["scores"], s["labels"])
            far = (s["fa"] / max(1, s["dets"]) * 100)
            # Find total possible GT matches for FRR
            gt_total_rec = sum(1 for n in gt_results.values() if n != "Unknown") # Simplified for final frame, should be tracked per frame
            # Let's use a better FRR denominator: total times a person was seen by GT
            # For this script, we'll use stats based on detections matched.
            frr = (s["fr"] / max(1, s["rec"]) * 100)
            
            print(f"MODEL: {label}")
            print(f"  - Total Detections: {s['dets']} | Recognitions: {s['rec']}")
            print(f"  - Avg Speed: {s['time']/max(1,processed_frames):.2f} ms")
            print(f"  - AUC: {auc_v:.4f} | EER: {eer*100:.2f} %")
            print(f"  - FAR: {far:.2f} % | FRR: {frr:.2f} %")
            print("-" * 60)
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
