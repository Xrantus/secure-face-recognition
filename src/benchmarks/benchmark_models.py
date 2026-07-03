import argparse
import time
import os
import cv2
import numpy as np
from pathlib import Path

from face_detector import FaceDetector
from face_recognizer import FaceRecognizer

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def crop_with_padding(img_bgr: np.ndarray, bbox: tuple[int, int, int, int], pad_ratio: float) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    h, w = img_bgr.shape[:2]
    x1 = clamp(x1, 0, w - 1)
    x2 = clamp(x2, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    y2 = clamp(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    bw, bh = x2 - x1, y2 - y1
    if min(bw, bh) < 35: 
        return None
    pw = int(bw * pad_ratio)
    ph = int(bh * pad_ratio)
    roi = img_bgr[max(0, y1 - ph) : min(h, y2 + ph), max(0, x1 - pw) : min(w, x2 + pw)]
    return roi if roi.size > 0 else None

def create_db_in_memory(image_folder, detector, recognizer):
    image_root = Path(image_folder)
    person_dirs = sorted([p for p in image_root.iterdir() if p.is_dir()])
    
    global_embeddings = []
    global_names = []
    
    for person_dir in person_dirs:
        name = person_dir.name
        per_embeddings = []
        for filename in sorted(os.listdir(person_dir)):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = str(person_dir / filename)
            try:
                arr = np.fromfile(img_path, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                continue
                
            if img is None: continue
            
            dets = detector.detect(img)
            best = FaceDetector.best_by_conf(dets)
            if best is None: continue
            
            roi = crop_with_padding(img, best.bbox, 0.20)
            if roi is None: continue
            
            emb = recognizer.embed_from_roi(roi)
            if emb is not None:
                per_embeddings.append(emb)
        
        if per_embeddings:
            mean_emb = np.mean(np.stack(per_embeddings, axis=0), axis=0)
            mean_emb = FaceRecognizer.l2_normalize(mean_emb)
            global_embeddings.append(mean_emb)
            global_names.append(name)
            
    return np.array(global_embeddings), np.array(global_names)

def main():
    parser = argparse.ArgumentParser(description="Compares different InsightFace models (e.g. buffalo_s, buffalo_l).")
    parser.add_argument("--video", required=True, help="Video file path under evaluation")
    parser.add_argument("--video_type", choices=["authorized", "unauthorized"], required=True, help="Type of evaluation: TAR or FAR")
    parser.add_argument("--model", default="buffalo_s", help="InsightFace model identifier (e.g., buffalo_s, buffalo_l, antelopev2)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    video_path = project_root / args.video if not Path(args.video).is_absolute() else Path(args.video)

    if not video_path.exists():
        print(f"ERROR: Video not found -> {video_path}")
        return

    print("Loading YOLO Detector...")
    det_model_path = project_root / "yolo11-models/face_yolo11n.onnx"
    detector = FaceDetector(model_path=str(det_model_path), img_size=640, pred_conf=0.5, iou=0.4, max_det=10, det_threshold=0.5)
    
    print(f"\n--- Loading Face Recognition Model: '{args.model}' ---")
    print("Note: If the model is not found locally, InsightFace will download it automatically. Please wait...")
    try:
        recognizer = FaceRecognizer(det_size=(160, 160), model_name=args.model)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return

    print("\nGenerating model-specific database in memory...")
    # Since the embedding vectors vary across different models, we construct
    # the temporary database on the fly from raw enrollment images.
    db_embs, db_names = create_db_in_memory(str(project_root / "db_images"), detector, recognizer)
    
    if db_embs.size == 0:
        print("ERROR: Database could not be initialized! Verify the db_images folder.")
        return
    print(f"Database ready. Registered personnel count: {len(db_names)}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERROR: Could not open video file.")
        return

    print(f"\nAnalyzing video frames to measure embedding extraction throughput (FPS)...")
    extracted_embs = []
    inference_times = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        dets = detector.detect(frame)
        for d in dets:
            roi = crop_with_padding(frame, d.bbox, 0.20)
            if roi is not None:
                # Measure latency specifically for the embedding process
                t0 = time.time()
                emb = recognizer.embed_from_roi(roi)
                t1 = time.time()
                
                if emb is not None:
                    extracted_embs.append(emb)
                    inference_times.append((t1 - t0) * 1000) # In milliseconds (ms)
                    
    cap.release()
    
    if not extracted_embs:
        print("No faces detected in the video stream.")
        return

    avg_time_ms = np.mean(inference_times)
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0

    print("\n" + "="*50)
    print(f"--- MODEL PERFORMANCE REPORT ({args.model.upper()}) ---")
    print(f"Processed Faces: {len(extracted_embs)}")
    print(f"Average Feature Extraction Latency: {avg_time_ms:.2f} ms")
    print(f"Estimated Throughput:               {fps:.1f} FPS")
    print("="*50)

    # Cosine Threshold Sweeping
    thresholds = np.arange(0.2, 0.85, 0.05)
    best_th = None
    best_rate = -1 if args.video_type == "authorized" else float('inf')

    print(f"\n--- ACCURACY ANALYSIS (Cosine Similarity) ---")
    total_faces = len(extracted_embs)
    for th in thresholds:
        accepted = 0
        for emb in extracted_embs:
            scores = np.dot(db_embs, emb)
            if np.max(scores) >= th:
                accepted += 1
                
        rate = (accepted / total_faces) * 100
        
        if args.video_type == "authorized":
            print(f"  Threshold: {th:.2f} -> TAR: {rate:.2f}%")
            if rate > best_rate:
                best_rate = rate
                best_th = th
        else:
            print(f"  Threshold: {th:.2f} -> FAR: {rate:.2f}%")
            if rate < best_rate:
                best_rate = rate
                best_th = th

    print("\n--- OPTIMIZATION RESULT ---")
    print(f"Goal: {'Highest TAR' if args.video_type == 'authorized' else 'Lowest FAR'}")
    print(f"Recommended Threshold: {best_th:.2f} (Accuracy: {best_rate:.2f}%)")

if __name__ == "__main__":
    main()
