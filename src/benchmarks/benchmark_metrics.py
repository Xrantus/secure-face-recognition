import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

from ultralytics import YOLO
from face_recognizer import FaceRecognizer

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def crop_with_padding(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int], pad_ratio: float) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
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
    roi = frame_bgr[max(0, y1 - ph) : min(h, y2 + ph), max(0, x1 - pw) : min(w, x2 + pw)]
    return roi if roi.size > 0 else None

def extract_all_embeddings(video_path, det_model, recognizer):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Video could not be opened -> {video_path}")
        return []

    print("Detecting faces and extracting embeddings from video...")
    extracted_embs = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        res = det_model.predict(frame, imgsz=640, verbose=False, conf=0.5)[0]
        
        if res.boxes is not None:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = crop_with_padding(frame, (x1, y1, x2, y2), pad_ratio=0.20)
                if roi is not None:
                    emb = recognizer.embed_from_roi(roi)
                    if emb is not None:
                        extracted_embs.append(emb)

    cap.release()
    print(f"Process completed. Total of {len(extracted_embs)} face embeddings extracted.")
    return extracted_embs

def evaluate_metric(metric_name, video_embs, db_embs, thresholds, higher_is_better=True, VI=None):
    results = {}
    
    # Find the best match score in the DB for each face in the video
    best_scores = []
    
    for emb in video_embs:
        if metric_name == "cosine":
            scores = np.dot(db_embs, emb)
            best_scores.append(np.max(scores))
        elif metric_name == "euclidean":
            dists = np.linalg.norm(db_embs - emb, axis=1)
            best_scores.append(np.min(dists))
        elif metric_name == "manhattan":
            dists = np.sum(np.abs(db_embs - emb), axis=1)
            best_scores.append(np.min(dists))
        elif metric_name == "dot_product":
            scores = np.dot(db_embs, emb)
            best_scores.append(np.max(scores))
        elif metric_name == "pearson":
            # cdist returns correlation distance (1 - correlation). Lower is better.
            dists = cdist(db_embs, [emb], metric='correlation').flatten()
            best_scores.append(np.min(dists))
        elif metric_name == "mahalanobis":
            try:
                # VI is pseudo-inverse of covariance
                dists = cdist(db_embs, [emb], metric='mahalanobis', VI=VI).flatten()
                best_scores.append(np.min(dists))
            except Exception:
                best_scores.append(float('inf'))
                
    # Calculate Accept and Reject counts for each threshold value
    for th in thresholds:
        accepted = 0
        for score in best_scores:
            if higher_is_better:
                if score >= th: accepted += 1
            else:
                if score <= th: accepted += 1
        
        results[round(th, 3)] = accepted
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Compares distance metrics and threshold values.")
    parser.add_argument("--video", required=True, help="Path to video file under evaluation")
    parser.add_argument("--video_type", choices=["authorized", "unauthorized"], required=True, 
                        help="Video content type: 'authorized' if all persons are registered (measures TAR), 'unauthorized' if they are strangers (measures FAR)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    video_path = project_root / args.video if not Path(args.video).is_absolute() else Path(args.video)
    
    if not video_path.exists():
        print(f"ERROR: Video not found -> {video_path}")
        return

    print("Loading model and database...")
    det_model_path = project_root / "yolo11-models/face_yolo11n.onnx"
    det_model = YOLO(str(det_model_path), task='detect')
    recognizer = FaceRecognizer(det_size=(160, 160), model_name="buffalo_s")
    
    db_path = project_root / "known_faces_embeddings.npz"
    if not db_path.exists():
        print(f"ERROR: Database not found -> {db_path}")
        return
    
    db_embs, db_names = FaceRecognizer.load_db(str(db_path))
    if db_embs.size == 0:
        print("ERROR: Database is empty!")
        return

    # Attempt to calculate covariance inverse matrix for Mahalanobis
    VI = None
    try:
        # Use pseudo-inverse because if dimension (512) is larger than sample count, matrix is singular
        cov = np.cov(db_embs.T)
        VI = np.linalg.pinv(cov)
    except Exception as e:
        print(f"Mahalanobis covariance could not be calculated: {e}")

    # Extract faces from video
    video_embs = extract_all_embeddings(video_path, det_model, recognizer)
    if not video_embs:
        return

    total_faces = len(video_embs)
    print(f"\n--- TEST RESULTS (FOR {args.video_type.upper()} VIDEO) ---")
    print(f"If video is 'authorized', targeted True Acceptance Rate (TAR) should be 100%.")
    print(f"If video is 'unauthorized', targeted False Acceptance Rate (FAR) should be 0%.\n")

    metrics_config = [
        {"name": "cosine", "display": "1. Cosine Similarity", "thresholds": np.arange(0.2, 0.85, 0.05), "higher_better": True},
        {"name": "euclidean", "display": "2. Euclidean Dist (L2)", "thresholds": np.arange(0.5, 1.55, 0.1), "higher_better": False},
        {"name": "manhattan", "display": "3. Manhattan Dist (L1)", "thresholds": np.arange(10.0, 30.0, 1.0), "higher_better": False},
        {"name": "dot_product", "display": "4. Dot Product", "thresholds": np.arange(0.2, 0.85, 0.05), "higher_better": True},
        {"name": "pearson", "display": "5. Pearson Correlation", "thresholds": np.arange(0.2, 0.85, 0.05), "higher_better": False},
        {"name": "mahalanobis", "display": "6. Mahalanobis Dist", "thresholds": np.arange(0.2, 2.2, 0.2), "higher_better": False},
    ]

    best_summary = []

    for mc in metrics_config:
        print(f"\n>> {mc['display']} <<")
        results = evaluate_metric(mc["name"], video_embs, db_embs, mc["thresholds"], mc["higher_better"], VI)
        
        best_th = None
        best_rate = -1 if args.video_type == "authorized" else float('inf')
        
        for th, accepted in results.items():
            rate = (accepted / total_faces) * 100
            
            # Print to console
            if args.video_type == "authorized":
                print(f"  Threshold: {th:.2f} -> TAR (True Acceptance): {rate:.2f}% ({accepted}/{total_faces})")
                # Find best TAR (ideally highest)
                if rate > best_rate:
                    best_rate = rate
                    best_th = th
            else:
                print(f"  Threshold: {th:.2f} -> FAR (False Acceptance): {rate:.2f}% ({accepted}/{total_faces})")
                # Find best FAR (ideally lowest, closest to 0)
                if rate < best_rate:
                    best_rate = rate
                    best_th = th

        if best_th is not None:
            best_summary.append({
                "name": mc["display"],
                "best_th": best_th,
                "best_rate": best_rate
            })

    print("\n" + "="*50)
    print("--- OPTIMIZATION SUMMARY ---")
    if args.video_type == "authorized":
        print("Goal: Highest TAR (closest to 100%)")
        # Sort: highest rate first
        best_summary.sort(key=lambda x: x["best_rate"], reverse=True)
    else:
        print("Goal: Lowest FAR (closest to 0%)")
        # Sort: lowest rate first
        best_summary.sort(key=lambda x: x["best_rate"])

    for i, item in enumerate(best_summary):
        print(f"{i+1}. {item['name']} -> Recommended Threshold: {item['best_th']:.2f} | Success Rate: {item['best_rate']:.2f}%")

if __name__ == "__main__":
    main()
