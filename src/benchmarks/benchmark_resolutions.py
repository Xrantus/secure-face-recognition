import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import benchmark_video # Helper functions for calculate_iou and calculate_ap
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
    if min(bw, bh) < 35: # min face size
        return None
    pw = int(bw * pad_ratio)
    ph = int(bh * pad_ratio)
    roi = frame_bgr[max(0, y1 - ph) : min(h, y2 + ph), max(0, x1 - pw) : min(w, x2 + pw)]
    return roi if roi.size > 0 else None

def run_benchmark_for_size(video_path, teacher_model, student_model, main_size, recognizer, db_embs, db_names):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Video could not be opened -> {video_path}")
        return None

    # Stats
    all_preds = [] 
    total_truths = 0
    fps_list_student = []
    recognition_stats = {}

    # Original video details
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Simulate the downscaling (main_size) performed by camera hardware
        # If the source video is raw (e.g. 1640x1232), this mimics ISP downscaling.
        if (orig_w, orig_h) != main_size:
            frame_resized = cv2.resize(frame, main_size)
        else:
            frame_resized = frame

        # 1. Inference with Teacher (Baseline) model on resized frame
        res_teacher = teacher_model.predict(frame_resized, imgsz=640, verbose=False, conf=0.5)[0]
        
        truths = []
        if res_teacher.boxes is not None:
            for box in res_teacher.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                truths.append({"bbox": (x1, y1, x2, y2), "matched": False})
        total_truths += len(truths)

        # 2. Inference with Student (INT8) model
        t0 = time.time()
        res_student = student_model.predict(frame_resized, imgsz=640, verbose=False, conf=0.01)[0]
        t1 = time.time()
        fps_list_student.append(1 / (t1 - t0 + 1e-6))
        
        preds = []
        if res_student.boxes is not None:
            for box in res_student.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                preds.append({"bbox": (x1, y1, x2, y2), "conf": conf, "matched": False})
                
                # --- Face Recognition (Downstream) ---
                roi = crop_with_padding(frame_resized, (x1, y1, x2, y2), pad_ratio=0.20)
                if roi is not None:
                    emb = recognizer.embed_from_roi(roi)
                    if emb is not None:
                        name, score = FaceRecognizer.predict_identity(
                            emb=emb,
                            db_embs=db_embs,
                            db_names=db_names,
                            metric="cosine",
                            threshold=0.50
                        )
                        recognition_stats[name] = recognition_stats.get(name, 0) + 1

        preds = sorted(preds, key=lambda x: x["conf"], reverse=True)

        # 3. IoU Matching
        for p in preds:
            best_iou = 0
            best_truth_idx = -1
            
            for i, t in enumerate(truths):
                if not t["matched"]:
                    iou = benchmark_video.calculate_iou(p["bbox"], t["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_truth_idx = i
            
            if best_iou >= 0.5: # iou_threshold
                truths[best_truth_idx]["matched"] = True
                p["matched"] = True
            
            all_preds.append({"conf": p["conf"], "matched": p["matched"]})

    cap.release()

    if total_truths == 0:
        return None

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

    ap_05 = benchmark_video.calculate_ap(recalls, precisions)

    conf_threshold = 0.25
    filtered_preds = [p for p in all_preds if p["conf"] >= conf_threshold]
    
    final_tp = sum(1 for p in filtered_preds if p["matched"])
    final_fp = sum(1 for p in filtered_preds if not p["matched"])
    final_fn = total_truths - final_tp

    precision = final_tp / (final_tp + final_fp + 1e-6)
    recall = final_tp / (final_tp + final_fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    avg_fps_student = sum(fps_list_student) / len(fps_list_student) if fps_list_student else 0

    return {
        "total_truths": total_truths,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "ap_05": ap_05,
        "fps_student": avg_fps_student,
        "recognition_stats": recognition_stats
    }

def main():
    parser = argparse.ArgumentParser(description="Simulate different main_size (camera resolution) scenarios on the same video.")
    parser.add_argument("--video", required=True, help="Path to video file under evaluation (e.g. test_videos/sample.h264)")
    parser.add_argument("--teacher", default="yolo11-models/face_yolo11n.onnx", help="Path to reference (Baseline) model")
    parser.add_argument("--student", default="yolo11-models/face_yolo11n_int8.onnx", help="Path to target model (INT8)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    video_path = project_root / args.video if not Path(args.video).is_absolute() else Path(args.video)
    teacher_path = project_root / args.teacher if not Path(args.teacher).is_absolute() else Path(args.teacher)
    student_path = project_root / args.student if not Path(args.student).is_absolute() else Path(args.student)

    if not video_path.exists():
        print(f"ERROR: Video not found -> {video_path}")
        return

    print("Loading models...")
    teacher_model = YOLO(str(teacher_path), task='detect')
    student_model = YOLO(str(student_path), task='detect')

    print("Loading face recognition model (buffalo_s) and database...")
    recognizer = FaceRecognizer(det_size=(160, 160), model_name="buffalo_s")
    db_path = project_root / "known_faces_embeddings.npz"
    if db_path.exists():
        db_embs, db_names = FaceRecognizer.load_db(str(db_path))
    else:
        print(f"WARNING: Database not found -> {db_path}. Defaulting to 'Unknown' predictions.")
        db_embs, db_names = np.array([]), np.array([])

    # Different resolution configurations supported by RPi camera hardware
    main_sizes_to_test = [
        (112, 112),
        (160, 160),
        (320, 320),
        (320, 240),
        (640, 480),
        (800, 600),
        (1024, 768),
        (1280, 720),
        (1640, 1232)
    ]

    print(f"\nVideo: {video_path}")
    print("="*50)

    for main_size in main_sizes_to_test:
        print(f"\n--- Starting Evaluation for Main Size: {main_size} ---")
        
        try:
            metrics = run_benchmark_for_size(video_path, teacher_model, student_model, main_size, recognizer, db_embs, db_names)
            
            if metrics:
                print(f"  Total Ground Truth Faces: {metrics['total_truths']}")
                print(f"  Precision:                 {metrics['precision'] * 100:.2f} %")
                print(f"  Recall:                    {metrics['recall'] * 100:.2f} %")
                print(f"  F1-Score:                  {metrics['f1_score'] * 100:.2f} %")
                print(f"  mAP@0.5:                   {metrics['ap_05'] * 100:.2f} %")
                print(f"  Student Avg FPS:           {metrics['fps_student']:.1f} FPS")
                
                print("  --- Recognition Statistics ---")
                if not metrics["recognition_stats"]:
                    print("    No faces recognized.")
                else:
                    for name, count in metrics["recognition_stats"].items():
                        print(f"    {name}: {count} times")
            else:
                print("  No faces detected or benchmark failed.")
                
        except Exception as e:
             print(f"  ERROR: {e}")

    print("\n" + "="*50)
    print("All tests completed.")

if __name__ == "__main__":
    main()
