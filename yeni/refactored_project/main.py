"""Live face detection + recognition (refactored).

This script unifies Mac and Raspberry Pi camera backends while keeping model choice
and metric choice hardware-agnostic via `config`.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from . import config
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer, SimilarityMetric


def clamp(v: int, lo: int, hi: int) -> int:
    """Clamp integer between lo and hi."""

    return max(lo, min(hi, v))


def crop_with_padding(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad_ratio: float,
) -> np.ndarray | None:
    """Crop and pad ROI from frame using a bbox."""

    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1 = clamp(x1, 0, w - 1)
    x2 = clamp(x2, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    y2 = clamp(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None

    bw, bh = x2 - x1, y2 - y1
    if min(bw, bh) < config.MODEL_CONFIG.min_face_size:
        return None

    pw = int(bw * pad_ratio)
    ph = int(bh * pad_ratio)
    roi = frame_bgr[max(0, y1 - ph) : min(h, y2 + ph), max(0, x1 - pw) : min(w, x2 + pw)]
    if roi.size == 0:
        return None
    return roi


def metric_threshold(metric: SimilarityMetric) -> float:
    """Pick threshold based on metric selection."""

    if metric == "cosine":
        return float(config.METRIC_CONFIG.cosine_threshold)
    return float(config.METRIC_CONFIG.euclidean_threshold)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for quick benchmarking/override experiments."""

    p = argparse.ArgumentParser(description="Live face recognition (refactored)")
    p.add_argument("--hardware-env", choices=["MAC", "RPI"], default=config.HARDWARE_ENV)
    p.add_argument("--yolo-model-path", default=config.MODEL_CONFIG.yolo_model_path)
    p.add_argument("--recognizer-model-name", default=config.MODEL_CONFIG.recognizer_model_name)
    p.add_argument("--metric", choices=["cosine", "euclidean"], default=config.METRIC_CONFIG.similarity_metric)
    p.add_argument("--threshold", type=float, default=None, help="Override metric threshold")
    p.add_argument("--db-path", default=config.MODEL_CONFIG.db_path)
    p.add_argument(
        "--video",
        default=None,
        help="Optional video file path (if set, uses cv2.VideoCapture on this file instead of a camera backend). "
        "If a relative name is provided, it will be resolved under ./test-videos/ when possible.",
    )
    return p


def resolve_model_path(project_root: Path, model_path: str) -> str:
    """Resolve YOLO model path, preferring `./yolo11-modes/` for relative names."""

    p = Path(model_path)
    if p.is_absolute() and p.is_file():
        return str(p)

    # If the caller passes a bare filename, try yolo11-modes/<name>
    candidate = project_root / "yolo11-modes" / model_path
    if candidate.is_file():
        return str(candidate)

    # If relative path already includes directories, resolve against repo root.
    candidate2 = (project_root / model_path).resolve()
    return str(candidate2)


def resolve_video_path(project_root: Path, video_path: str) -> str:
    """Resolve video path, preferring `./test-videos/` for relative names."""

    p = Path(video_path)
    if p.is_absolute() and p.is_file():
        return str(p)

    candidate = project_root / "test-videos" / video_path
    if candidate.is_file():
        return str(candidate)

    candidate2 = (project_root / video_path).resolve()
    return str(candidate2)


def run_live(
    hardware_env: Literal["MAC", "RPI"],
    yolo_model_path: str,
    recognizer_model_name: str,
    metric: SimilarityMetric,
    threshold_override: float | None,
    db_path: str,
    video: str | None,
) -> None:
    """Run the live camera pipeline for the selected hardware backend."""

    project_root = Path(__file__).resolve().parents[1]
    db_abs = str(project_root / db_path)

    yolo_model_abs = resolve_model_path(project_root, yolo_model_path)

    detector = FaceDetector(
        model_path=yolo_model_abs,
        img_size=config.MODEL_CONFIG.yolo_img_size,
        pred_conf=config.MODEL_CONFIG.yolo_pred_conf,
        iou=config.MODEL_CONFIG.yolo_iou,
        max_det=config.MODEL_CONFIG.max_det,
        det_threshold=config.MODEL_CONFIG.yolo_det_threshold,
    )

    recognizer = FaceRecognizer(
        det_size=config.MODEL_CONFIG.det_size,
        model_name=recognizer_model_name,
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"] if hardware_env == "MAC" else None,
    )

    db_embs, db_names = FaceRecognizer.load_db(db_abs)
    threshold = float(threshold_override) if threshold_override is not None else metric_threshold(metric)

    if video is not None:
        video_abs = resolve_video_path(project_root, video)
        _run_video(video_abs, detector, recognizer, db_embs, db_names, metric, threshold)
        return

    if hardware_env == "MAC":
        _run_mac(detector, recognizer, db_embs, db_names, metric, threshold)
    else:
        _run_rpi(detector, recognizer, db_embs, db_names, metric, threshold)


def _run_mac(
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    db_embs: np.ndarray,
    db_names: np.ndarray,
    metric: SimilarityMetric,
    threshold: float,
) -> None:
    """Mac backend: OpenCV VideoCapture + GUI."""

    latest_frame: np.ndarray | None = None
    frame_lock = threading.Lock()
    running = True

    def reader(cap: cv2.VideoCapture) -> None:
        nonlocal latest_frame, running
        while running:
            ok, frame = cap.read()
            if not ok:
                running = False
                break
            with frame_lock:
                latest_frame = frame

    cap = cv2.VideoCapture(config.CAMERA_CONFIG.mac_camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_CONFIG.mac_frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_CONFIG.mac_frame_height)
    if not cap.isOpened():
        raise SystemExit("Mac kamerasi baslatilamadi.")

    t = threading.Thread(target=reader, args=(cap,), daemon=True)
    t.start()

    while latest_frame is None and running:
        time.sleep(0.05)

    print("Sistem Aktif! Penceriyi kapatmak icin 'q' tusuna basin.\n")

    frame_counter = 0
    last_dets: list = []
    fps_t0 = time.time()
    fps_n = 0

    try:
        while running:
            with frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()

            frame = cv2.flip(frame, 1)
            fps_n += 1

            if frame_counter % config.MODEL_CONFIG.frame_skip == 0:
                last_dets = detector.detect(frame)

            for det in last_dets:
                roi = crop_with_padding(frame, det.bbox, config.MODEL_CONFIG.landmark_pad)
                if roi is None:
                    continue

                emb = recognizer.embed_from_roi(roi)
                if emb is None:
                    continue

                name, score = FaceRecognizer.predict_identity(
                    emb=emb,
                    db_embs=db_embs,
                    db_names=db_names,
                    metric=metric,
                    threshold=threshold,
                )

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} {score:.2f}" if metric == "cosine" else f"{name} {score:.3f}"

                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            now = time.time()
            if now - fps_t0 >= 1.0:
                print(f"Anlik FPS: {fps_n / (now - fps_t0):.2f}")
                fps_t0 = now
                fps_n = 0

            cv2.imshow("Mac Live (Refactored)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False
                break

            frame_counter += 1
    finally:
        running = False
        cap.release()
        t.join(timeout=1)
        cv2.destroyAllWindows()


def _run_rpi(
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    db_embs: np.ndarray,
    db_names: np.ndarray,
    metric: SimilarityMetric,
    threshold: float,
) -> None:
    """RPi backend: Picamera2 + headless loop (CTRL+C to exit)."""

    try:
        from picamera2 import Picamera2
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Picamera2 import edilemedi. Bu backend sadece Raspberry Pi icindir."
        ) from e

    latest_frame: np.ndarray | None = None
    frame_lock = threading.Lock()
    running = True

    def reader(picam: Picamera2) -> None:
        nonlocal latest_frame, running
        while running:
            try:
                rgb = picam.capture_array()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                with frame_lock:
                    latest_frame = bgr
            except Exception:
                running = False
                break

    picam = Picamera2()
    cfg = picam.create_preview_configuration({"size": config.CAMERA_CONFIG.rpi_preview_size})
    picam.configure(cfg)
    picam.start()

    t = threading.Thread(target=reader, args=(picam,), daemon=True)
    t.start()

    while latest_frame is None and running:
        time.sleep(0.05)

    print("Sistem Aktif! (Durdurmak icin terminalde CTRL+C)\n")

    frame_counter = 0
    last_dets: list = []
    fps_t0 = time.time()
    fps_n = 0

    try:
        while running:
            with frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()

            frame = cv2.flip(frame, 1)
            fps_n += 1

            if frame_counter % config.MODEL_CONFIG.frame_skip == 0:
                last_dets = detector.detect(frame)

            for det in last_dets:
                roi = crop_with_padding(frame, det.bbox, config.MODEL_CONFIG.landmark_pad)
                if roi is None:
                    continue

                emb = recognizer.embed_from_roi(roi)
                if emb is None:
                    continue

                name, score = FaceRecognizer.predict_identity(
                    emb=emb,
                    db_embs=db_embs,
                    db_names=db_names,
                    metric=metric,
                    threshold=threshold,
                )

                if name != "Unknown":
                    print(f"[BASARILI] {name} tespit edildi! (Skor: {score:.3f})")

            now = time.time()
            if now - fps_t0 >= 1.0:
                print(f"Anlik FPS: {fps_n / (now - fps_t0):.2f}")
                fps_t0 = now
                fps_n = 0

            frame_counter += 1
    except KeyboardInterrupt:
        print("\n[BILGI] CTRL+C algilandi, sistem kapatiliyor...")
    finally:
        running = False
        t.join(timeout=1)
        try:
            picam.stop()
        except Exception:
            pass


def _run_video(
    video_path: str,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    db_embs: np.ndarray,
    db_names: np.ndarray,
    metric: SimilarityMetric,
    threshold: float,
) -> None:
    """Video backend: run the pipeline on a video file using OpenCV."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Video acilamadi: {video_path}")

    print(f"Video modu aktif: {video_path}")
    print("Cikmak icin 'q' tusuna basin.\n")

    fps_t0 = time.time()
    fps_n = 0
    frame_counter = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame = cv2.flip(frame, 1)
            fps_n += 1

            dets = detector.detect(frame) if frame_counter % config.MODEL_CONFIG.frame_skip == 0 else []

            for det in dets:
                roi = crop_with_padding(frame, det.bbox, config.MODEL_CONFIG.landmark_pad)
                if roi is None:
                    continue

                emb = recognizer.embed_from_roi(roi)
                if emb is None:
                    continue

                name, score = FaceRecognizer.predict_identity(
                    emb=emb,
                    db_embs=db_embs,
                    db_names=db_names,
                    metric=metric,
                    threshold=threshold,
                )
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} {score:.2f}" if metric == "cosine" else f"{name} {score:.3f}"

                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            now = time.time()
            if now - fps_t0 >= 1.0:
                print(f"Anlik FPS: {fps_n / (now - fps_t0):.2f}")
                fps_t0 = now
                fps_n = 0

            cv2.imshow("Video (Refactored)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_counter += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    """CLI entry-point."""

    args = build_arg_parser().parse_args()
    run_live(
        hardware_env=args.hardware_env,
        yolo_model_path=args.yolo_model_path,
        recognizer_model_name=args.recognizer_model_name,
        metric=args.metric,
        threshold_override=args.threshold,
        db_path=args.db_path,
        video=args.video,
    )


if __name__ == "__main__":
    main()

