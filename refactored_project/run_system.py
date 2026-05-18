"""Entrypoint for running the Live Camera Loop alongside the FastAPI Server.

This script starts the FastAPI server in a background thread and runs the 
camera loop (Mac/WIN or RPi) in the main thread.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import uvicorn
import requests

from . import config
from .api_server import app, setup_api
from .backend_client import fetch_and_save_embeddings, send_access_log, sync_offline_logs
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer, SimilarityMetric
from .main import resolve_model_path, resolve_video_path, crop_with_padding, metric_threshold

# Global thread-safety locks
inference_lock = threading.Lock()
db_lock = threading.Lock()

# We store the DB state in a dictionary so we can update it from the reload callback
db_state = {
    "embs": np.array([]),
    "names": np.array([])
}

# Gecis loglarinin her saniye spam olmamasi icin bekleme suresi (debounce)
LOG_COOLDOWN_SECONDS = 5.0
last_seen = {}

def fetch_and_reload_db(db_abs: str):
    """Fetch new embeddings from Backend and update db_state in memory."""
    new_data = fetch_and_save_embeddings(db_abs)
    
    if new_data is not None:
        new_embs, new_names = new_data
        with db_lock:
            db_state["embs"] = new_embs
            db_state["names"] = new_names
        print("[API] Veritabani basariyla guncellendi!\n")
    else:
        print("[API] Backend baglantisi kurulamadi veya veri alinamadi. Mevcut/Eski DB kullaniliyor.\n")



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live face recognition with API (refactored)")
    p.add_argument("--hardware-env", choices=["MAC", "RPI", "WIN"], default=config.HARDWARE_ENV)
    p.add_argument("--yolo-model-path", default=config.MODEL_CONFIG.yolo_model_path)
    p.add_argument("--recognizer-model-name", default=config.MODEL_CONFIG.recognizer_model_name)
    p.add_argument("--metric", choices=["cosine", "euclidean"], default=config.METRIC_CONFIG.similarity_metric)
    p.add_argument("--threshold", type=float, default=None, help="Override metric threshold")
    p.add_argument("--db-path", default=config.MODEL_CONFIG.db_path)
    p.add_argument("--video", default=None)
    return p


def run_live_with_api(
    hardware_env: Literal["MAC", "RPI", "WIN"],
    yolo_model_path: str,
    recognizer_model_name: str,
    metric: SimilarityMetric,
    threshold_override: float | None,
    db_path: str,
    video: str | None,
) -> None:
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

    # Initial DB load (onceki verilerle basla)
    try:
        init_embs, init_names = FaceRecognizer.load_db(db_abs)
        db_state["embs"] = init_embs
        db_state["names"] = init_names
    except Exception:
        print("[SISTEM] Lokal DB bulunamadi. Ilk senkronizasyon bekleniyor...")
        
    # Arka planda baslangic senkronizasyonu ve offline log gonderimi yap
    threading.Thread(target=fetch_and_reload_db, args=(db_abs,), daemon=True).start()
    threading.Thread(target=sync_offline_logs, daemon=True).start()

    threshold = float(threshold_override) if threshold_override is not None else metric_threshold(metric)

    # Setup the API with dependencies
    setup_api(
        recognizer=recognizer,
        inference_lock=inference_lock,
        db_lock=db_lock,
        reload_callback=lambda: fetch_and_reload_db(db_abs)
    )

    # Start FastAPI in a background thread
    def start_uvicorn():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

    api_thread = threading.Thread(target=start_uvicorn, daemon=True)
    api_thread.start()
    print("[SYSTEM] FastAPI sunucusu 0.0.0.0:8000 adresinde baslatildi.")

    if video is not None:
        video_abs = resolve_video_path(project_root, video)
        _run_video(video_abs, detector, recognizer, metric, threshold)
        return

    if hardware_env in ("MAC", "WIN"):
        _run_mac(detector, recognizer, metric, threshold)
    else:
        _run_rpi(detector, recognizer, metric, threshold)


def _run_mac(
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    metric: SimilarityMetric,
    threshold: float,
) -> None:
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

    cap = cv2.VideoCapture(config.CAMERA_CONFIG.opencv_camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_CONFIG.opencv_frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_CONFIG.opencv_frame_height)
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
                with inference_lock:
                    last_dets = detector.detect(frame)

            for det in last_dets:
                roi = crop_with_padding(frame, det.bbox, config.MODEL_CONFIG.landmark_pad)
                if roi is None:
                    continue

                with inference_lock:
                    emb = recognizer.embed_from_roi(roi)
                if emb is None:
                    continue

                with db_lock:
                    curr_embs = db_state["embs"]
                    curr_names = db_state["names"]

                name, score = FaceRecognizer.predict_identity(
                    emb=emb,
                    db_embs=curr_embs,
                    db_names=curr_names,
                    metric=metric,
                    threshold=threshold,
                )

                if name != "Unknown":
                    t_now = time.time()
                    if name not in last_seen or (t_now - last_seen[name]) > LOG_COOLDOWN_SECONDS:
                        last_seen[name] = t_now
                        threading.Thread(target=send_access_log, args=(name,), daemon=True).start()

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} {score:.2f}" if metric == "cosine" else f"{name} {score:.3f}"

                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            now = time.time()
            if now - fps_t0 >= 1.0:
                print(f"Anlik FPS: {fps_n / (now - fps_t0):.2f}")
                fps_t0 = now
                fps_n = 0

            cv2.imshow("Mac Live + API", frame)
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
    metric: SimilarityMetric,
    threshold: float,
) -> None:
    try:
        from picamera2 import Picamera2
    except Exception as e:
        raise SystemExit("Picamera2 import edilemedi. Bu backend sadece Raspberry Pi icindir.") from e

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
                with inference_lock:
                    last_dets = detector.detect(frame)

            for det in last_dets:
                roi = crop_with_padding(frame, det.bbox, config.MODEL_CONFIG.landmark_pad)
                if roi is None:
                    continue

                with inference_lock:
                    emb = recognizer.embed_from_roi(roi)
                if emb is None:
                    continue

                with db_lock:
                    curr_embs = db_state["embs"]
                    curr_names = db_state["names"]

                name, score = FaceRecognizer.predict_identity(
                    emb=emb,
                    db_embs=curr_embs,
                    db_names=curr_names,
                    metric=metric,
                    threshold=threshold,
                )

                if name != "Unknown":
                    print(f"[BASARILI] {name} tespit edildi! (Skor: {score:.3f})")
                    t_now = time.time()
                    if name not in last_seen or (t_now - last_seen[name]) > LOG_COOLDOWN_SECONDS:
                        last_seen[name] = t_now
                        threading.Thread(target=send_access_log, args=(name,), daemon=True).start()

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
    metric: SimilarityMetric,
    threshold: float,
) -> None:
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

            if frame_counter % config.MODEL_CONFIG.frame_skip == 0:
                with inference_lock:
                    dets = detector.detect(frame)
            else:
                dets = []

            for det in dets:
                roi = crop_with_padding(frame, det.bbox, config.MODEL_CONFIG.landmark_pad)
                if roi is None:
                    continue

                with inference_lock:
                    emb = recognizer.embed_from_roi(roi)
                if emb is None:
                    continue

                with db_lock:
                    curr_embs = db_state["embs"]
                    curr_names = db_state["names"]

                name, score = FaceRecognizer.predict_identity(
                    emb=emb,
                    db_embs=curr_embs,
                    db_names=curr_names,
                    metric=metric,
                    threshold=threshold,
                )
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} {score:.2f}" if metric == "cosine" else f"{name} {score:.3f}"

                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            now = time.time()
            if now - fps_t0 >= 1.0:
                print(f"Anlik FPS: {fps_n / (now - fps_t0):.2f}")
                fps_t0 = now
                fps_n = 0

            cv2.imshow("Video + API", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_counter += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = build_arg_parser().parse_args()
    run_live_with_api(
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
