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

from . import config
from .api_server import app, setup_api
from .backend_client import fetch_and_save_embeddings, send_access_log, send_unknown_access_log, sync_offline_logs
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer, SimilarityMetric
from .face_ui import WINDOW_TITLE, draw_face_label, show_frame, start_rpi_preview, stop_rpi_preview
from .main import resolve_model_path, resolve_video_path, crop_with_padding, metric_threshold


class LiveFaceRecognitionSystem:
    def __init__(
        self,
        hardware_env: Literal["MAC", "RPI", "WIN"],
        yolo_model_path: str,
        recognizer_model_name: str,
        metric: SimilarityMetric,
        threshold_override: float | None,
        db_path: str,
        video: str | None,
    ):
        self.hardware_env = hardware_env
        self.metric = metric
        self.threshold = float(threshold_override) if threshold_override is not None else metric_threshold(metric)
        self.video = video
        
        self.project_root = Path(__file__).resolve().parents[1]
        self.db_abs = str(self.project_root / db_path)
        yolo_model_abs = resolve_model_path(self.project_root, yolo_model_path)
        
        self.detector = FaceDetector(
            model_path=yolo_model_abs,
            img_size=config.MODEL_CONFIG.yolo_img_size,
            pred_conf=config.MODEL_CONFIG.yolo_pred_conf,
            iou=config.MODEL_CONFIG.yolo_iou,
            max_det=config.MODEL_CONFIG.max_det,
            det_threshold=config.MODEL_CONFIG.yolo_det_threshold,
        )

        self.recognizer = FaceRecognizer(
            det_size=config.MODEL_CONFIG.det_size,
            model_name=recognizer_model_name,
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"] if hardware_env == "MAC" else None,
        )

        # Locks and state for thread-safe operations
        self.inference_lock = threading.Lock()
        self.db_lock = threading.Lock()
        self.db_state = {
            "embs": np.array([]),
            "names": np.array([])
        }
        
        self.LOG_COOLDOWN_SECONDS = 5.0
        self.last_seen = {}
        self.UNKNOWN_LOG_KEY = "__UNKNOWN__"

    def fetch_and_reload_db(self) -> None:
        """Fetch new embeddings from Backend and update db_state in memory."""
        new_data = fetch_and_save_embeddings(self.db_abs)
        
        if new_data is not None:
            new_embs, new_names = new_data
            with self.db_lock:
                self.db_state["embs"] = new_embs
                self.db_state["names"] = new_names
            print("[API] Veritabani basariyla guncellendi!\n")
        else:
            print("[API] Backend baglantisi kurulamadi veya veri alinamadi. Mevcut/Eski DB kullaniliyor.\n")

    def run(self) -> None:
        # Initial DB load
        try:
            init_embs, init_names = FaceRecognizer.load_db(self.db_abs)
            self.db_state["embs"] = init_embs
            self.db_state["names"] = init_names
        except Exception:
            print("[SISTEM] Lokal DB bulunamadi. Ilk senkronizasyon bekleniyor...")
            
        # Arka planda baslangic senkronizasyonu ve offline log gonderimi yap
        threading.Thread(target=self.fetch_and_reload_db, daemon=True).start()
        threading.Thread(target=sync_offline_logs, daemon=True).start()

        # Setup the API with dependencies
        setup_api(
            recognizer=self.recognizer,
            inference_lock=self.inference_lock,
            db_lock=self.db_lock,
            reload_callback=self.fetch_and_reload_db
        )

        # Start FastAPI in a background thread
        def start_uvicorn():
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

        api_thread = threading.Thread(target=start_uvicorn, daemon=True)
        api_thread.start()
        print("[SYSTEM] FastAPI sunucusu 0.0.0.0:8000 adresinde baslatildi.")

        if self.video is not None:
            video_abs = resolve_video_path(self.project_root, self.video)
            self._run_video(video_abs)
            return

        if self.hardware_env in ("MAC", "WIN"):
            self._run_mac()
        else:
            self._run_rpi()

    def _run_mac(self) -> None:
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
                    with self.inference_lock:
                        last_dets = self.detector.detect(frame)

                for det in last_dets:
                    roi = crop_with_padding(frame, det.bbox, config.MODEL_CONFIG.landmark_pad)
                    if roi is None:
                        continue

                    with self.inference_lock:
                        emb = self.recognizer.embed_from_roi(roi)
                    if emb is None:
                        continue

                    with self.db_lock:
                        curr_embs = self.db_state["embs"]
                        curr_names = self.db_state["names"]

                    name, score = FaceRecognizer.predict_identity(
                        emb=emb,
                        db_embs=curr_embs,
                        db_names=curr_names,
                        metric=self.metric,
                        threshold=self.threshold,
                    )

                    if name != "Unknown":
                        t_now = time.time()
                        if name not in self.last_seen or (t_now - self.last_seen[name]) > self.LOG_COOLDOWN_SECONDS:
                            self.last_seen[name] = t_now
                            threading.Thread(target=send_access_log, args=(name,), daemon=True).start()

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    label = f"{name} {score:.2f}" if self.metric == "cosine" else f"{name} {score:.3f}"

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

    def _run_rpi(self) -> None:
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
        start_rpi_preview(picam)

        t = threading.Thread(target=reader, args=(picam,), daemon=True)
        t.start()

        while latest_frame is None and running:
            time.sleep(0.05)

        print("Sistem Aktif! Pencereyi kapatmak icin 'q' tusuna basin (veya CTRL+C).\n")

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
                    with self.inference_lock:
                        last_dets = self.detector.detect(frame)

                for det in last_dets:
                    roi = crop_with_padding(frame, det.bbox, config.MODEL_CONFIG.landmark_pad)
                    if roi is None:
                        continue

                    with self.inference_lock:
                        emb = self.recognizer.embed_from_roi(roi)
                    if emb is None:
                        continue

                    with self.db_lock:
                        curr_embs = self.db_state["embs"]
                        curr_names = self.db_state["names"]

                    name, score = FaceRecognizer.predict_identity(
                        emb=emb,
                        db_embs=curr_embs,
                        db_names=curr_names,
                        metric=self.metric,
                        threshold=self.threshold,
                    )

                    if name != "Unknown":
                        print(f"[BASARILI] {name} tespit edildi! (Skor: {score:.3f})")
                        t_now = time.time()
                        if name not in self.last_seen or (t_now - self.last_seen[name]) > self.LOG_COOLDOWN_SECONDS:
                            self.last_seen[name] = t_now
                            threading.Thread(target=send_access_log, args=(name,), daemon=True).start()
                    else:
                        print(f"[HATA AYIKLAMA] Yuz algilandi ama eslesmedi. En yakin: {name} (Skor: {score:.3f})")
                        t_now = time.time()
                        if (
                            self.UNKNOWN_LOG_KEY not in self.last_seen
                            or (t_now - self.last_seen[self.UNKNOWN_LOG_KEY]) > self.LOG_COOLDOWN_SECONDS
                        ):
                            self.last_seen[self.UNKNOWN_LOG_KEY] = t_now
                            threading.Thread(target=send_unknown_access_log, args=(score,), daemon=True).start()

                    draw_face_label(frame, det.bbox, name, score, self.metric)

                now = time.time()
                if now - fps_t0 >= 1.0:
                    print(f"Anlik FPS: {fps_n / (now - fps_t0):.2f}")
                    fps_t0 = now
                    fps_n = 0

                if not show_frame(frame, WINDOW_TITLE, picam=picam):
                    running = False
                    break

                frame_counter += 1
        except KeyboardInterrupt:
            print("\n[BILGI] CTRL+C algilandi, sistem kapatiliyor...")
        finally:
            running = False
            t.join(timeout=1)
            stop_rpi_preview(picam)
            try:
                picam.stop()
            except Exception:
                pass
            cv2.destroyAllWindows()

    def _run_video(self, video_path: str) -> None:
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
                    with self.inference_lock:
                        dets = self.detector.detect(frame)
                else:
                    dets = []

                for det in dets:
                    roi = crop_with_padding(frame, det.bbox, config.MODEL_CONFIG.landmark_pad)
                    if roi is None:
                        continue

                    with self.inference_lock:
                        emb = self.recognizer.embed_from_roi(roi)
                    if emb is None:
                        continue

                    with self.db_lock:
                        curr_embs = self.db_state["embs"]
                        curr_names = self.db_state["names"]

                    name, score = FaceRecognizer.predict_identity(
                        emb=emb,
                        db_embs=curr_embs,
                        db_names=curr_names,
                        metric=self.metric,
                        threshold=self.threshold,
                    )
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    label = f"{name} {score:.2f}" if self.metric == "cosine" else f"{name} {score:.3f}"

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


def main() -> None:
    args = build_arg_parser().parse_args()
    system = LiveFaceRecognitionSystem(
        hardware_env=args.hardware_env,
        yolo_model_path=args.yolo_model_path,
        recognizer_model_name=args.recognizer_model_name,
        metric=args.metric,
        threshold_override=args.threshold,
        db_path=args.db_path,
        video=args.video,
    )
    system.run()


if __name__ == "__main__":
    main()
