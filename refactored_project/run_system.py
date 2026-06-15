"""Entrypoint for running the Live Camera Loop alongside the FastAPI Server.

This script starts the FastAPI server in a background thread and runs the 
camera loop (Mac/WIN or RPi) in the main thread.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import uvicorn

from . import config
from .access_log_policy import AccessLogPolicy, FaceObservation
from .api_server import app, setup_api
from .backend_client import fetch_and_save_embeddings, send_access_log, send_unknown_access_log, sync_offline_logs
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer, SimilarityMetric
from .distance_sensor import ProximityTrigger
from .face_ui import (
    WINDOW_TITLE,
    draw_face_label,
    draw_status_hud,
    init_display,
    is_headless,
    show_frame,
    stop_rpi_preview,
    DashboardRenderer,
    get_screen_resolution,
)
from .main import resolve_model_path, resolve_video_path, crop_with_padding, metric_threshold


def _is_raspberry_pi() -> bool:
    try:
        with open("/proc/device-tree/model", encoding="utf-8") as f:
            return "raspberry pi" in f.read().lower()
    except OSError:
        return False


def resolve_hardware_env(requested: Literal["MAC", "RPI", "WIN"], explicit: bool) -> Literal["MAC", "RPI", "WIN"]:
    """Pi'de --hardware-env verilmediyse otomatik Picamera2 modu."""
    if not explicit and _is_raspberry_pi():
        print("[SISTEM] Raspberry Pi algilandi -> Picamera2 (RPI) modu kullaniliyor.")
        return "RPI"
    return requested


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
        no_proximity: bool = False,
    ):
        self.hardware_env = hardware_env
        self.no_proximity = no_proximity
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
        # Determine screen resolution based on hardware environment
        if hardware_env == "RPI":
            screen_res = get_screen_resolution()
            if screen_res:
                self.screen_w, self.screen_h = screen_res
                print(f"[SISTEM] Raspberry Pi ekran cozunurlugu algilandi: {self.screen_w}x{self.screen_h}")
            else:
                self.screen_w, self.screen_h = (1920, 1080)
                print(f"[SISTEM] Raspberry Pi ekran cozunurlugu algilanamadi, varsayilan kullaniliyor: 1920x1080")
        else:
            self.screen_w, self.screen_h = (1280, 720)

        self.dashboard = DashboardRenderer(width=self.screen_w, height=self.screen_h)
        self.access_log_policy = self._make_access_log_policy()

    def _make_access_log_policy(self) -> AccessLogPolicy:
        def on_authorized(user_id_combined: str) -> None:
            # Parse user_id_combined which is "userId:userName"
            parts = user_id_combined.split(":")
            actual_id = parts[0]
            display_name = parts[1] if len(parts) > 1 else parts[0]

            print(f"[LOG] Gecis logu gonderiliyor: {display_name} (ID: {actual_id})")
            threading.Thread(target=send_access_log, args=(actual_id,), daemon=True).start()
            self.dashboard.add_log(display_name, "AUTHORIZED")

        def on_unknown(track_id: int, score: float | None) -> None:
            score_txt = f" (Skor: {score:.3f})" if score is not None else ""
            print(f"[UYARI] Taninmayan yuz track:{track_id}{score_txt} (log gonderiliyor)")
            threading.Thread(
                target=send_unknown_access_log,
                args=(score, track_id),
                daemon=True,
            ).start()
            self.dashboard.add_log("Unknown", "UNKNOWN", score)

        return AccessLogPolicy(on_authorized=on_authorized, on_unknown=on_unknown)

    def _recognize_observations(self, frame: np.ndarray, dets: list) -> list[FaceObservation]:
        observations: list[FaceObservation] = []

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
            if name != "Unknown":
                print(f"[BASARILI] {name} tespit edildi! (Skor: {score:.3f})")
            observations.append(FaceObservation(bbox=det.bbox, name=name, score=score, roi=roi))

        return observations

    @staticmethod
    def _draw_observations(frame: np.ndarray, observations: list[FaceObservation], metric: SimilarityMetric) -> None:
        for obs in observations:
            parts = obs.name.split(":")
            display_name = parts[1] if len(parts) > 1 else parts[0]
            draw_face_label(frame, obs.bbox, display_name, obs.score, metric)

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

        # Initialize display mode
        init_display(window_title=WINDOW_TITLE, width=self.screen_w, height=self.screen_h)

        print("Sistem Aktif! Penceriyi kapatmak icin 'q' tusuna basin.\n")

        frame_counter = 0
        last_dets: list = []
        last_observations: list[FaceObservation] = []
        fps_t0 = time.time()
        fps_n = 0
        display_fps = 0.0

        try:
            while running:
                with frame_lock:
                    if latest_frame is None:
                        continue
                    frame = latest_frame.copy()

                frame = cv2.flip(frame, 1)
                fps_n += 1

                ran_detection = frame_counter % config.MODEL_CONFIG.frame_skip == 0
                if ran_detection:
                    with self.inference_lock:
                        last_dets = self.detector.detect(frame)
                    last_observations = self._recognize_observations(frame, last_dets)
                    self.access_log_policy.update(last_observations, time.time())

                    # Update the dashboard last recognized face card
                    if last_observations:
                        best_obs = None
                        for obs in last_observations:
                            if obs.name != "Unknown":
                                best_obs = obs
                                break
                        if not best_obs:
                            best_obs = last_observations[0]
                        
                        # Split name for display
                        parts = best_obs.name.split(":")
                        display_name = parts[1] if len(parts) > 1 else parts[0]

                        self.dashboard.update_face(
                            name=display_name,
                            score=best_obs.score,
                            crop=best_obs.roi,
                            status="AUTHORIZED" if best_obs.name != "Unknown" else "UNKNOWN"
                        )

                self._draw_observations(frame, last_observations, self.metric)

                now = time.time()
                if now - fps_t0 >= 1.0:
                    display_fps = fps_n / (now - fps_t0)
                    print(f"Anlik FPS: {display_fps:.2f}")
                    fps_t0 = now
                    fps_n = 0

                # Render full dashboard frame
                dashboard_frame = self.dashboard.render(
                    camera_frame=frame,
                    fps=display_fps,
                    proximity_active=True,
                    proximity_dist=None
                )

                if not show_frame(dashboard_frame, WINDOW_TITLE):
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
        cfg = picam.create_preview_configuration({"size": (self.screen_w, self.screen_h)})
        picam.configure(cfg)
        # Preview, picam.start() oncesinde acilmali (aksi halde event loop catismasi).
        init_display(picam, width=self.screen_w, height=self.screen_h)
        picam.start()

        t = threading.Thread(target=reader, args=(picam,), daemon=True)
        t.start()

        while latest_frame is None and running:
            time.sleep(0.05)

        if latest_frame is None:
            raise SystemExit("Picamera2 akisi baslatilamadi. Kamera kablosu ve picamera2 kurulumunu kontrol edin.")

        if is_headless():
            print("Sistem Aktif! (Headless — durdurmak icin CTRL+C)\n")
        else:
            print("Sistem Aktif! Pencereyi kapatmak icin 'q' tusuna basin (veya CTRL+C).\n")

        proximity = ProximityTrigger(
            config.PROXIMITY_CONFIG,
            force_active=self.no_proximity,
        )
        try:
            proximity.start()
        except Exception as exc:
            print(f"[Proximity] Baslatma hatasi (sistem devam ediyor): {exc}")

        frame_counter = 0
        last_dets: list = []
        last_observations: list[FaceObservation] = []
        fps_t0 = time.time()
        fps_n = 0
        display_fps = 0.0

        try:
            while running:
                with frame_lock:
                    if latest_frame is None:
                        continue
                    frame = latest_frame.copy()

                frame = cv2.flip(frame, 1)
                fps_n += 1

                ran_detection = frame_counter % config.MODEL_CONFIG.frame_skip == 0
                if proximity.is_active():
                    if ran_detection:
                        with self.inference_lock:
                            last_dets = self.detector.detect(frame)
                        last_observations = self._recognize_observations(frame, last_dets)
                    self.access_log_policy.update(last_observations, time.time())

                    # Update the dashboard last recognized face card
                    if last_observations:
                        best_obs = None
                        for obs in last_observations:
                            if obs.name != "Unknown":
                                best_obs = obs
                                break
                        if not best_obs:
                            best_obs = last_observations[0]
                        
                        # Split name for display
                        parts = best_obs.name.split(":")
                        display_name = parts[1] if len(parts) > 1 else parts[0]

                        self.dashboard.update_face(
                            name=display_name,
                            score=best_obs.score,
                            crop=best_obs.roi,
                            status="AUTHORIZED" if best_obs.name != "Unknown" else "UNKNOWN"
                        )
                else:
                    last_dets = []
                    last_observations = []
                    if ran_detection:
                        self.access_log_policy.update([], time.time())

                self._draw_observations(frame, last_observations, self.metric)

                now = time.time()
                if now - fps_t0 >= 1.0:
                    display_fps = fps_n / (now - fps_t0)
                    print(f"Anlik FPS: {display_fps:.2f}")
                    fps_t0 = now
                    fps_n = 0

                # Render full dashboard frame
                dashboard_frame = self.dashboard.render(
                    camera_frame=frame,
                    fps=display_fps,
                    proximity_active=proximity.is_active(),
                    proximity_dist=proximity.last_distance_cm
                )

                if not show_frame(dashboard_frame, WINDOW_TITLE, picam=picam):
                    running = False
                    break

                frame_counter += 1
        except KeyboardInterrupt:
            print("\n[BILGI] CTRL+C algilandi, sistem kapatiliyor...")
        finally:
            running = False
            proximity.stop()
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

        # Initialize display mode
        init_display(window_title=WINDOW_TITLE, width=self.screen_w, height=self.screen_h)

        fps_t0 = time.time()
        fps_n = 0
        frame_counter = 0
        display_fps = 0.0
        last_observations: list[FaceObservation] = []

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                frame = cv2.flip(frame, 1)
                fps_n += 1

                ran_detection = frame_counter % config.MODEL_CONFIG.frame_skip == 0
                if ran_detection:
                    with self.inference_lock:
                        dets = self.detector.detect(frame)
                    last_observations = self._recognize_observations(frame, dets)
                    self.access_log_policy.update(last_observations, time.time())

                    # Update the dashboard last recognized face card
                    if last_observations:
                        best_obs = None
                        for obs in last_observations:
                            if obs.name != "Unknown":
                                best_obs = obs
                                break
                        if not best_obs:
                            best_obs = last_observations[0]
                        
                        # Split name for display
                        parts = best_obs.name.split(":")
                        display_name = parts[1] if len(parts) > 1 else parts[0]

                        self.dashboard.update_face(
                            name=display_name,
                            score=best_obs.score,
                            crop=best_obs.roi,
                            status="AUTHORIZED" if best_obs.name != "Unknown" else "UNKNOWN"
                        )

                self._draw_observations(frame, last_observations, self.metric)

                now = time.time()
                if now - fps_t0 >= 1.0:
                    display_fps = fps_n / (now - fps_t0)
                    print(f"Anlik FPS: {display_fps:.2f}")
                    fps_t0 = now
                    fps_n = 0

                # Render full dashboard frame
                dashboard_frame = self.dashboard.render(
                    camera_frame=frame,
                    fps=display_fps,
                    proximity_active=True,
                    proximity_dist=None
                )

                if not show_frame(dashboard_frame, WINDOW_TITLE):
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
    p.add_argument(
        "--no-proximity",
        action="store_true",
        help="HC-SR04 olmadan veya demo icin yuz tespitini surekli acik tut",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    hardware_env = resolve_hardware_env(args.hardware_env, explicit="--hardware-env" in sys.argv)
    system = LiveFaceRecognitionSystem(
        hardware_env=hardware_env,
        yolo_model_path=args.yolo_model_path,
        recognizer_model_name=args.recognizer_model_name,
        metric=args.metric,
        threshold_override=args.threshold,
        db_path=args.db_path,
        video=args.video,
        no_proximity=args.no_proximity,
    )
    system.run()


if __name__ == "__main__":
    main()
