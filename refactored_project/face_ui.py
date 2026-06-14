"""OpenCV / Picamera2 overlay helpers for live face recognition preview."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from .face_recognizer import SimilarityMetric

WINDOW_TITLE = "Face Recognition (YOLO + InsightFace)"

_gui_warned = False
_rpi_preview_mode: str | None = None
_display_mode: str = "unknown"  # "rpi" | "opencv" | "headless"


def draw_face_label(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    name: str,
    score: float,
    metric: SimilarityMetric,
) -> None:
    """Draw bounding box and identity label on frame."""
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    label = f"{name} {score:.2f}" if metric == "cosine" else f"{name} {score:.3f}"
    x1, y1, x2, y2 = bbox
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


def start_rpi_preview(picam) -> str | None:
    """Start HDMI preview on Raspberry Pi. Returns backend name or None.

    Must be called after ``picam.configure()`` and **before** ``picam.start()``.
    """
    global _rpi_preview_mode

    try:
        from picamera2 import Preview
    except ImportError:
        print("[UI] Picamera2 Preview modulu bulunamadi.")
        _rpi_preview_mode = None
        return None

    # SSH oturumunda X11 yok; DRM dogrudan HDMI framebuffer kullanir.
    if _x_display_available():
        backends = ((Preview.QTGL, "QTGL"), (Preview.DRM, "DRM"))
    else:
        backends = ((Preview.DRM, "DRM"), (Preview.QTGL, "QTGL"))

    for preview_cls, name in backends:
        try:
            picam.start_preview(preview_cls)
            _rpi_preview_mode = name
            print(
                f"[UI] RPi onizleme baslatildi ({name}). "
                "Goruntu Pi'ye bagli HDMI/monitorde acilir (SSH penceresinde degil)."
            )
            return name
        except Exception as exc:
            print(f"[UI] {name} onizleme basarisiz: {exc}")

    _rpi_preview_mode = None
    print(
        "[UI] UYARI: HDMI onizleme acilamadi. "
        "Monitor HDMI'ye bagli mi? (SSH ile calistirmak sorun degil; "
        "onizleme Pi'nin bagli monitorunde acilir.)"
    )
    return None


def draw_status_hud(frame: np.ndarray, lines: list[str]) -> None:
    """Draw small status text on the top-left of the preview frame."""
    for i, line in enumerate(lines):
        y = 22 + i * 26
        cv2.putText(
            frame,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )


def show_frame_rpi(picam, frame: np.ndarray) -> None:
    """Push annotated frame to Picamera2 HDMI overlay."""
    cfg = picam.camera_configuration()
    target_w, target_h = frame.shape[1], frame.shape[0]
    for stream_name in ("main", "lores"):
        stream = cfg.get(stream_name, {})
        size = stream.get("size")
        if size:
            target_w, target_h = size
            break

    h, w = frame.shape[:2]
    if (w, h) != (target_w, target_h):
        frame = cv2.resize(frame, (target_w, target_h))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    alpha = np.full((target_h, target_w, 1), 255, dtype=np.uint8)
    overlay = np.ascontiguousarray(np.concatenate([rgb, alpha], axis=2))
    picam.set_overlay(overlay)


def stop_rpi_preview(picam) -> None:
    """Clear overlay and stop Picamera2 preview."""
    global _rpi_preview_mode, _display_mode

    try:
        picam.set_overlay(None)
    except Exception:
        pass
    try:
        picam.stop_preview()
    except Exception:
        pass
    _rpi_preview_mode = None
    _display_mode = "unknown"


def _x_display_available() -> bool:
    if sys.platform in ("win32", "darwin"):
        return True

    display = os.environ.get("DISPLAY", "").strip()
    if not display:
        return False
    if display.startswith(":"):
        try:
            num = display[1:].split(".")[0]
            return os.path.exists(f"/tmp/.X11-unix/X{num}")
        except (ValueError, IndexError):
            return False
    return True


def _probe_opencv_display(window_title: str) -> bool:
    if not _x_display_available():
        return False
    try:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(window_title)
        return True
    except cv2.error:
        return False


def init_display(picam=None, window_title: str = WINDOW_TITLE) -> str:
    """
    Detect available display backend once at startup.
    Returns "rpi", "opencv", or "headless".
    """
    global _display_mode, _gui_warned

    if picam is not None and start_rpi_preview(picam):
        _display_mode = "rpi"
        return _display_mode

    if _probe_opencv_display(window_title):
        _display_mode = "opencv"
        print("[UI] OpenCV onizleme aktif.")
        return _display_mode

    _display_mode = "headless"
    if not _gui_warned:
        print(
            "[UI] Ekran bulunamadi; headless modda devam ediliyor "
            "(yuz tanima calisir, onizleme yok — CTRL+C ile cikis)."
        )
        _gui_warned = True
    return _display_mode


def is_headless() -> bool:
    return _display_mode == "headless"


def _try_opencv_window(frame: np.ndarray, window_title: str) -> bool:
    """Show frame in OpenCV window. Returns False if user pressed 'q'."""
    global _gui_warned, _display_mode

    try:
        cv2.imshow(window_title, frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")
    except cv2.error as exc:
        _display_mode = "headless"
        if not _gui_warned:
            print(
                f"[UI] OpenCV penceresi kapandi ({exc}); headless moda gecildi."
            )
            _gui_warned = True
        return True


def show_frame(
    frame: np.ndarray,
    window_title: str = WINDOW_TITLE,
    picam=None,
) -> bool:
    """
    Show annotated frame when a display is available.
    On RPi with active preview uses HDMI overlay; otherwise tries OpenCV.
    In headless mode this is a no-op. Returns False if user pressed 'q' (OpenCV only).
    """
    global _gui_warned, _display_mode

    if _display_mode == "headless":
        return True

    if _display_mode == "rpi" and picam is not None:
        try:
            show_frame_rpi(picam, frame)
            return True
        except Exception as exc:
            _display_mode = "headless"
            if not _gui_warned:
                print(f"[UI] RPi overlay hatasi; headless moda gecildi: {exc}")
                _gui_warned = True
            return True

    if _display_mode == "opencv":
        return _try_opencv_window(frame, window_title)

    return True


class DashboardRenderer:
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        # Thread-safe lock for state updates
        import threading
        self.lock = threading.Lock()

        # Last detected face details
        self.last_name = "None"
        self.last_score = 0.0
        self.last_crop = None
        self.last_time = "--:--:--"
        self.last_status = "WAITING"

        # Log history (list of dicts)
        # Each dict: {"time": str, "name": str, "status": str, "score": float}
        self.logs = []

    def update_face(self, name: str, score: float, crop: np.ndarray | None, status: str) -> None:
        """Update details of the last detected person."""
        with self.lock:
            self.last_name = name
            self.last_score = score
            if crop is not None and crop.size > 0:
                self.last_crop = crop.copy()
            else:
                self.last_crop = None
            self.last_status = status
            self.last_time = datetime.now().strftime("%H:%M:%S")

    def add_log(self, name: str, status: str, score: float | None = None) -> None:
        """Add a log entry to the rolling log history (thread-safe)."""
        with self.lock:
            now_str = datetime.now().strftime("%H:%M:%S")
            log_entry = {
                "time": now_str,
                "name": name,
                "status": status,
                "score": score
            }
            self.logs.insert(0, log_entry)  # Add to top
            self.logs = self.logs[:5]  # Limit to 5 logs

    def _resize_letterbox(self, img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize image keeping aspect ratio, padding with dark background."""
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh))

        out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        # Fill background with a very dark BGR color
        out[:, :] = (22, 18, 18)
        dx = (target_w - nw) // 2
        dy = (target_h - nh) // 2
        out[dy:dy+nh, dx:dx+nw] = resized
        return out

    def render(
        self,
        camera_frame: np.ndarray,
        fps: float,
        proximity_active: bool,
        proximity_dist: float | None
    ) -> np.ndarray:
        """Render the full 1280x720 dashboard composite image."""
        # Create empty dark canvas (BGR)
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:, :] = (22, 18, 18)  # BGR background: Dark gray/slate

        # 1. Header Bar: X in [0, 1280], Y in [0, 70]
        cv2.rectangle(canvas, (0, 0), (self.width, 70), (38, 30, 30), -1)  # Lighter header BGR
        # Accent Line below header
        cv2.line(canvas, (0, 70), (self.width, 70), (255, 191, 0), 2)  # Glowing Cyan

        # Header Title
        cv2.putText(
            canvas,
            "AI SURVEILLANCE & ACCESS MONITOR",
            (20, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Header Right Information (Time & System Status)
        now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_txt = "ACTIVE" if proximity_active else "STANDBY"
        dist_txt = f"{proximity_dist:.0f}cm" if proximity_dist is not None else "--"
        info_txt = f"SYS: {status_txt} | DIST: {dist_txt} | FPS: {fps:.1f} | {now_time}"
        
        cv2.putText(
            canvas,
            info_txt,
            (self.width - 620, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        # Pulsating heartbeat dot next to the clock
        pulse_r = 5 + int(3 * np.abs(np.sin(time.time() * 3)))
        dot_color = (90, 220, 90) if proximity_active else (0, 165, 255) # Green / Amber BGR
        # Outer glow
        cv2.circle(canvas, (self.width - 35, 36), pulse_r + 3, dot_color, 1, cv2.LINE_AA)
        # Inner dot
        cv2.circle(canvas, (self.width - 35, 36), pulse_r, dot_color, -1, cv2.LINE_AA)

        # 2. Left Pane: Camera Stream (letterboxed into 800x600, positioned at [20, 90])
        cam_w, cam_h = 800, 600
        cam_x, cam_y = 20, 90
        
        # Fit camera frame using letterbox
        fitted_cam = self._resize_letterbox(camera_frame, cam_w, cam_h)
        canvas[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = fitted_cam

        # Camera frame bezel/border
        cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h), (255, 191, 0), 2)
        # Label in the top-left of the camera window
        cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + 140, cam_y + 25), (255, 191, 0), -1)
        cv2.putText(
            canvas,
            "CAMERA FEED",
            (cam_x + 10, cam_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        with self.lock:
            # 3. Top-Right Panel: Last Recognition / Face Details (X in [840, 1260], Y in [90, 380])
            card_x, card_y = 840, 90
            card_w, card_h = 420, 290
            
            # Draw card background & border
            cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (38, 30, 30), -1)
            cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (60, 50, 50), 1)
            
            # Card Title
            cv2.putText(
                canvas,
                "LAST RECOGNIZED PROFILE",
                (card_x + 20, card_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.line(canvas, (card_x + 20, card_y + 45), (card_x + card_w - 20, card_y + 45), (60, 50, 50), 1)

            # Crop dimensions: 130x130 at (card_x + 20, card_y + 70)
            crop_x, crop_y = card_x + 20, card_y + 70
            crop_size = 130

            if self.last_crop is not None:
                # Resize and place face crop
                resized_crop = cv2.resize(self.last_crop, (crop_size, crop_size))
                canvas[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size] = resized_crop
                
                # Corner brackets around face crop
                color_theme = (90, 220, 90) if self.last_status == "AUTHORIZED" else (90, 90, 220)
                # Draw corner brackets
                l_len = 15
                # Top-left
                cv2.line(canvas, (crop_x, crop_y), (crop_x + l_len, crop_y), color_theme, 2)
                cv2.line(canvas, (crop_x, crop_y), (crop_x, crop_y + l_len), color_theme, 2)
                # Top-right
                cv2.line(canvas, (crop_x + crop_size, crop_y), (crop_x + crop_size - l_len, crop_y), color_theme, 2)
                cv2.line(canvas, (crop_x + crop_size, crop_y), (crop_x + crop_size, crop_y + l_len), color_theme, 2)
                # Bottom-left
                cv2.line(canvas, (crop_x, crop_y + crop_size), (crop_x + l_len, crop_y + crop_size), color_theme, 2)
                cv2.line(canvas, (crop_x, crop_y + crop_size), (crop_x, crop_y + crop_size - l_len), color_theme, 2)
                # Bottom-right
                cv2.line(canvas, (crop_x + crop_size, crop_y + crop_size), (crop_x + crop_size - l_len, crop_y + crop_size), color_theme, 2)
                cv2.line(canvas, (crop_x + crop_size, crop_y + crop_size), (crop_x + crop_size, crop_y + crop_size - l_len), color_theme, 2)
            else:
                # Draw scanning reticle background
                cv2.rectangle(canvas, (crop_x, crop_y), (crop_x + crop_size, crop_y + crop_size), (30, 24, 24), -1)
                
                # Draw corner brackets in standby amber BGR (0, 165, 255)
                color_theme = (0, 165, 255)
                l_len = 12
                cv2.line(canvas, (crop_x, crop_y), (crop_x + l_len, crop_y), color_theme, 1)
                cv2.line(canvas, (crop_x, crop_y), (crop_x, crop_y + l_len), color_theme, 1)
                cv2.line(canvas, (crop_x + crop_size, crop_y), (crop_x + crop_size - l_len, crop_y), color_theme, 1)
                cv2.line(canvas, (crop_x + crop_size, crop_y), (crop_x + crop_size, crop_y + l_len), color_theme, 1)
                cv2.line(canvas, (crop_x, crop_y + crop_size), (crop_x + l_len, crop_y + crop_size), color_theme, 1)
                cv2.line(canvas, (crop_x, crop_y + crop_size), (crop_x, crop_y + crop_size - l_len), color_theme, 1)
                cv2.line(canvas, (crop_x + crop_size, crop_y + crop_size), (crop_x + crop_size - l_len, crop_y + crop_size), color_theme, 1)
                cv2.line(canvas, (crop_x + crop_size, crop_y + crop_size), (crop_x + crop_size, crop_y + crop_size - l_len), color_theme, 1)
                
                # Ticking scanline inside reticle
                y_scan = crop_y + 5 + int((crop_size - 10) * (0.5 + 0.5 * np.sin(time.time() * 4)))
                cv2.line(canvas, (crop_x + 5, y_scan), (crop_x + crop_size - 5, y_scan), (255, 255, 0), 1) # BGR Cyan
                
                # Text indicator
                cv2.putText(
                    canvas,
                    "SCANNING",
                    (crop_x + 28, crop_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (120, 120, 140),
                    1,
                    cv2.LINE_AA
                )

            # Details next to face crop (start at X = card_x + 175)
            text_x = card_x + 175
            
            # Name
            cv2.putText(
                canvas,
                f"NAME: {self.last_name}",
                (text_x, card_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            # Status Badge Text
            status_color = (90, 220, 90) if self.last_status == "AUTHORIZED" else ((90, 90, 220) if self.last_status == "UNKNOWN" else (0, 165, 255))
            cv2.putText(
                canvas,
                f"STATUS: {self.last_status}",
                (text_x, card_y + 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                status_color,
                2,
                cv2.LINE_AA
            )
            
            # Score
            score_str = f"SCORE: {self.last_score:.2f}" if self.last_status != "WAITING" else "SCORE: --"
            cv2.putText(
                canvas,
                score_str,
                (text_x, card_y + 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA
            )
            
            # Match Time
            cv2.putText(
                canvas,
                f"TIME: {self.last_time}",
                (text_x, card_y + 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA
            )

            # 4. Bottom-Right Panel: Access Log History (X in [840, 1260], Y in [400, 700])
            hist_x, hist_y = 840, 400
            hist_w, hist_h = 420, 290
            
            # Draw card background & border
            cv2.rectangle(canvas, (hist_x, hist_y), (hist_x + hist_w, hist_y + hist_h), (38, 30, 30), -1)
            cv2.rectangle(canvas, (hist_x, hist_y), (hist_x + hist_w, hist_y + hist_h), (60, 50, 50), 1)

            # Card Title
            cv2.putText(
                canvas,
                "ACCESS LOG HISTORY",
                (hist_x + 20, hist_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.line(canvas, (hist_x + 20, hist_y + 45), (hist_x + hist_w - 20, hist_y + 45), (60, 50, 50), 1)

            # Draw logs (up to 5 entries)
            for idx, log in enumerate(self.logs):
                y_offset = hist_y + 80 + idx * 40
                
                # Draw line separator between entries
                if idx > 0:
                    cv2.line(canvas, (hist_x + 20, y_offset - 25), (hist_x + hist_w - 20, y_offset - 25), (50, 42, 42), 1)
                
                # Badge color
                badge_color = (90, 220, 90) if log["status"] == "AUTHORIZED" else (90, 90, 220)
                # Pulse outer circle
                cv2.circle(canvas, (hist_x + 30, y_offset - 5), 6, badge_color, -1, cv2.LINE_AA)
                
                # Timestamp
                cv2.putText(
                    canvas,
                    log["time"],
                    (hist_x + 55, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (170, 170, 170),
                    1,
                    cv2.LINE_AA
                )
                
                # Name (e.g. User ID)
                cv2.putText(
                    canvas,
                    log["name"],
                    (hist_x + 140, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                
                # Status tag text
                status_lbl = "OK" if log["status"] == "AUTHORIZED" else "UNKNOWN"
                cv2.putText(
                    canvas,
                    status_lbl,
                    (hist_x + hist_w - 90, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    badge_color,
                    1,
                    cv2.LINE_AA
                )
                
            # If logs list is empty
            if not self.logs:
                cv2.putText(
                    canvas,
                    "NO RECORDED LOGS YET",
                    (hist_x + 110, hist_y + 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (100, 100, 120),
                    1,
                    cv2.LINE_AA
                )

        return canvas
