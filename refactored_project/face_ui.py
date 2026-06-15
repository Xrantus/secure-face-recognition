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


def get_linux_screen_resolution() -> tuple[int, int] | None:
    """Read Linux DRM/KMS connector modes or fb virtual_size to detect display resolution."""
    # 1. Try sysfs DRM connector modes (most reliable on Pi 4/5 under KMS/DRM)
    drm_path = "/sys/class/drm"
    if os.path.isdir(drm_path):
        try:
            for conn in os.listdir(drm_path):
                if any(x in conn for x in ("HDMI", "DSI", "eDP")):
                    status_file = os.path.join(drm_path, conn, "status")
                    modes_file = os.path.join(drm_path, conn, "modes")
                    if os.path.isfile(status_file) and os.path.isfile(modes_file):
                        with open(status_file, "r") as f:
                            status = f.read().strip()
                        if status == "connected":
                            with open(modes_file, "r") as f:
                                modes = f.read().splitlines()
                            if modes:
                                parts = modes[0].split("x")
                                if len(parts) == 2:
                                    return int(parts[0]), int(parts[1])
        except Exception:
            pass

    # 2. Try framebuffer virtual size
    fb_path = "/sys/class/graphics/fb0/virtual_size"
    if os.path.isfile(fb_path):
        try:
            with open(fb_path, "r") as f:
                val = f.read().strip()
            parts = val.split(",")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        except Exception:
            pass

    # 3. Try xrandr if X11 is running
    if os.environ.get("DISPLAY"):
        try:
            import subprocess
            out = subprocess.check_output(["xrandr"], stderr=subprocess.DEVNULL).decode()
            for line in out.splitlines():
                if "current" in line:
                    parts = line.split("current")
                    if len(parts) > 1:
                        subparts = parts[1].split(",")
                        res = subparts[0].strip().split("x")
                        if len(res) == 2:
                            return int(res[0].strip()), int(res[1].strip())
        except Exception:
            pass

    return None


def get_screen_resolution() -> tuple[int, int] | None:
    """Helper to detect display/screen resolution across platforms."""
    # Try Linux DRM/KMS/fb first (useful for headless RPi overlays)
    res = get_linux_screen_resolution()
    if res:
        return res

    # Try Tkinter on other platforms (Mac/Windows/X11 Linux)
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass

    return None


def start_rpi_preview(picam, width: int = 1280, height: int = 720) -> str | None:
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
            # Pass width and height to start the preview in custom size matching screen
            picam.start_preview(preview_cls, x=0, y=0, width=width, height=height)
            _rpi_preview_mode = name
            print(
                f"[UI] RPi onizleme baslatildi ({name}) @ {width}x{height}. "
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


def init_display(picam=None, window_title: str = WINDOW_TITLE, width: int = 1280, height: int = 720) -> str:
    """
    Detect available display backend once at startup.
    Returns "rpi", "opencv", or "headless".
    """
    global _display_mode, _gui_warned

    if picam is not None and start_rpi_preview(picam, width=width, height=height):
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
        """Render the fully responsive dashboard composite image."""
        # Create empty dark canvas (BGR)
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:, :] = (22, 18, 18)  # BGR background: Dark gray/slate

        # Define layout scale factor relative to base 1280x720
        scale_factor = min(self.width / 1280.0, self.height / 720.0)

        # Margins to account for TV overscan and add elegant spacing (top margin as requested)
        margin_top = int(self.height * 0.04)
        margin_left = int(self.width * 0.02)
        margin_right = int(self.width * 0.02)
        margin_bottom = int(self.height * 0.03)

        # 1. Header Bar: float within the width margin
        header_h = int(self.height * 0.08)
        header_y1 = margin_top
        header_y2 = header_y1 + header_h
        header_x1 = margin_left
        header_x2 = self.width - margin_right

        cv2.rectangle(canvas, (header_x1, header_y1), (header_x2, header_y2), (38, 30, 30), -1)  # Lighter header BGR
        # Accent Line below header
        cv2.line(canvas, (header_x1, header_y2), (header_x2, header_y2), (255, 191, 0), 2)  # Glowing Cyan

        # Header Title
        title_font_scale = 0.75 * scale_factor
        title_thickness = max(1, int(2 * scale_factor))
        cv2.putText(
            canvas,
            "AI SURVEILLANCE & ACCESS MONITOR",
            (header_x1 + int(20 * scale_factor), header_y1 + int(header_h * 0.62)),
            cv2.FONT_HERSHEY_SIMPLEX,
            title_font_scale,
            (255, 255, 255),
            title_thickness,
            cv2.LINE_AA
        )

        # Header Right Information (Time & System Status)
        now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_txt = "ACTIVE" if proximity_active else "STANDBY"
        dist_txt = f"{proximity_dist:.0f}cm" if proximity_dist is not None else "--"
        info_txt = f"SYS: {status_txt} | DIST: {dist_txt} | FPS: {fps:.1f} | {now_time}"
        
        info_font_scale = 0.50 * scale_factor
        info_thickness = max(1, int(1 * scale_factor))
        info_x = header_x2 - int(600 * scale_factor)
        cv2.putText(
            canvas,
            info_txt,
            (info_x, header_y1 + int(header_h * 0.60)),
            cv2.FONT_HERSHEY_SIMPLEX,
            info_font_scale,
            (200, 200, 200),
            info_thickness,
            cv2.LINE_AA
        )

        # Pulsating heartbeat dot next to the clock
        pulse_r = max(2, int((5 + int(3 * np.abs(np.sin(time.time() * 3)))) * scale_factor))
        dot_color = (90, 220, 90) if proximity_active else (0, 165, 255) # Green / Amber BGR
        dot_x = header_x2 - int(35 * scale_factor)
        dot_y = header_y1 + header_h // 2
        # Outer glow
        cv2.circle(canvas, (dot_x, dot_y), pulse_r + max(1, int(3 * scale_factor)), dot_color, 1, cv2.LINE_AA)
        # Inner dot
        cv2.circle(canvas, (dot_x, dot_y), pulse_r, dot_color, -1, cv2.LINE_AA)

        # Gaps between panes
        gap_y = int(self.height * 0.02)
        gap_x = int(self.width * 0.015)

        content_y1 = header_y2 + gap_y
        content_y2 = self.height - margin_bottom
        content_h = content_y2 - content_y1
        content_w = header_x2 - header_x1

        # 2. Left Pane: Camera Stream (letterboxed and scaled)
        cam_w = int(content_w * 0.65)
        cam_h = content_h
        cam_x = header_x1
        cam_y = content_y1
        
        # Fit camera frame using letterbox
        fitted_cam = self._resize_letterbox(camera_frame, cam_w, cam_h)
        canvas[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = fitted_cam

        # Camera frame bezel/border
        cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + cam_w, cam_y + cam_h), (255, 191, 0), 2)
        # Label in the top-left of the camera window
        label_w = int(140 * scale_factor)
        label_h = int(25 * scale_factor)
        cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + label_w, cam_y + label_h), (255, 191, 0), -1)
        
        feed_font_scale = 0.45 * scale_factor
        feed_thickness = max(1, int(1 * scale_factor))
        cv2.putText(
            canvas,
            "CAMERA FEED",
            (cam_x + int(10 * scale_factor), cam_y + int(18 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            feed_font_scale,
            (0, 0, 0),
            feed_thickness,
            cv2.LINE_AA
        )

        # Right Pane components
        right_x = cam_x + cam_w + gap_x
        right_w = header_x2 - right_x
        
        card_h = (content_h - gap_y) // 2
        card_w = right_w

        title_card_scale = 0.55 * scale_factor
        title_card_thickness = max(1, int(2 * scale_factor))
        details_font_scale = 0.45 * scale_factor
        details_thickness = max(1, int(1 * scale_factor))

        with self.lock:
            # 3. Top-Right Panel: Last Recognition / Face Details
            card1_y = cam_y
            
            # Draw card background & border
            cv2.rectangle(canvas, (right_x, card1_y), (right_x + card_w, card1_y + card_h), (38, 30, 30), -1)
            cv2.rectangle(canvas, (right_x, card1_y), (right_x + card_w, card1_y + card_h), (60, 50, 50), 1)
            
            # Card Title
            cv2.putText(
                canvas,
                "LAST RECOGNIZED PROFILE",
                (right_x + int(20 * scale_factor), card1_y + int(35 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX,
                title_card_scale,
                (255, 255, 255),
                title_card_thickness,
                cv2.LINE_AA
            )
            cv2.line(canvas, 
                     (right_x + int(20 * scale_factor), card1_y + int(45 * scale_factor)), 
                     (right_x + card_w - int(20 * scale_factor), card1_y + int(45 * scale_factor)), 
                     (60, 50, 50), 1)

            # Crop dimensions: dynamically scaled relative to card height
            crop_size = int(card_h * 0.45)
            crop_x = right_x + int(20 * scale_factor)
            crop_y = card1_y + int(70 * scale_factor)

            if self.last_crop is not None:
                # Resize and place face crop
                resized_crop = cv2.resize(self.last_crop, (crop_size, crop_size))
                canvas[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size] = resized_crop
                
                # Corner brackets around face crop
                color_theme = (90, 220, 90) if self.last_status == "AUTHORIZED" else (90, 90, 220)
                l_len = int(15 * scale_factor)
                bracket_thickness = max(1, int(2 * scale_factor))
                # Top-left
                cv2.line(canvas, (crop_x, crop_y), (crop_x + l_len, crop_y), color_theme, bracket_thickness)
                cv2.line(canvas, (crop_x, crop_y), (crop_x, crop_y + l_len), color_theme, bracket_thickness)
                # Top-right
                cv2.line(canvas, (crop_x + crop_size, crop_y), (crop_x + crop_size - l_len, crop_y), color_theme, bracket_thickness)
                cv2.line(canvas, (crop_x + crop_size, crop_y), (crop_x + crop_size, crop_y + l_len), color_theme, bracket_thickness)
                # Bottom-left
                cv2.line(canvas, (crop_x, crop_y + crop_size), (crop_x + l_len, crop_y + crop_size), color_theme, bracket_thickness)
                cv2.line(canvas, (crop_x, crop_y + crop_size), (crop_x, crop_y + crop_size - l_len), color_theme, bracket_thickness)
                # Bottom-right
                cv2.line(canvas, (crop_x + crop_size, crop_y + crop_size), (crop_x + crop_size - l_len, crop_y + crop_size), color_theme, bracket_thickness)
                cv2.line(canvas, (crop_x + crop_size, crop_y + crop_size), (crop_x + crop_size, crop_y + crop_size - l_len), color_theme, bracket_thickness)
            else:
                # Draw scanning reticle background
                cv2.rectangle(canvas, (crop_x, crop_y), (crop_x + crop_size, crop_y + crop_size), (30, 24, 24), -1)
                
                # Draw corner brackets in standby amber BGR (0, 165, 255)
                color_theme = (0, 165, 255)
                l_len = int(12 * scale_factor)
                bracket_thickness = max(1, int(1 * scale_factor))
                cv2.line(canvas, (crop_x, crop_y), (crop_x + l_len, crop_y), color_theme, bracket_thickness)
                cv2.line(canvas, (crop_x, crop_y), (crop_x, crop_y + l_len), color_theme, bracket_thickness)
                cv2.line(canvas, (crop_x + crop_size, crop_y), (crop_x + crop_size - l_len, crop_y), color_theme, bracket_thickness)
                cv2.line(canvas, (crop_x + crop_size, crop_y), (crop_x + crop_size, crop_y + l_len), color_theme, bracket_thickness)
                cv2.line(canvas, (crop_x, crop_y + crop_size), (crop_x + l_len, crop_y + crop_size), color_theme, bracket_thickness)
                cv2.line(canvas, (crop_x, crop_y + crop_size), (crop_x, crop_y + crop_size - l_len), color_theme, bracket_thickness)
                cv2.line(canvas, (crop_x + crop_size, crop_y + crop_size), (crop_x + crop_size - l_len, crop_y + crop_size), color_theme, bracket_thickness)
                cv2.line(canvas, (crop_x + crop_size, crop_y + crop_size), (crop_x + crop_size, crop_y + crop_size - l_len), color_theme, bracket_thickness)
                
                # Ticking scanline inside reticle
                y_scan = crop_y + int(5 * scale_factor) + int((crop_size - int(10 * scale_factor)) * (0.5 + 0.5 * np.sin(time.time() * 4)))
                cv2.line(canvas, (crop_x + int(5 * scale_factor), y_scan), (crop_x + crop_size - int(5 * scale_factor), y_scan), (255, 255, 0), 1) # BGR Cyan
                
                # Text indicator
                scanning_scale = 0.4 * scale_factor
                cv2.putText(
                    canvas,
                    "SCANNING",
                    (crop_x + int(28 * scale_factor), crop_y + int(70 * scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scanning_scale,
                    (120, 120, 140),
                    1,
                    cv2.LINE_AA
                )

            # Details next to face crop (start after crop box)
            text_x = crop_x + crop_size + int(25 * scale_factor)
            
            # Name
            name_scale = 0.50 * scale_factor
            cv2.putText(
                canvas,
                f"NAME: {self.last_name}",
                (text_x, card1_y + int(90 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX,
                name_scale,
                (255, 255, 255),
                details_thickness,
                cv2.LINE_AA
            )
            
            # Status Badge Text
            status_color = (90, 220, 90) if self.last_status == "AUTHORIZED" else ((90, 90, 220) if self.last_status == "UNKNOWN" else (0, 165, 255))
            status_thickness = max(1, int(2 * scale_factor))
            cv2.putText(
                canvas,
                f"STATUS: {self.last_status}",
                (text_x, card1_y + int(125 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX,
                details_font_scale,
                status_color,
                status_thickness,
                cv2.LINE_AA
            )
            
            # Score
            score_str = f"SCORE: {self.last_score:.2f}" if self.last_status != "WAITING" else "SCORE: --"
            cv2.putText(
                canvas,
                score_str,
                (text_x, card1_y + int(160 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX,
                details_font_scale,
                (180, 180, 180),
                details_thickness,
                cv2.LINE_AA
            )
            
            # Match Time
            cv2.putText(
                canvas,
                f"TIME: {self.last_time}",
                (text_x, card1_y + int(195 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX,
                details_font_scale,
                (180, 180, 180),
                details_thickness,
                cv2.LINE_AA
            )

            # 4. Bottom-Right Panel: Access Log History
            card2_y = card1_y + card_h + gap_y
            
            # Draw card background & border
            cv2.rectangle(canvas, (right_x, card2_y), (right_x + card_w, card2_y + card_h), (38, 30, 30), -1)
            cv2.rectangle(canvas, (right_x, card2_y), (right_x + card_w, card2_y + card_h), (60, 50, 50), 1)

            # Card Title
            cv2.putText(
                canvas,
                "ACCESS LOG HISTORY",
                (right_x + int(20 * scale_factor), card2_y + int(35 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX,
                title_card_scale,
                (255, 255, 255),
                title_card_thickness,
                cv2.LINE_AA
            )
            cv2.line(canvas, 
                     (right_x + int(20 * scale_factor), card2_y + int(45 * scale_factor)), 
                     (right_x + card_w - int(20 * scale_factor), card2_y + int(45 * scale_factor)), 
                     (60, 50, 50), 1)

            # Draw logs (up to 5 entries)
            log_start_y = card2_y + int(80 * scale_factor)
            log_step_y = int(40 * scale_factor)
            
            for idx, log in enumerate(self.logs):
                y_offset = log_start_y + idx * log_step_y
                
                # Draw line separator between entries
                if idx > 0:
                    cv2.line(canvas, 
                             (right_x + int(20 * scale_factor), y_offset - int(25 * scale_factor)), 
                             (right_x + card_w - int(20 * scale_factor), y_offset - int(25 * scale_factor)), 
                             (50, 42, 42), 1)
                
                # Badge color
                badge_color = (90, 220, 90) if log["status"] == "AUTHORIZED" else (90, 90, 220)
                # Pulse outer circle
                cv2.circle(canvas, (right_x + int(30 * scale_factor), y_offset - int(5 * scale_factor)), int(6 * scale_factor), badge_color, -1, cv2.LINE_AA)
                
                # Timestamp
                cv2.putText(
                    canvas,
                    log["time"],
                    (right_x + int(55 * scale_factor), y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    details_font_scale,
                    (170, 170, 170),
                    details_thickness,
                    cv2.LINE_AA
                )
                
                # Name (e.g. User ID)
                cv2.putText(
                    canvas,
                    log["name"],
                    (right_x + int(140 * scale_factor), y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    details_font_scale,
                    (255, 255, 255),
                    details_thickness,
                    cv2.LINE_AA
                )
                
                # Status tag text
                status_lbl = "OK" if log["status"] == "AUTHORIZED" else "UNKNOWN"
                status_lbl_scale = 0.42 * scale_factor
                cv2.putText(
                    canvas,
                    status_lbl,
                    (right_x + card_w - int(90 * scale_factor), y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    status_lbl_scale,
                    badge_color,
                    details_thickness,
                    cv2.LINE_AA
                )
                
            # If logs list is empty
            if not self.logs:
                empty_msg_scale = 0.45 * scale_factor
                cv2.putText(
                    canvas,
                    "NO RECORDED LOGS YET",
                    (right_x + int(110 * scale_factor), card2_y + int(150 * scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    empty_msg_scale,
                    (100, 100, 120),
                    details_thickness,
                    cv2.LINE_AA
                )

        return canvas
