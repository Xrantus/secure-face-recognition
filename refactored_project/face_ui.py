"""OpenCV / Picamera2 overlay helpers for live face recognition preview."""

from __future__ import annotations

import os
import sys

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


def show_frame_rpi(picam, frame: np.ndarray) -> None:
    """Push annotated frame to Picamera2 HDMI overlay."""
    cfg = picam.camera_configuration()
    main = cfg.get("main", {})
    target_w, target_h = main.get("size", (frame.shape[1], frame.shape[0]))

    h, w = frame.shape[:2]
    if (w, h) != (target_w, target_h):
        frame = cv2.resize(frame, (target_w, target_h))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    alpha = np.full((target_h, target_w, 1), 255, dtype=np.uint8)
    picam.set_overlay(np.concatenate([rgb, alpha], axis=2))


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
