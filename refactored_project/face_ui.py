"""OpenCV / Picamera2 overlay helpers for live face recognition preview."""

from __future__ import annotations

import cv2
import numpy as np

from .face_recognizer import SimilarityMetric

WINDOW_TITLE = "Face Recognition (YOLO + InsightFace)"

_gui_warned = False


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
    """Start HDMI preview on Raspberry Pi. Returns backend name or None."""
    try:
        from picamera2 import Preview
    except ImportError:
        print("[UI] Picamera2 Preview modulu bulunamadi.")
        return None

    for preview_cls, name in ((Preview.DRM, "DRM"), (Preview.QTGL, "QTGL")):
        try:
            picam.start_preview(preview_cls)
            print(
                f"[UI] RPi onizleme baslatildi ({name}). "
                "Goruntu Pi'ye bagli HDMI/monitorde acilir (SSH penceresinde degil)."
            )
            return name
        except Exception:
            continue

    print("[UI] UYARI: RPi onizlemesi baslatilamadi. HDMI kablosu ve masaustu oturumu kontrol edin.")
    return None


def show_frame_rpi(picam, frame: np.ndarray) -> None:
    """Push annotated frame to Picamera2 HDMI overlay."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    picam.set_overlay(np.concatenate([rgb, alpha], axis=2))


def stop_rpi_preview(picam) -> None:
    """Clear overlay and stop Picamera2 preview."""
    try:
        picam.set_overlay(None)
    except Exception:
        pass
    try:
        picam.stop_preview()
    except Exception:
        pass


def show_frame(
    frame: np.ndarray,
    window_title: str = WINDOW_TITLE,
    picam=None,
) -> bool:
    """
    Show annotated frame.
    On RPi pass `picam` for HDMI overlay; otherwise uses OpenCV window.
    Returns False if user pressed 'q' (OpenCV mode only).
    """
    global _gui_warned

    if picam is not None:
        try:
            show_frame_rpi(picam, frame)
            return True
        except Exception as exc:
            if not _gui_warned:
                print(f"[UI] RPi overlay hatasi: {exc}")
                _gui_warned = True

    try:
        cv2.imshow(window_title, frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")
    except cv2.error as exc:
        if not _gui_warned:
            print(
                f"[UI] UYARI: OpenCV penceresi acilamadi ({exc}). "
                "RPi'de HDMI monitor kullanin veya opencv-python-headless yerine opencv-python kurun."
            )
            _gui_warned = True
        return True
