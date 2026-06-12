"""HC-SR04 proximity trigger — gates inference when nobody is within range."""

from __future__ import annotations

import threading
import time

from .config import ProximityConfig


class ProximityTrigger:
    def __init__(self, cfg: ProximityConfig, *, force_active: bool = False) -> None:
        self._cfg = cfg
        self._force_active = force_active
        self._active = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._sensor = None
        self._error_streak = 0
        self._fallback_logged = False

    def start(self) -> None:
        if self._force_active or not self._cfg.enabled:
            self._active = True
            print("[Proximity] Devre disi; yuz tespiti surekli aktif.")
            return

        try:
            from gpiozero import DistanceSensor
        except ImportError:
            print("[Proximity] gpiozero bulunamadi; yuz tespiti surekli aktif.")
            self._active = True
            return

        self._sensor = DistanceSensor(
            echo=self._cfg.echo_pin,
            trigger=self._cfg.trigger_pin,
            max_distance=2.0,
        )

        try:
            _ = self._sensor.distance
        except Exception as exc:
            self._sensor.close()
            self._sensor = None
            self._active = True
            print(
                f"[Proximity] Sensor okunamadi ({exc}); "
                "yuz tespiti surekli aktif. (--no-proximity ile de acilabilir)"
            )
            return

        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._sensor is not None:
            self._sensor.close()
            self._sensor = None

    def is_active(self) -> bool:
        return self._active

    def _enable_fallback(self) -> None:
        self._active = True
        if not self._fallback_logged:
            print(
                "[Proximity] Sensor hatasi devam ediyor; "
                "yuz tespiti surekli aktif moda gecildi."
            )
            self._fallback_logged = True

    def _poll(self) -> None:
        activate_m = self._cfg.activate_cm / 100.0
        deactivate_m = self._cfg.deactivate_cm / 100.0

        while self._running:
            try:
                dist = self._sensor.distance
                self._error_streak = 0
                if self._active:
                    if dist > deactivate_m:
                        self._active = False
                elif dist <= activate_m:
                    self._active = True
            except Exception:
                self._error_streak += 1
                if self._cfg.fallback_on_error and self._error_streak >= 10:
                    self._enable_fallback()
            time.sleep(self._cfg.poll_interval_s)
