"""HC-SR04 proximity trigger — gates inference when nobody is within range."""

from __future__ import annotations

import threading
import time

from .config import ProximityConfig


class ProximityTrigger:
    def __init__(self, cfg: ProximityConfig) -> None:
        self._cfg = cfg
        self._active = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._sensor = None

    def start(self) -> None:
        try:
            from gpiozero import DistanceSensor
        except ImportError:
            print("[Proximity] gpiozero bulunamadi; inference surekli aktif.")
            self._active = True
            return

        self._sensor = DistanceSensor(
            echo=self._cfg.echo_pin,
            trigger=self._cfg.trigger_pin,
            max_distance=2.0,
        )
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

    def _poll(self) -> None:
        activate_m = self._cfg.activate_cm / 100.0
        deactivate_m = self._cfg.deactivate_cm / 100.0

        while self._running:
            try:
                dist = self._sensor.distance
                if self._active:
                    if dist > deactivate_m:
                        self._active = False
                elif dist <= activate_m:
                    self._active = True
            except Exception:
                pass
            time.sleep(self._cfg.poll_interval_s)
