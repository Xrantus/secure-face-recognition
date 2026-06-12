"""HC-SR04 proximity trigger — gates inference when nobody is within range."""

from __future__ import annotations

import threading
import time

from .config import ProximityConfig

_pin_factory_configured = False


def _setup_pin_factory() -> None:
    """Raspberry Pi 5 icin gpiozero pin fabrikasini ayarla."""
    global _pin_factory_configured
    if _pin_factory_configured:
        return

    import gpiozero

    try:
        from gpiozero.pins.lgpio import LGPIOFactory

        gpiozero.Device.pin_factory = LGPIOFactory()
        print("[Proximity] GPIO pin fabrikasi: lgpio (Pi 5 uyumlu)")
    except ImportError:
        try:
            from gpiozero.pins.rpigpio import RPiGPIOFactory

            gpiozero.Device.pin_factory = RPiGPIOFactory()
            print("[Proximity] GPIO pin fabrikasi: RPi.GPIO")
        except ImportError:
            print(
                "[Proximity] UYARI: lgpio kurulu degil. "
                "Pi 5'te: pip install lgpio gpiozero"
            )

    _pin_factory_configured = True


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
        self._last_distance_cm: float | None = None

    def start(self) -> None:
        if self._force_active or not self._cfg.enabled:
            self._active = True
            print("[Proximity] Devre disi; yuz tespiti surekli aktif.")
            return

        try:
            from gpiozero import DistanceSensor
        except ImportError:
            print("[Proximity] gpiozero bulunamadi; pip install gpiozero lgpio")
            return

        _setup_pin_factory()

        self._sensor = DistanceSensor(
            echo=self._cfg.echo_pin,
            trigger=self._cfg.trigger_pin,
            max_distance=2.0,
            queue_len=5,
        )

        if not self._warmup_sensor():
            self._sensor.close()
            self._sensor = None
            if self._cfg.fallback_on_error:
                self._active = True
                print(
                    "[Proximity] Sensor hazir degil; yuz tespiti surekli aktif "
                    "(fallback_on_error=True)."
                )
            else:
                print(
                    "[Proximity] Sensor hazir degil; yuz tespiti kapali. "
                    "Kablolama / lgpio kontrol edin veya --no-proximity kullanin."
                )
            return

        print(
            f"[Proximity] Sensor hazir (TRIG=GPIO{self._cfg.trigger_pin}, "
            f"ECHO=GPIO{self._cfg.echo_pin}). "
            f"Aktif: <= {self._cfg.activate_cm:.0f} cm"
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

    @property
    def last_distance_cm(self) -> float | None:
        return self._last_distance_cm

    def _warmup_sensor(self) -> bool:
        """HC-SR04 ilk okumada hata verebilir; birkac deneme yap."""
        assert self._sensor is not None
        ok_reads = 0
        for attempt in range(self._cfg.warmup_attempts):
            try:
                time.sleep(self._cfg.min_read_interval_s)
                dist = float(self._sensor.distance)
                if 0.02 <= dist <= 2.0:
                    ok_reads += 1
                    self._last_distance_cm = dist * 100.0
            except Exception as exc:
                print(f"[Proximity] Isinma okumasi {attempt + 1}/{self._cfg.warmup_attempts}: {exc}")
        return ok_reads >= 2

    def _enable_fallback(self) -> None:
        self._active = True
        if not self._fallback_logged:
            print(
                "[Proximity] Surekli sensor hatasi; "
                "yuz tespiti surekli aktif moda gecildi."
            )
            self._fallback_logged = True

    def _poll(self) -> None:
        activate_m = self._cfg.activate_cm / 100.0
        deactivate_m = self._cfg.deactivate_cm / 100.0
        was_active = self._active

        while self._running:
            try:
                dist = float(self._sensor.distance)
                self._error_streak = 0
                self._last_distance_cm = dist * 100.0

                if self._active:
                    if dist > deactivate_m:
                        self._active = False
                elif dist <= activate_m:
                    self._active = True

                if self._active != was_active:
                    state = "AKTIF" if self._active else "pasif"
                    print(
                        f"[Proximity] Yuz tespiti {state} "
                        f"(mesafe: {self._last_distance_cm:.0f} cm)"
                    )
                    was_active = self._active

            except Exception as exc:
                self._error_streak += 1
                if self._error_streak <= 3 or self._error_streak % 20 == 0:
                    print(f"[Proximity] Okuma hatasi ({self._error_streak}x): {exc}")
                if self._cfg.fallback_on_error and self._error_streak >= 10:
                    self._enable_fallback()

            time.sleep(self._cfg.poll_interval_s)
