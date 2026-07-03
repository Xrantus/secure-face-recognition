"""HC-SR04 proximity trigger — gates inference when nobody is within range."""

from __future__ import annotations

import threading
import time

from .config import ProximityConfig


class _HCSR04:
    """HC-SR04 reader — direct lgpio for Pi 5 (instead of gpiozero DistanceSensor)."""

    def __init__(self, trigger_pin: int, echo_pin: int) -> None:
        import lgpio

        self._lgpio = lgpio
        self._chip = lgpio.gpiochip_open(0)
        self._trigger = trigger_pin
        self._echo = echo_pin
        lgpio.gpio_claim_output(self._chip, self._trigger, lgpio.SET_PULL_NONE)
        lgpio.gpio_claim_input(self._chip, self._echo, lgpio.SET_PULL_DOWN)

    def close(self) -> None:
        try:
            self._lgpio.gpiochip_close(self._chip)
        except Exception:
            pass

    def read_distance_m(self) -> float | None:
        lg = self._lgpio
        h = self._chip
        trigger = self._trigger
        echo = self._echo

        lg.gpio_write(h, trigger, 0)
        time.sleep(0.000002)
        lg.gpio_write(h, trigger, 1)
        time.sleep(0.00001)
        lg.gpio_write(h, trigger, 0)

        deadline = time.perf_counter() + 0.1
        while lg.gpio_read(h, echo) == 0:
            if time.perf_counter() > deadline:
                return None

        t0 = time.perf_counter()
        while lg.gpio_read(h, echo) == 1:
            if time.perf_counter() - t0 > 0.1:
                return None
        t1 = time.perf_counter()

        dist_m = (t1 - t0) * 343.0 / 2.0
        if dist_m < 0.02 or dist_m > 4.0:
            return None
        return dist_m


class ProximityTrigger:
    def __init__(self, cfg: ProximityConfig, *, force_active: bool = False) -> None:
        self._cfg = cfg
        self._force_active = force_active
        self._active = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._sensor: _HCSR04 | None = None
        self._error_streak = 0
        self._fallback_logged = False
        self._last_distance_cm: float | None = None

    def start(self) -> None:
        if self._force_active or not self._cfg.enabled:
            self._active = True
            print("[Proximity] Disabled; face detection constantly active.")
            return

        try:
            self._sensor = _HCSR04(self._cfg.trigger_pin, self._cfg.echo_pin)
        except ImportError:
            print("[Proximity] lgpio not found; run: pip install lgpio")
            self._handle_sensor_unavailable()
            return
        except Exception as exc:
            print(f"[Proximity] Sensor initialization failed: {exc}")
            self._sensor = None
            self._handle_sensor_unavailable()
            return

        print(
            f"[Proximity] Sensor opened via lgpio "
            f"(TRIG=GPIO{self._cfg.trigger_pin}, ECHO=GPIO{self._cfg.echo_pin})"
        )

        if not self._warmup_sensor():
            self._close_sensor()
            self._handle_sensor_unavailable()
            return

        print(
            f"[Proximity] Sensor ready. "
            f"Proximity threshold: <= {self._cfg.activate_cm:.0f} cm"
        )
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._close_sensor()

    def is_active(self) -> bool:
        return self._active

    @property
    def last_distance_cm(self) -> float | None:
        return self._last_distance_cm

    def _close_sensor(self) -> None:
        if self._sensor is not None:
            try:
                self._sensor.close()
            except Exception:
                pass
            self._sensor = None

    def _handle_sensor_unavailable(self) -> None:
        if self._cfg.fallback_on_error:
            self._active = True
            print(
                "[Proximity] Sensor unavailable; face detection constantly active "
                "(fallback_on_error=True)."
            )
        else:
            print(
                "[Proximity] Sensor unavailable; face detection disabled. "
                "Check wiring or use --no-proximity option."
            )

    def _warmup_sensor(self) -> bool:
        assert self._sensor is not None
        ok_reads = 0
        for attempt in range(self._cfg.warmup_attempts):
            time.sleep(self._cfg.min_read_interval_s)
            try:
                dist = self._sensor.read_distance_m()
            except Exception as exc:
                print(f"[Proximity] Warmup reading {attempt + 1}/{self._cfg.warmup_attempts}: {exc}")
                continue
            if dist is None:
                print(
                    f"[Proximity] Warmup reading {attempt + 1}/{self._cfg.warmup_attempts}: "
                    "no echo"
                )
                continue
            ok_reads += 1
            self._last_distance_cm = dist * 100.0
        return ok_reads >= 2

    def _enable_fallback(self) -> None:
        self._active = True
        if not self._fallback_logged:
            print(
                "[Proximity] Persistent sensor error; "
                "switching to constant face detection mode."
            )
            self._fallback_logged = True

    def _poll(self) -> None:
        activate_m = self._cfg.activate_cm / 100.0
        deactivate_m = self._cfg.deactivate_cm / 100.0
        was_active = self._active

        while self._running:
            try:
                dist = self._sensor.read_distance_m() if self._sensor else None
                if dist is None:
                    raise RuntimeError("no echo")

                self._error_streak = 0
                self._last_distance_cm = dist * 100.0

                if self._active:
                    if dist > deactivate_m:
                        self._active = False
                elif dist <= activate_m:
                    self._active = True

                if self._active != was_active:
                    state = "ACTIVE" if self._active else "inactive"
                    print(
                        f"[Proximity] Face detection {state} "
                        f"(distance: {self._last_distance_cm:.0f} cm)"
                    )
                    was_active = self._active

            except Exception as exc:
                self._error_streak += 1
                if self._error_streak <= 3 or self._error_streak % 20 == 0:
                    print(f"[Proximity] Read error ({self._error_streak}x): {exc}")
                if self._cfg.fallback_on_error and self._error_streak >= 10:
                    self._enable_fallback()

            time.sleep(self._cfg.poll_interval_s)
