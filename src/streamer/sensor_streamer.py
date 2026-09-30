"""
src/streamer/sensor_streamer.py
-------------------------------
Step 6: Build real-time dataset streamer.
Emulates physical sensors (AD8232 ECG, MAX30102 PPG, MPU6050 IMU, TMP117 Temp)
streaming time-stamped multi-channel packets in real time over an async queue or callback.

Supports:
- Playback speed control (0.5x, 1.0x, 2.0x, 5.0x)
- Interactive scenario injection (Rest, Exertion, Stress Anomaly, Lead-Off)
- Downsampled telemetry packets (50 Hz / 100 Hz) to simulate microcontroller SPI/I2C packet bus.
- Drop-in abstraction interface ready to be substituted by physical ESP32 sensor hardware.
"""

import time
import threading
from typing import Dict, Any, Callable, Optional
import numpy as np
from src.data.data_loader import generate_benchmark_record, load_physionet_record

class SensorStreamer:
    def __init__(self, record_name: str = "benchmark_sit", speed: float = 1.0, emit_hz: float = 50.0):
        self.record_name = record_name
        self.speed = speed
        self.emit_hz = emit_hz
        self.sample_interval = 1.0 / emit_hz
        self.is_running = False
        self.is_paused = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable[[Dict[str, Any]], None]] = []

        # Current active scenario buffer
        self.current_scenario = "sit"
        self._load_active_scenario(self.current_scenario)

    def _load_active_scenario(self, scenario: str):
        self.current_scenario = scenario
        if scenario.startswith("physionet_"):
            p_name = scenario.replace("physionet_", "")
            rec = load_physionet_record(p_name, sampto=15000)
        else:
            rec = generate_benchmark_record(activity=scenario, duration_s=120.0, fs=500.0)

        self.native_fs = rec['fs']
        self.signals = rec['signals']
        self.meta = rec['meta']
        self.total_samples = rec['n_samples']
        self.sample_step = int(self.native_fs / self.emit_hz) # downsample ratio to emit_hz
        self.current_idx = 0

    def register_callback(self, cb: Callable[[Dict[str, Any]], None]):
        self._callbacks.append(cb)

    def set_speed(self, speed: float):
        self.speed = max(0.1, min(10.0, speed))

    def inject_scenario(self, scenario: str):
        """Dynamically switches the active physiological scenario."""
        print(f"[SensorStreamer] Injected scenario: {scenario}")
        self._load_active_scenario(scenario)

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.is_paused = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="SensorStreamerThread")
        self._thread.start()

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run_loop(self):
        while self.is_running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            start_t = time.perf_counter()

            # Slice current packet of samples
            idx = self.current_idx
            if idx >= self.total_samples - self.sample_step:
                self.current_idx = 0
                idx = 0

            # Construct hardware packet representing SPI/I2C batch
            packet = {
                'timestamp_ms': int(time.time() * 1000),
                'sample_idx': idx,
                'scenario': self.current_scenario,
                'ecg_raw': float(self.signals['ecg'][idx]),
                'ppg_raw': float(self.signals['ppg'][idx]),
                'a_x': float(self.signals['a_x'][idx]),
                'a_y': float(self.signals['a_y'][idx]),
                'a_z': float(self.signals['a_z'][idx]),
                'g_x': float(self.signals['g_x'][idx]),
                'g_y': float(self.signals['g_y'][idx]),
                'g_z': float(self.signals['g_z'][idx]),
                'temp': float(self.signals['temp'][idx]),
                'contact': float(self.signals['contact'][idx]),
                'gsr_synthetic': float(self.signals['gsr_synthetic'][idx]),
                'meta': self.meta
            }

            self.current_idx += self.sample_step

            # Dispatch to subscribers (RTOS Ingestion queue)
            for cb in self._callbacks:
                try:
                    cb(packet)
                except Exception as e:
                    print(f"[SensorStreamer] Callback error: {e}")

            # Pace according to speed
            elapsed = time.perf_counter() - start_t
            sleep_time = (self.sample_interval / self.speed) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
