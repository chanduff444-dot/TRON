"""
src/streamer/demo_replay.py
---------------------------
Phase 3 — Demo Stream Replay

Reads demo_stream.csv row-by-row at 1 Hz (real wall-clock time) and
injects each row into the existing SensorStreamer callback chain so the
live dashboard at http://127.0.0.1:8005 shows the scripted NORMAL →
WARNING → CRITICAL timeline without any code changes to the server.

Scripted timeline (from generate_synthetic_dataset.py):
  00:00–01:02   NORMAL    — resting baseline
  01:03–01:40   WARNING   — LOW_SPO2 (SpO2 drops ~4.5 pp)
  01:41–04:59   NORMAL    — recovery
  05:00–05:45   WARNING   — HIGH_TEMPERATURE (skin temp +1.2 °C)
  05:46–06:39   NORMAL    — recovery
  06:40–07:09   CRITICAL  — FALL (acc_z spike → stillness)
  07:10–08:00   NORMAL    — post-event

Usage:
    # As a standalone script (replays once, prints state to console)
    python -m src.streamer.demo_replay

    # Programmatic use (integrates with the running server)
    replay = DemoReplay("demo_stream.csv", callbacks=[on_raw_sample])
    replay.start()
"""

import time
import threading
import math
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional
import pandas as pd
import numpy as np

# ── State label → numeric for alert engine ────────────────────────────────
_LABEL_NUM = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
_STATE_COLOR = {"NORMAL": "\033[32m", "WARNING": "\033[33m", "CRITICAL": "\033[31m"}
_RESET = "\033[0m"


class DemoReplay:
    """
    Replays demo_stream.csv at 1 Hz into the SensorStreamer callback chain.

    Each CSV row is converted to a packet dict that matches the format
    SensorStreamer already emits, so existing RTOS / alert / dashboard
    code receives it transparently.
    """

    def __init__(
        self,
        csv_path: str | Path = "demo_stream.csv",
        speed: float = 1.0,
        loop: bool = False,
        callbacks: Optional[List[Callable[[Dict[str, Any]], None]]] = None,
    ):
        self.csv_path = Path(csv_path)
        self.speed = max(0.1, speed)
        self.loop = loop
        self._callbacks: List[Callable] = list(callbacks or [])
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._df: Optional[pd.DataFrame] = None

    def register_callback(self, cb: Callable[[Dict[str, Any]], None]):
        self._callbacks.append(cb)

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        required = {"heart_rate", "spo2", "temperature",
                    "acc_x", "acc_y", "acc_z",
                    "gyro_x", "gyro_y", "gyro_z",
                    "state_label", "event_type", "activity"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"demo_stream.csv missing columns: {missing}")
        return df

    def _row_to_packet(self, row: pd.Series, elapsed_s: float) -> Dict[str, Any]:
        """Convert a 1 Hz CSV row to a SensorStreamer-compatible packet."""
        state = str(row["state_label"]).upper()
        event = str(row["event_type"])

        # Synthesise a plausible ECG/PPG signal from the bpm value
        # (the RTOS preprocessor expects raw-ish values; we keep it simple)
        hr = float(row["heart_rate"])
        ecg_raw = 2048.0 + 400.0 * math.sin(2 * math.pi * (hr / 60.0) * elapsed_s)
        ppg_raw = 71000.0 + 200.0 * math.sin(2 * math.pi * (hr / 60.0) * elapsed_s - 0.3)

        return {
            "timestamp_ms":   int(time.time() * 1000),
            "elapsed_s":      round(elapsed_s, 1),
            "scenario":       f"demo_{state.lower()}",
            "source":         "SYNTHETIC_DEMO",
            # 1 Hz synthetic columns
            "heart_rate":     hr,
            "spo2":           float(row["spo2"]),
            "temperature":    float(row["temperature"]),
            "acc_x":          float(row["acc_x"]),
            "acc_y":          float(row["acc_y"]),
            "acc_z":          float(row["acc_z"]),
            "gyro_x":         float(row["gyro_x"]),
            "gyro_y":         float(row["gyro_y"]),
            "gyro_z":         float(row["gyro_z"]),
            "activity":       str(row.get("activity", "REST")),
            "state_label":    state,
            "state_numeric":  _LABEL_NUM.get(state, 0),
            "event_type":     event,
            # Synthetic raw signals for RTOS pipeline
            "ecg_raw":        ecg_raw,
            "ppg_raw":        ppg_raw,
            "temp":           float(row["temperature"]),
            "a_x":            float(row["acc_x"]),
            "a_y":            float(row["acc_y"]),
            "a_z":            float(row["acc_z"]),
            "g_x":            float(row["gyro_x"]),
            "g_y":            float(row["gyro_y"]),
            "g_z":            float(row["gyro_z"]),
            "contact":        100.0,
            "gsr_synthetic":  0.0,
            "meta": {
                "subject_id": 99,
                "activity":   str(row.get("activity", "REST")),
                "source":     "SYNTHETIC_DEMO",
            },
        }

    def _run(self):
        df = self._load()
        n = len(df)
        interval = 1.0 / self.speed   # seconds between rows

        print(f"\n{'='*60}")
        print(f"  DEMO STREAM REPLAY  --  {self.csv_path.name}")
        print(f"  {n} rows  |  speed={self.speed}x  |  loop={self.loop}")
        print(f"  Dashboard -> http://127.0.0.1:8005")
        print(f"{'='*60}")

        while self._running:
            for i, row in df.iterrows():
                if not self._running:
                    break
                t0 = time.perf_counter()

                elapsed_s = float(i)
                packet = self._row_to_packet(row, elapsed_s)

                # Console progress bar every 10 rows
                if int(elapsed_s) % 10 == 0:
                    mm = int(elapsed_s) // 60
                    ss = int(elapsed_s) % 60
                    state = packet["state_label"]
                    color = _STATE_COLOR.get(state, "")
                    evt   = packet["event_type"] or "—"
                    print(f"  [{mm:02d}:{ss:02d}]  {color}{state:8s}{_RESET}"
                          f"  HR={packet['heart_rate']:5.1f}  "
                          f"SpO2={packet['spo2']:4.1f}%  "
                          f"Temp={packet['temperature']:5.2f}°C  "
                          f"Event={evt}")

                # Dispatch to all registered callbacks
                for cb in self._callbacks:
                    try:
                        cb(packet)
                    except Exception as e:
                        print(f"[DemoReplay] callback error: {e}")

                # Pace to wall-clock 1 Hz (adjusted for speed)
                elapsed = time.perf_counter() - t0
                sleep = interval - elapsed
                if sleep > 0:
                    time.sleep(sleep)

            if not self.loop:
                break
            print("[DemoReplay] Looping...")

        print("[DemoReplay] Finished.")
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="DemoReplayThread"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def join(self):
        """Block until replay finishes (use for standalone mode)."""
        if self._thread:
            self._thread.join()


# ── Standalone entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    def _console_cb(pkt: Dict[str, Any]):
        pass   # progress already printed in _run()

    replay = DemoReplay(
        csv_path="demo_stream.csv",
        speed=1.0,
        loop=False,
        callbacks=[_console_cb],
    )
    replay.start()
    replay.join()
