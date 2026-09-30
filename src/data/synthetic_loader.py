"""
src/data/synthetic_loader.py
-----------------------------
Adapter: maps the 1 Hz synthetic CSV (training_all.csv / demo_stream.csv)
into the feature-vector format expected by PhysiologicalStateDetector.

WHY THIS EXISTS
---------------
The existing SignalPreprocessor + FeatureExtractor pipeline was built for the
PhysioNet 500 Hz raw-waveform dataset (raw ECG mV, raw PPG intensity).
The synthetic CSV stores pre-computed 1 Hz physiological values (heart_rate bpm,
spo2 %, skin temperature °C, acc/gyro in g / deg/s).

This adapter bridges the two without touching either pipeline.

OUTPUT COLUMNS (match FEATURE_COLUMNS in state_detector.py)
------------------------------------------------------------
hr_bpm, ecg_rr_std_ms, ecg_rmssd_ms, ecg_qrs_energy,
ppg_pulse_rate_bpm, ppg_amplitude, ppg_perfusion_index,
accel_x_std, accel_y_std, accel_z_std,
motion_mag_mean, motion_sma, motion_energy, motion_jerk_std,
gyro_x_std, gyro_y_std, gyro_z_std,
temp_mean_c, temp_trend,
contact_mean, gsr_synthetic_mean_us,
state_label (int 0-3), subject_id

LABEL MAPPING
-------------
Synthetic CSV          ->  Detector 4-class scheme
NORMAL + REST          ->  0  RESTING
NORMAL + WALK/RUN      ->  1  ACTIVE_EXERTION
WARNING/CRITICAL + any ->  2  PHYSIOLOGICAL_STRAIN  (anomalous physiology)
FALL event             ->  3  MOTION_ARTIFACT        (impact + stillness)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Window length for computing rolling statistics from 1 Hz data (seconds = samples)
WINDOW_S = 30    # 30-sample rolling window (30 s at 1 Hz)
HOP_S    = 10    # 10-sample hop


# ---------------------------------------------------------------------------
# Label re-mapping
# ---------------------------------------------------------------------------
def _derive_label(state: str, event: str, activity: str) -> int:
    """
    Map synthetic CSV labels -> PhysiologicalStateDetector 4-class scheme.

    0  RESTING             – normal baseline, sedentary
    1  ACTIVE_EXERTION     – normal locomotion (walk / run)
    2  PHYSIOLOGICAL_STRAIN– warning/critical physiology
    3  MOTION_ARTIFACT     – fall impact
    """
    evt = str(event).upper()
    act = str(activity).upper()
    st  = str(state).upper()

    if "FALL" in evt:
        return 3   # motion artifact / impact
    if st in ("WARNING", "CRITICAL"):
        return 2   # physiological strain
    if act in ("WALK", "RUN"):
        return 1   # active exertion (normal HR elevation)
    return 0       # resting baseline


# ---------------------------------------------------------------------------
# Per-window feature computation
# ---------------------------------------------------------------------------
def _window_features(chunk: pd.DataFrame) -> dict:
    """
    Compute the FEATURE_COLUMNS vector from a 30-row (30 s) chunk of 1 Hz data.
    Columns that have no equivalent in the 1 Hz CSV are given safe defaults.
    """
    hr   = chunk["heart_rate"].values.astype(float)
    spo2 = chunk["spo2"].values.astype(float)
    temp = chunk["temperature"].values.astype(float)
    ax   = chunk["acc_x"].values.astype(float)
    ay   = chunk["acc_y"].values.astype(float)
    az   = chunk["acc_z"].values.astype(float)
    gx   = chunk["gyro_x"].values.astype(float)
    gy   = chunk["gyro_y"].values.astype(float)
    gz   = chunk["gyro_z"].values.astype(float)

    # --- HR-derived proxies (no raw ECG waveform available) ---
    hr_mean   = float(np.mean(hr))
    # Approximate RR interval from mean HR; std as HRV proxy
    rr_ms     = (60000.0 / np.clip(hr, 30, 220))          # ms
    rr_std_ms = float(np.std(rr_ms))
    rmssd_ms  = float(np.sqrt(np.mean(np.diff(rr_ms)**2))) if len(rr_ms) > 1 else rr_std_ms
    qrs_energy = 0.0   # not derivable from 1 Hz bpm

    # --- PPG-derived proxies (SpO2 as amplitude proxy) ---
    ppg_pr    = hr_mean          # pulse rate ≈ heart rate at 1 Hz
    ppg_amp   = float(np.std(spo2)) * 10.0   # pulsatile variation proxy
    ppg_pi    = float(np.mean(spo2)) / 100.0 * 5.0  # perfusion index proxy

    # --- IMU features ---
    mag       = np.sqrt(ax**2 + ay**2 + az**2)
    sma       = float(np.mean(np.abs(ax) + np.abs(ay) + np.abs(az)))
    energy    = float(np.mean(mag**2))
    jerk      = np.abs(np.diff(mag))
    jerk_std  = float(np.std(jerk)) if len(jerk) > 0 else 0.0

    # --- Temperature (skin) ---
    t_x       = np.arange(len(temp), dtype=float)
    slope, _  = np.polyfit(t_x, temp, 1) if len(temp) > 1 else (0.0, 0.0)
    temp_trend = float(slope * 60.0)   # deg C / min

    return {
        "hr_bpm":                 round(hr_mean, 1),
        "ecg_rr_std_ms":          round(rr_std_ms, 1),
        "ecg_rmssd_ms":           round(rmssd_ms, 1),
        "ecg_qrs_energy":         qrs_energy,
        "ppg_pulse_rate_bpm":     round(ppg_pr, 1),
        "ppg_amplitude":          round(ppg_amp, 3),
        "ppg_perfusion_index":    round(ppg_pi, 3),
        "accel_x_std":            round(float(np.std(ax)), 4),
        "accel_y_std":            round(float(np.std(ay)), 4),
        "accel_z_std":            round(float(np.std(az)), 4),
        "motion_mag_mean":        round(float(np.mean(mag)), 4),
        "motion_sma":             round(sma, 4),
        "motion_energy":          round(energy, 4),
        "motion_jerk_std":        round(jerk_std, 5),
        "gyro_x_std":             round(float(np.std(gx)), 3),
        "gyro_y_std":             round(float(np.std(gy)), 3),
        "gyro_z_std":             round(float(np.std(gz)), 3),
        "temp_mean_c":            round(float(np.mean(temp)), 2),
        "temp_trend":             round(temp_trend, 4),
        "contact_mean":           100.0,   # no contact sensor in 1 Hz CSV → safe default
        "gsr_synthetic_mean_us":  0.0,     # no GSR in 1 Hz CSV → safe default
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_synthetic_csv(
    csv_path: str | Path,
    window_s: int = WINDOW_S,
    hop_s: int = HOP_S,
) -> pd.DataFrame:
    """
    Load a synthetic CSV and return a DataFrame of windowed feature vectors
    compatible with PhysiologicalStateDetector.train_and_evaluate().

    Parameters
    ----------
    csv_path : path to training_all.csv or demo_stream.csv
    window_s : rolling window length in seconds (= samples at 1 Hz)
    hop_s    : hop between windows in seconds

    Returns
    -------
    DataFrame with columns: FEATURE_COLUMNS + [state_label, subject_id]
    """
    df = pd.read_csv(csv_path)

    required = {"heart_rate", "spo2", "temperature",
                "acc_x", "acc_y", "acc_z",
                "gyro_x", "gyro_y", "gyro_z",
                "activity", "state_label", "event_type", "subject_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    rows = []
    for sid, subj_df in df.groupby("subject_id"):
        subj_df = subj_df.reset_index(drop=True)
        n = len(subj_df)
        for start in range(0, n - window_s + 1, hop_s):
            chunk = subj_df.iloc[start : start + window_s]
            feat  = _window_features(chunk)

            # Majority label over the window
            labels_int = [
                _derive_label(r.state_label, r.event_type, r.activity)
                for r in chunk.itertuples(index=False)
            ]
            majority_label = int(pd.Series(labels_int).mode()[0])

            feat["state_label"] = majority_label
            feat["subject_id"]  = sid
            rows.append(feat)

    result = pd.DataFrame(rows)
    print(f"[synthetic_loader] {csv_path}: {len(df)} raw rows "
          f"-> {len(result)} windows  "
          f"(window={window_s}s, hop={hop_s}s)")
    label_counts = result["state_label"].value_counts().sort_index()
    labels = {0: "RESTING", 1: "ACTIVE_EXERTION", 2: "STRAIN", 3: "ARTIFACT"}
    for k, v in label_counts.items():
        print(f"  [{k}] {labels.get(k,'?'):22s}: {v:4d}  ({v/len(result)*100:.1f}%)")
    return result


def load_demo_stream(
    csv_path: str | Path,
) -> pd.DataFrame:
    """
    Load demo_stream.csv as a flat DataFrame with per-row feature vectors
    (window_s=1, hop_s=1) for real-time replay through the WebSocket server.
    Each row becomes one inference step.
    """
    return load_synthetic_csv(csv_path, window_s=1, hop_s=1)
