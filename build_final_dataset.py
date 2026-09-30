"""
build_final_dataset.py
-----------------------
End-to-end pipeline: raw PhysioNet "Pulse Transit Time PPG Dataset" WFDB
records -> synthetic GSR -> windowed, per-sensor features -> one final
labeled CSV ready for model training.

This is the next stage after ptt_ppg_extract.py. You can run this script
directly on the raw WFDB folder (it re-reads records itself); you don't
need to run ptt_ppg_extract.py first, though doing so first is useful for
sanity-checking the raw signals.

USAGE
-----
    pip install wfdb pandas numpy scipy
    python build_final_dataset.py --raw_dir ptt_ppg_raw --out_csv final_dataset.csv

WHAT IT DOES, STAGE BY STAGE
-----------------------------
1. Reads every record's signals + header metadata (age, gender, height,
   weight, activity, bp/hr/spo2 at start & end).
2. Adds a SYNTHETIC GSR channel (there is no real GSR sensor in this
   dataset). It's built from activity intensity (accel energy) + a slow
   random-walk drift + noise, loosely mimicking how skin conductance
   tracks arousal/exertion. It is clearly marked synthetic in the output
   — never present this as measured data.
3. Splits each recording into fixed-length windows (default 5 s) and
   computes features per sensor group per window:
     - ECG:      estimated heart rate (peak detection), RR-interval std (HRV proxy)
     - PPG:      estimated pulse rate, pulse amplitude (from pleth_1)
     - Accel:    per-axis mean/std, overall motion energy
     - Gyro:     per-axis mean/std
     - Temp:     mean, linear trend (slope) across the window
     - Contact pressure (lc_1/2): mean, std
     - Synthetic GSR: mean, std
4. Attaches record-level metadata to every window, and interpolates the
   start/end BP, HR and SpO2 values across the recording's duration so
   each window gets a time-appropriate label rather than only a single
   record-level value.
5. Writes one row per window to --out_csv.

This produces a real dataset (given the real raw files) -- I can't
generate the numbers myself without the raw data, but this script will
do it in one pass once you have them.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError:
    sys.exit("Missing dependency. Run: pip install wfdb pandas numpy scipy")

try:
    from scipy.signal import find_peaks
except ImportError:
    sys.exit("Missing dependency. Run: pip install scipy")

META_RE = re.compile(r"<([a-zA-Z0-9_]+)>:\s*([^\s<]+)")

WINDOW_SEC_DEFAULT = 5.0


# ---------------------------------------------------------------- helpers

def parse_metadata(record) -> dict:
    meta = {}
    for comment in getattr(record, "comments", []):
        for key, val in META_RE.findall(comment):
            meta[key] = val
    return meta


def synthesize_gsr(accel_energy: np.ndarray, fs: float, seed: int) -> np.ndarray:
    """
    Very simple synthetic GSR generator, purely for pipeline development:
        baseline + slow random-walk drift + small noise
        + a scaled, low-pass version of motion energy (activity -> arousal proxy)
    NOT a substitute for a real GSR sensor. Flag as synthetic downstream.
    """
    rng = np.random.default_rng(seed)
    n = len(accel_energy)

    # slow random walk (skin conductance drifts slowly)
    drift = np.cumsum(rng.normal(0, 0.002, n))

    # low-pass the motion energy so it doesn't jitter sample-to-sample
    kernel_len = max(1, int(fs * 2))  # ~2 second smoothing
    kernel = np.ones(kernel_len) / kernel_len
    activity_component = np.convolve(accel_energy, kernel, mode="same")
    activity_component = activity_component / (activity_component.max() + 1e-9)

    noise = rng.normal(0, 0.01, n)

    baseline = 2.0  # arbitrary microsiemens-like baseline
    gsr = baseline + drift + 1.5 * activity_component + noise
    return gsr


def estimate_rate_bpm(signal: np.ndarray, fs: float, min_hz=0.6, max_hz=3.5):
    """Rough peak-based rate estimate (heart or pulse rate), returns (bpm, rr_std_s)."""
    if len(signal) < fs:  # window too short
        return np.nan, np.nan
    sig = signal - np.mean(signal)
    min_distance = int(fs / max_hz)
    peaks, _ = find_peaks(sig, distance=max(1, min_distance))
    if len(peaks) < 2:
        return np.nan, np.nan
    rr_intervals = np.diff(peaks) / fs  # seconds between beats
    valid = rr_intervals[(rr_intervals > 1 / max_hz) & (rr_intervals < 1 / min_hz)]
    if len(valid) < 1:
        return np.nan, np.nan
    bpm = 60.0 / np.mean(valid)
    rr_std = np.std(valid)
    return bpm, rr_std


def linear_trend(signal: np.ndarray) -> float:
    if len(signal) < 2:
        return 0.0
    x = np.arange(len(signal))
    slope, _ = np.polyfit(x, signal, 1)
    return float(slope)


def interpolate_label(start_val, end_val, frac):
    try:
        start_val, end_val = float(start_val), float(end_val)
    except (TypeError, ValueError):
        return np.nan
    return start_val + (end_val - start_val) * frac


# ------------------------------------------------------------ per-record

def process_record(raw_dir: Path, record_name: str, window_sec: float, seed: int):
    record = wfdb.rdrecord(str(raw_dir / record_name))
    fs = record.fs
    sig_names = record.sig_name
    signals = record.p_signal  # (n_samples, n_channels)
    meta = parse_metadata(record)

    idx = {name: i for i, name in enumerate(sig_names)}
    n_samples = signals.shape[0]
    duration_s = n_samples / fs

    accel_cols = [c for c in ("a_x", "a_y", "a_z") if c in idx]
    accel_energy = None
    if accel_cols:
        accel_energy = np.sqrt(sum(signals[:, idx[c]] ** 2 for c in accel_cols))
    else:
        accel_energy = np.zeros(n_samples)

    gsr_synthetic = synthesize_gsr(accel_energy, fs, seed=seed)

    window_len = int(window_sec * fs)
    n_windows = n_samples // window_len

    subject_id = record_name.split("_")[0]
    activity = meta.get("activity", record_name.split("_", 1)[-1] if "_" in record_name else "unknown")

    rows = []
    for w in range(n_windows):
        s, e = w * window_len, (w + 1) * window_len
        frac = ((s + e) / 2) / n_samples  # position of window center, 0..1

        row = {
            "record_id": record_name,
            "subject_id": subject_id,
            "window_index": w,
            "window_start_s": s / fs,
            "activity": activity,
            "gender": meta.get("gender"),
            "age": meta.get("age"),
            "height_cm": meta.get("height"),
            "weight_kg": meta.get("weight"),
        }

        # ECG -> heart rate + HRV proxy
        if "ecg" in idx:
            hr_bpm, rr_std = estimate_rate_bpm(signals[s:e, idx["ecg"]], fs, min_hz=0.6, max_hz=3.5)
            row["ecg_hr_bpm_est"] = hr_bpm
            row["ecg_rr_std_s"] = rr_std

        # PPG (site 1) -> pulse rate + amplitude
        if "pleth_1" in idx:
            seg = signals[s:e, idx["pleth_1"]]
            pr_bpm, _ = estimate_rate_bpm(seg, fs, min_hz=0.6, max_hz=3.5)
            row["ppg_pulse_bpm_est"] = pr_bpm
            row["ppg_amplitude"] = float(np.ptp(seg)) if len(seg) else np.nan

        # Accel -> per-axis stats + motion energy
        for c in accel_cols:
            seg = signals[s:e, idx[c]]
            row[f"accel_{c[-1]}_mean"] = float(np.mean(seg))
            row[f"accel_{c[-1]}_std"] = float(np.std(seg))
        row["motion_energy_mean"] = float(np.mean(accel_energy[s:e])) if accel_cols else np.nan

        # Gyro -> per-axis stats
        for c in ("g_x", "g_y", "g_z"):
            if c in idx:
                seg = signals[s:e, idx[c]]
                row[f"gyro_{c[-1]}_mean"] = float(np.mean(seg))
                row[f"gyro_{c[-1]}_std"] = float(np.std(seg))

        # Temperature -> mean + trend, per available site
        for c in ("temp_1", "temp_2", "temp_3"):
            if c in idx:
                seg = signals[s:e, idx[c]]
                row[f"{c}_mean_C"] = float(np.mean(seg))
                row[f"{c}_trend"] = linear_trend(seg)

        # Contact/attachment pressure -> stand-in for stretch sensor
        for c in ("lc_1", "lc_2"):
            if c in idx:
                seg = signals[s:e, idx[c]]
                row[f"contact_pressure_{c[-1]}_mean"] = float(np.mean(seg))
                row[f"contact_pressure_{c[-1]}_std"] = float(np.std(seg))

        # Synthetic GSR
        seg = gsr_synthetic[s:e]
        row["gsr_synthetic_mean"] = float(np.mean(seg))
        row["gsr_synthetic_std"] = float(np.std(seg))
        row["gsr_is_synthetic"] = True

        # Interpolated labels from record-level start/end metadata
        row["label_bp_sys"] = interpolate_label(meta.get("bp_sys_start"), meta.get("bp_sys_end"), frac)
        row["label_bp_dia"] = interpolate_label(meta.get("bp_dia_start"), meta.get("bp_dia_end"), frac)
        row["label_hr_ref"] = interpolate_label(meta.get("hr_1_start"), meta.get("hr_1_end"), frac)
        row["label_spo2"] = interpolate_label(meta.get("spo2_start"), meta.get("spo2_end"), frac)

        rows.append(row)

    return rows


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", required=True, help="Folder with the downloaded .hea/.dat files")
    ap.add_argument("--out_csv", required=True, help="Path to write the final labeled dataset CSV")
    ap.add_argument("--window_sec", type=float, default=WINDOW_SEC_DEFAULT, help="Window length in seconds (default 5.0)")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    records = sorted(p.stem for p in raw_dir.glob("*.hea"))
    if not records:
        sys.exit(f"No .hea files found in {raw_dir}. Did the download finish?")

    print(f"Found {len(records)} records. Building windowed feature dataset...")

    all_rows = []
    for i, name in enumerate(records):
        try:
            rows = process_record(raw_dir, name, args.window_sec, seed=i)
            all_rows.extend(rows)
            print(f"  {name}: {len(rows)} windows")
        except Exception as e:
            print(f"  SKIPPED {name}: {e}", file=sys.stderr)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {len(df)} windowed rows across {df['record_id'].nunique()} records to {args.out_csv}")
    print(f"Subjects: {df['subject_id'].nunique()}, activities: {sorted(df['activity'].dropna().unique())}")


if __name__ == "__main__":
    main()
