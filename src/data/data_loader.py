"""
src/data/data_loader.py
-----------------------
Step 2: Build data loader.
Loads multi-channel physiological data from PhysioNet WFDB records or generates
high-fidelity synthetic benchmark streams matching the target wearable sensor BOM.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

try:
    import wfdb
except ImportError:
    wfdb = None

META_RE = re.compile(r"<([a-zA-Z0-9_]+)>:\s*([^\s<]+)")

CHANNEL_BOM_MAP = {
    'ecg': 'AD8232_ECG',
    'pleth_1': 'MAX30102_PPG',
    'a_x': 'MPU6050_ACC_X',
    'a_y': 'MPU6050_ACC_Y',
    'a_z': 'MPU6050_ACC_Z',
    'g_x': 'MPU6050_GYRO_X',
    'g_y': 'MPU6050_GYRO_Y',
    'g_z': 'MPU6050_GYRO_Z',
    'temp_1': 'TMP117_TEMP',
    'lc_1': 'STRETCH_CONTACT_PRESSURE',
    'gsr_synthetic': 'SYNTHETIC_GSR'
}

def parse_metadata(comments: List[str]) -> Dict[str, Any]:
    meta = {}
    for comment in comments:
        for key, val in META_RE.findall(comment):
            try:
                if '.' in val:
                    meta[key] = float(val)
                else:
                    meta[key] = int(val)
            except ValueError:
                meta[key] = val
    return meta

def synthesize_gsr(accel_energy: np.ndarray, fs: float, seed: int = 42) -> np.ndarray:
    """
    Generates a synthetic GSR (Galvanic Skin Response) channel in microsiemens (uS).
    Loosely models autonomic arousal driven by physical exertion + slow drift.
    Flagged clearly as SYNTHETIC.
    """
    rng = np.random.default_rng(seed)
    n = len(accel_energy)
    drift = np.cumsum(rng.normal(0, 0.001, n))
    kernel_len = max(1, int(fs * 2.0))
    kernel = np.ones(kernel_len) / kernel_len
    smoothed_motion = np.convolve(accel_energy, kernel, mode='same')
    max_motion = smoothed_motion.max() if smoothed_motion.max() > 1e-6 else 1.0
    activity_norm = smoothed_motion / max_motion
    baseline_us = 3.5
    noise = rng.normal(0, 0.02, n)
    gsr = baseline_us + drift + 2.0 * activity_norm + noise
    return np.clip(gsr, 0.5, 25.0)

def load_physionet_record(record_name: str, raw_dir: Optional[str] = None, sampto: Optional[int] = None) -> Dict[str, Any]:
    """
    Loads a record from local raw_dir or fetches directly from PhysioNet.
    """
    if wfdb is None:
        raise ImportError("wfdb is required to load PhysioNet records.")
    
    if raw_dir and (Path(raw_dir) / f"{record_name}.hea").exists():
        rec = wfdb.rdrecord(str(Path(raw_dir) / record_name), sampto=sampto)
    else:
        rec = wfdb.rdrecord(record_name, pn_dir='pulse-transit-time-ppg', sampto=sampto)
    
    fs = float(rec.fs)
    sig_names = rec.sig_name
    signals = rec.p_signal
    meta = parse_metadata(rec.comments)
    idx = {name: i for i, name in enumerate(sig_names)}
    n_samples = signals.shape[0]

    accel_cols = [c for c in ('a_x', 'a_y', 'a_z') if c in idx]
    if accel_cols:
        accel_energy = np.sqrt(np.sum([signals[:, idx[c]] ** 2 for c in accel_cols], axis=0))
    else:
        accel_energy = np.zeros(n_samples)

    seed_val = abs(hash(record_name)) % (2**31)
    gsr = synthesize_gsr(accel_energy, fs, seed=seed_val)

    channel_data = {
        'ecg': signals[:, idx['ecg']] if 'ecg' in idx else np.zeros(n_samples),
        'ppg': signals[:, idx['pleth_1']] if 'pleth_1' in idx else np.zeros(n_samples),
        'a_x': signals[:, idx['a_x']] if 'a_x' in idx else np.zeros(n_samples),
        'a_y': signals[:, idx['a_y']] if 'a_y' in idx else np.zeros(n_samples),
        'a_z': signals[:, idx['a_z']] if 'a_z' in idx else np.zeros(n_samples),
        'g_x': signals[:, idx['g_x']] if 'g_x' in idx else np.zeros(n_samples),
        'g_y': signals[:, idx['g_y']] if 'g_y' in idx else np.zeros(n_samples),
        'g_z': signals[:, idx['g_z']] if 'g_z' in idx else np.zeros(n_samples),
        'temp': signals[:, idx['temp_1']] if 'temp_1' in idx else np.full(n_samples, 36.5),
        'contact': signals[:, idx['lc_1']] if 'lc_1' in idx else np.ones(n_samples),
        'gsr_synthetic': gsr
    }

    return {
        'record_name': record_name,
        'fs': fs,
        'n_samples': n_samples,
        'duration_s': n_samples / fs,
        'meta': meta,
        'signals': channel_data
    }

def generate_benchmark_record(activity: str = 'sit', duration_s: float = 60.0, fs: float = 500.0, seed: int = 42) -> Dict[str, Any]:
    """
    High-fidelity physiological signal generator adhering to real PhysioNet morphology.
    Used for reliable offline benchmarking and controlled stress/anomaly testing.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    if activity == 'sit':
        target_hr = 68.0 + rng.normal(0, 2)
        motion_amp = 0.02
        temp_val = 36.6
        activity_label = 'sit'
    elif activity == 'walk':
        target_hr = 95.0 + rng.normal(0, 3)
        motion_amp = 1.2
        temp_val = 36.8
        activity_label = 'walk'
    elif activity == 'run':
        target_hr = 138.0 + rng.normal(0, 4)
        motion_amp = 2.8
        temp_val = 37.3
        activity_label = 'run'
    elif activity == 'stress_anomaly':
        target_hr = 126.0  # Elevated HR with ZERO motion (physiological strain)
        motion_amp = 0.02
        temp_val = 37.4
        activity_label = 'stress_anomaly'
    elif activity == 'lead_off':
        target_hr = 0.0
        motion_amp = 0.0
        temp_val = 22.0
        activity_label = 'lead_off'
    else:
        target_hr = 72.0
        motion_amp = 0.05
        temp_val = 36.6
        activity_label = activity

    if activity == 'lead_off':
        ecg_raw = np.full(n, 0.0) + rng.normal(0, 1.0, n)
        ppg_raw = np.full(n, 0.0) + rng.normal(0, 1.0, n)
        a_x = np.zeros(n)
        a_y = np.zeros(n)
        a_z = np.zeros(n)
        g_x = np.zeros(n)
        g_y = np.zeros(n)
        g_z = np.zeros(n)
        temp = np.full(n, 22.0)
        contact = np.zeros(n)
        gsr = np.zeros(n)
    else:
        # ECG generation
        beat_interval = 60.0 / target_hr
        beat_times = np.arange(0, duration_s, beat_interval)
        ecg = np.zeros(n)
        for bt in beat_times:
            b_idx = int(bt * fs)
            w = int(0.08 * fs)
            start = max(0, b_idx - w)
            end = min(n, b_idx + w)
            tt = (np.arange(start, end) - b_idx) / fs
            qrs = 1.5 * (1 - 1600 * (tt ** 2)) * np.exp(-800 * (tt ** 2))
            ecg[start:end] += qrs

        ecg_raw = 32500.0 + ecg * 1000.0 + rng.normal(0, 15.0, n)

        # PPG generation
        ppg = np.zeros(n)
        for bt in beat_times:
            b_idx = int((bt + 0.15) * fs)
            w = int(0.4 * fs)
            start = max(0, b_idx)
            end = min(n, b_idx + w)
            if end > start:
                phase = (np.arange(start, end) - b_idx) / w
                pulse = np.sin(np.pi * phase) ** 2 + 0.3 * np.sin(2 * np.pi * phase) ** 2
                ppg[start:end] += pulse
        ppg_raw = 72000.0 + ppg * 400.0 + rng.normal(0, 10.0, n)

        cadence_hz = target_hr / 60.0 if motion_amp > 0.1 else 0.5
        a_x = rng.normal(0, motion_amp, n) + motion_amp * np.sin(2 * np.pi * cadence_hz * t)
        a_y = rng.normal(0, motion_amp, n) + motion_amp * np.cos(2 * np.pi * cadence_hz * t)
        a_z = 1.0 + rng.normal(0, motion_amp * 0.5, n) + motion_amp * 0.5 * np.sin(4 * np.pi * cadence_hz * t)

        g_x = a_x * 45.0 + rng.normal(0, 2.0, n)
        g_y = a_y * 45.0 + rng.normal(0, 2.0, n)
        g_z = a_z * 20.0 + rng.normal(0, 2.0, n)

        temp = np.full(n, temp_val) + 0.05 * (t / duration_s) + rng.normal(0, 0.01, n)
        contact = np.full(n, 120.0) + rng.normal(0, 2.0, n)
        accel_energy = np.sqrt(a_x**2 + a_y**2 + a_z**2)
        gsr = synthesize_gsr(accel_energy, fs, seed=seed)

    meta = {
        'activity': activity_label,
        'gender': 'neutral',
        'age': 28,
        'height': 172,
        'weight': 68,
        'bp_sys_start': 118 if activity != 'run' else 135,
        'bp_dia_start': 78,
        'hr_1_start': int(target_hr),
        'spo2_start': 98
    }

    return {
        'record_name': f'benchmark_{activity}',
        'fs': fs,
        'n_samples': n,
        'duration_s': duration_s,
        'meta': meta,
        'signals': {
            'ecg': ecg_raw,
            'ppg': ppg_raw,
            'a_x': a_x,
            'a_y': a_y,
            'a_z': a_z,
            'g_x': g_x,
            'g_y': g_y,
            'g_z': g_z,
            'temp': temp,
            'contact': contact,
            'gsr_synthetic': gsr
        }
    }
