"""
src/features/feature_extractor.py
---------------------------------
Step 4: Build feature extraction.
Computes lightweight time-domain and frequency-proxy physiological features
from preprocessed windows. Designed for low computational complexity on microcontrollers.
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Dict, Any

class FeatureExtractor:
    def __init__(self, fs: float = 500.0):
        self.fs = fs

    def extract_ecg_features(self, ecg_filt: np.ndarray) -> Dict[str, float]:
        """Extracts heart rate and HRV proxies from filtered ECG."""
        if len(ecg_filt) < int(self.fs):
            return {'ecg_hr_bpm': np.nan, 'ecg_rr_std_ms': np.nan, 'ecg_rmssd_ms': np.nan, 'ecg_qrs_energy': 0.0}

        # Adaptive peak detection
        sig = ecg_filt - np.mean(ecg_filt)
        threshold = np.std(sig) * 1.2
        min_dist = int(self.fs * 0.3)  # Max ~200 bpm
        peaks, _ = find_peaks(sig, height=threshold, distance=min_dist)

        if len(peaks) < 2:
            return {'ecg_hr_bpm': np.nan, 'ecg_rr_std_ms': np.nan, 'ecg_rmssd_ms': np.nan, 'ecg_qrs_energy': float(np.mean(sig**2))}

        rr_intervals_s = np.diff(peaks) / self.fs
        # Filter physiological valid intervals (0.3s to 1.8s -> 33 to 200 bpm)
        valid_rr = rr_intervals_s[(rr_intervals_s >= 0.3) & (rr_intervals_s <= 1.8)]

        if len(valid_rr) < 1:
            return {'ecg_hr_bpm': np.nan, 'ecg_rr_std_ms': np.nan, 'ecg_rmssd_ms': np.nan, 'ecg_qrs_energy': float(np.mean(sig**2))}

        mean_rr = float(np.mean(valid_rr))
        hr_bpm = 60.0 / mean_rr if mean_rr > 0 else np.nan
        rr_std_ms = float(np.std(valid_rr) * 1000.0)

        if len(valid_rr) >= 2:
            rmssd_ms = float(np.sqrt(np.mean(np.diff(valid_rr * 1000.0)**2)))
        else:
            rmssd_ms = rr_std_ms

        return {
            'ecg_hr_bpm': round(hr_bpm, 1),
            'ecg_rr_std_ms': round(rr_std_ms, 1),
            'ecg_rmssd_ms': round(rmssd_ms, 1),
            'ecg_qrs_energy': float(np.mean(sig**2))
        }

    def extract_ppg_features(self, ppg_filt: np.ndarray, ppg_raw: np.ndarray) -> Dict[str, float]:
        """Extracts pulse rate, pulsatile amplitude, and perfusion index proxy."""
        if len(ppg_filt) < int(self.fs):
            return {'ppg_pulse_rate_bpm': np.nan, 'ppg_amplitude': np.nan, 'ppg_perfusion_index': np.nan}

        sig = ppg_filt - np.mean(ppg_filt)
        min_dist = int(self.fs * 0.3)
        peaks, _ = find_peaks(sig, distance=min_dist)

        if len(peaks) < 2:
            amp = float(np.ptp(sig)) if len(sig) else 0.0
            return {'ppg_pulse_rate_bpm': np.nan, 'ppg_amplitude': amp, 'ppg_perfusion_index': np.nan}

        pp_intervals_s = np.diff(peaks) / self.fs
        valid_pp = pp_intervals_s[(pp_intervals_s >= 0.3) & (pp_intervals_s <= 1.8)]

        if len(valid_pp) < 1:
            return {'ppg_pulse_rate_bpm': np.nan, 'ppg_amplitude': float(np.ptp(sig)), 'ppg_perfusion_index': np.nan}

        pr_bpm = 60.0 / float(np.mean(valid_pp))
        amp = float(np.ptp(sig))
        dc = float(np.mean(np.abs(ppg_raw))) + 1e-6
        pi = (amp / dc) * 100.0  # Perfusion index percentage proxy

        return {
            'ppg_pulse_rate_bpm': round(pr_bpm, 1),
            'ppg_amplitude': round(amp, 2),
            'ppg_perfusion_index': round(pi, 3)
        }

    def extract_imu_features(self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
                             gx: np.ndarray, gy: np.ndarray, gz: np.ndarray) -> Dict[str, float]:
        """Extracts 3-axis motion energy, SMA, and jerk."""
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        sma = float(np.mean(np.abs(ax) + np.abs(ay) + np.abs(az)))
        energy = float(np.mean(mag**2))
        jerk = np.diff(mag)
        jerk_std = float(np.std(jerk)) if len(jerk) else 0.0

        return {
            'accel_x_mean': round(float(np.mean(ax)), 3),
            'accel_y_mean': round(float(np.mean(ay)), 3),
            'accel_z_mean': round(float(np.mean(az)), 3),
            'accel_x_std': round(float(np.std(ax)), 3),
            'accel_y_std': round(float(np.std(ay)), 3),
            'accel_z_std': round(float(np.std(az)), 3),
            'motion_mag_mean': round(float(np.mean(mag)), 3),
            'motion_sma': round(sma, 3),
            'motion_energy': round(energy, 3),
            'motion_jerk_std': round(jerk_std, 4),
            'gyro_x_std': round(float(np.std(gx)), 3),
            'gyro_y_std': round(float(np.std(gy)), 3),
            'gyro_z_std': round(float(np.std(gz)), 3),
        }

    def extract_temp_features(self, temp: np.ndarray) -> Dict[str, float]:
        """Extracts skin temperature mean and linear trend (slope)."""
        if len(temp) < 2:
            return {'temp_mean_c': 36.5, 'temp_trend': 0.0}
        t_x = np.arange(len(temp)) / self.fs
        slope, _ = np.polyfit(t_x, temp, 1)  # deg C / sec
        return {
            'temp_mean_c': round(float(np.mean(temp)), 2),
            'temp_trend': round(float(slope * 60.0), 3)  # deg C / min
        }

    def extract_all(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts the complete feature dictionary from a preprocessed window."""
        sigs = window['signals']
        ecg_f = self.extract_ecg_features(sigs['ecg_filt'])
        ppg_f = self.extract_ppg_features(sigs['ppg_filt'], sigs['ppg_raw'])
        imu_f = self.extract_imu_features(
            sigs['a_x'], sigs['a_y'], sigs['a_z'],
            sigs['g_x'], sigs['g_y'], sigs['g_z']
        )
        temp_f = self.extract_temp_features(sigs['temp'])

        # Contact and GSR
        contact_mean = float(np.mean(sigs['contact']))
        gsr_mean = float(np.mean(sigs['gsr_synthetic']))
        gsr_std = float(np.std(sigs['gsr_synthetic']))

        # Consolidated best physiological HR estimate (fallback to PPG if ECG degraded)
        hr_est = ecg_f['ecg_hr_bpm']
        if np.isnan(hr_est) or hr_est < 30 or hr_est > 220:
            hr_est = ppg_f['ppg_pulse_rate_bpm']
        if np.isnan(hr_est):
            hr_est = 72.0  # Safe default baseline if lead off

        features = {
            'start_time_s': window.get('start_time_s', 0.0),
            'end_time_s': window.get('end_time_s', 0.0),
            'hr_bpm': round(hr_est, 1),
            **ecg_f,
            **ppg_f,
            **imu_f,
            **temp_f,
            'contact_mean': round(contact_mean, 2),
            'gsr_synthetic_mean_us': round(gsr_mean, 2),
            'gsr_synthetic_std_us': round(gsr_std, 3)
        }
        return features
