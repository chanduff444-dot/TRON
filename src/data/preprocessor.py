"""
src/data/preprocessor.py
------------------------
Step 3: Build preprocessing.
Real-time embedded-friendly signal conditioning:
- 0.5-40 Hz Butterworth bandpass for ECG (DC removal, QRS enhancement)
- 0.5-5.0 Hz Butterworth bandpass for PPG (DC baseline elimination, pulsatile isolation)
- Causal online IIR filtering with state retention (suitable for RTOS streaming)
- Motion vector magnitude computation
- Sliding window segmentation (5.0s window, configurable hop)
"""

import numpy as np
from scipy import signal
from typing import Dict, Any, Tuple, Optional

class SignalPreprocessor:
    def __init__(self, fs: float = 500.0, window_sec: float = 5.0, hop_sec: float = 1.0):
        self.fs = fs
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.window_len = int(window_sec * fs)
        self.hop_len = int(hop_sec * fs)

        # Design ECG bandpass: 0.5 Hz - 40 Hz
        nyq = 0.5 * fs
        low_ecg = 0.5 / nyq
        high_ecg = min(40.0 / nyq, 0.95)
        self.b_ecg, self.a_ecg = signal.butter(2, [low_ecg, high_ecg], btype='bandpass')
        self.zi_ecg = signal.lfilter_zi(self.b_ecg, self.a_ecg) * 0.0

        # Design PPG bandpass: 0.5 Hz - 5.0 Hz
        low_ppg = 0.5 / nyq
        high_ppg = min(5.0 / nyq, 0.95)
        self.b_ppg, self.a_ppg = signal.butter(2, [low_ppg, high_ppg], btype='bandpass')
        self.zi_ppg = signal.lfilter_zi(self.b_ppg, self.a_ppg) * 0.0

    def filter_batch(self, ecg_raw: np.ndarray, ppg_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Zero-phase offline/batch filtering using filtfilt."""
        # Detrend first
        ecg_detrend = ecg_raw - np.mean(ecg_raw)
        ppg_detrend = ppg_raw - np.mean(ppg_raw)

        if len(ecg_detrend) > 30:
            ecg_filt = signal.filtfilt(self.b_ecg, self.a_ecg, ecg_detrend)
        else:
            ecg_filt = ecg_detrend

        if len(ppg_detrend) > 30:
            ppg_filt = signal.filtfilt(self.b_ppg, self.a_ppg, ppg_detrend)
        else:
            ppg_filt = ppg_detrend

        return ecg_filt, ppg_filt

    def filter_online_chunk(self, ecg_chunk: np.ndarray, ppg_chunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Causal forward filter for RTOS real-time streaming chunks."""
        ecg_filt, self.zi_ecg = signal.lfilter(self.b_ecg, self.a_ecg, ecg_chunk, zi=self.zi_ecg)
        ppg_filt, self.zi_ppg = signal.lfilter(self.b_ppg, self.a_ppg, ppg_chunk, zi=self.zi_ppg)
        return ecg_filt, ppg_filt

    def compute_motion_features(self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> Dict[str, np.ndarray]:
        """Computes acceleration magnitude and jerk."""
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        diff_mag = np.abs(np.diff(mag, prepend=mag[0]))
        return {
            'motion_mag': mag,
            'motion_jerk': diff_mag
        }

    def segment_windows(self, signals_dict: Dict[str, np.ndarray], meta: Dict[str, Any]):
        """
        Yields filtered 5.0-second sliding windows with metadata and time bounds.
        """
        n_samples = len(signals_dict['ecg'])
        if n_samples < self.window_len:
            return

        # Pre-filter ECG and PPG
        ecg_filt, ppg_filt = self.filter_batch(signals_dict['ecg'], signals_dict['ppg'])

        for start_idx in range(0, n_samples - self.window_len + 1, self.hop_len):
            end_idx = start_idx + self.window_len
            window = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time_s': start_idx / self.fs,
                'end_time_s': end_idx / self.fs,
                'fs': self.fs,
                'meta': meta,
                'signals': {
                    'ecg_filt': ecg_filt[start_idx:end_idx],
                    'ppg_filt': ppg_filt[start_idx:end_idx],
                    'ecg_raw': signals_dict['ecg'][start_idx:end_idx],
                    'ppg_raw': signals_dict['ppg'][start_idx:end_idx],
                    'a_x': signals_dict['a_x'][start_idx:end_idx],
                    'a_y': signals_dict['a_y'][start_idx:end_idx],
                    'a_z': signals_dict['a_z'][start_idx:end_idx],
                    'g_x': signals_dict['g_x'][start_idx:end_idx],
                    'g_y': signals_dict['g_y'][start_idx:end_idx],
                    'g_z': signals_dict['g_z'][start_idx:end_idx],
                    'temp': signals_dict['temp'][start_idx:end_idx],
                    'contact': signals_dict['contact'][start_idx:end_idx],
                    'gsr_synthetic': signals_dict['gsr_synthetic'][start_idx:end_idx],
                }
            }
            yield window
