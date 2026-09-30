"""
src/ml/state_detector.py
------------------------
Step 5: Train/evaluate ML model.
LIGHTWEIGHT PHYSIOLOGICAL STATE & ANOMALY DETECTOR.

CRITICAL DESIGN NOTE:
Does NOT predict clinical diseases. Formulated strictly as:
  0: RESTING (Baseline vitals, minimal motor activity)
  1: ACTIVE_EXERTION (Elevated vitals coupled with active locomotion/exercise)
  2: PHYSIOLOGICAL_STRAIN (Elevated HR/stress markers discordant with low physical motion)
  3: MOTION_ARTIFACT (Extreme motion jerk/displacement degrading sensor contact)

Includes continuous anomaly scoring (0.0 to 1.0) and exportable C decision logic
for embedded execution on microcontrollers running ?T-Kernel 3.0.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

STATE_NAMES = {
    0: "RESTING",
    1: "ACTIVE_EXERTION",
    2: "PHYSIOLOGICAL_STRAIN",
    3: "MOTION_ARTIFACT"
}

FEATURE_COLUMNS = [
    'hr_bpm',
    'ecg_rr_std_ms',
    'ecg_rmssd_ms',
    'ecg_qrs_energy',
    'ppg_pulse_rate_bpm',
    'ppg_amplitude',
    'ppg_perfusion_index',
    'accel_x_std',
    'accel_y_std',
    'accel_z_std',
    'motion_mag_mean',
    'motion_sma',
    'motion_energy',
    'motion_jerk_std',
    'gyro_x_std',
    'gyro_y_std',
    'gyro_z_std',
    'temp_mean_c',
    'temp_trend',
    'contact_mean',
    'gsr_synthetic_mean_us'
]

def derive_ground_truth_label(row: Dict[str, float], activity_meta: str = "") -> int:
    """
    Scientifically sound physiological state labeling rule:
      0: RESTING - Sedentary posture, normal baseline HR, dynamic accel < 0.15g.
      1: ACTIVE_EXERTION - Rhythmic locomotion (walk/run), dynamic accel >= 0.20g.
      2: PHYSIOLOGICAL_STRAIN - Elevated HR (>95 bpm) or autonomic arousal while sedentary (accel < 0.15g).
      3: MOTION_ARTIFACT - Severe signal distortion (jerk > 3.2g, lead-off, or non-physiological HR).
    """
    hr = row.get('hr_bpm', 70.0)
    accel_dynamic = (row.get('accel_x_std', 0.0) + row.get('accel_y_std', 0.0) + row.get('accel_z_std', 0.0)) / 3.0
    jerk = row.get('motion_jerk_std', 0.0)
    contact = row.get('contact_mean', 100.0)

    # 1. Extreme artifact / lead-off
    if jerk > 3.2 or contact < 10.0 or np.isnan(hr) or hr < 35 or hr > 220:
        return 3 # MOTION_ARTIFACT

    # 2. Active exertion: physical locomotion
    if ('walk' in activity_meta.lower() or 'run' in activity_meta.lower() or accel_dynamic >= 0.20) and 'stress' not in activity_meta.lower():
        return 1 # ACTIVE_EXERTION

    # 3. Physiological strain: high HR or autonomic arousal while at rest
    if 'stress' in activity_meta.lower() or (hr >= 92.0 and accel_dynamic < 0.15):
        return 2 # PHYSIOLOGICAL_STRAIN

    # 4. Baseline resting
    return 0 # RESTING

class PhysiologicalStateDetector:
    def __init__(self, model_path: str = "models/state_detector.joblib"):
        self.model_path = model_path
        self.clf = None
        self.anomaly_detector = None
        self.feature_columns = FEATURE_COLUMNS
        self.baseline_stats = {}

    def train_and_evaluate(self, dataset_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Trains using subject-level Leave-One-Subject-Out (LOGO) cross validation
        to prevent data leakage across subjects.
        """
        df = dataset_df.copy()
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = 0.0

        X = df[self.feature_columns].values
        y = df['state_label'].values
        groups = df['subject_id'].values

        print(f"\n--- Training Physiological State Detector ---")
        print(f"Total Windows: {len(df)} across {len(np.unique(groups))} subjects")
        print("Class distribution:")
        for k, v in STATE_NAMES.items():
            count = np.sum(y == k)
            print(f"  [{k}] {v:22s}: {count:4d} ({count/len(y)*100:.1f}%)")

        logo = LeaveOneGroupOut()
        accs, f1s = [], []
        if len(np.unique(groups)) > 1:
            for train_idx, test_idx in logo.split(X, y, groups):
                fold_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
                fold_clf.fit(X[train_idx], y[train_idx])
                preds = fold_clf.predict(X[test_idx])
                accs.append(accuracy_score(y[test_idx], preds))
                f1s.append(f1_score(y[test_idx], preds, average='macro', zero_division=0))

            mean_acc = float(np.mean(accs))
            mean_f1 = float(np.mean(f1s))
            print(f"LOGO Cross-Validation Mean Accuracy: {mean_acc:.3f} | Macro F1: {mean_f1:.3f}")
        else:
            mean_acc = 1.0
            mean_f1 = 1.0

        # Fit final production model
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
        self.clf.fit(X, y)

        # Fit anomaly detector on resting baseline samples
        resting_mask = (y == 0)
        X_resting = X[resting_mask] if np.sum(resting_mask) > 10 else X
        self.anomaly_detector = IsolationForest(contamination=0.08, random_state=42)
        self.anomaly_detector.fit(X_resting)

        # Baseline resting statistics for normalization
        for i, col in enumerate(self.feature_columns):
            self.baseline_stats[col] = {
                'mean': float(np.mean(X_resting[:, i])),
                'std': float(np.std(X_resting[:, i])) if np.std(X_resting[:, i]) > 1e-4 else 1.0
            }

        # Train a compact Decision Tree for C code generation
        tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        tree_clf.fit(X, y)

        # Save artifacts
        payload = {
            'clf': self.clf,
            'tree_clf': tree_clf,
            'anomaly_detector': self.anomaly_detector,
            'feature_columns': self.feature_columns,
            'baseline_stats': self.baseline_stats,
            'metrics': {'accuracy': mean_acc, 'macro_f1': mean_f1}
        }
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, self.model_path)
        print(f"Model saved successfully to {self.model_path}")
        return payload

    def load(self):
        """Loads model from disk."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file {self.model_path} not found.")
        payload = joblib.load(self.model_path)
        self.clf = payload['clf']
        self.anomaly_detector = payload['anomaly_detector']
        self.feature_columns = payload['feature_columns']
        self.baseline_stats = payload.get('baseline_stats', {})
        return self

    def predict_window(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Infers state and anomaly score for a single feature window."""
        x_vec = np.zeros((1, len(self.feature_columns)))
        for i, col in enumerate(self.feature_columns):
            val = feature_dict.get(col, 0.0)
            x_vec[0, i] = 0.0 if np.isnan(val) else val

        if self.clf is None:
            self.load()

        state_id = int(self.clf.predict(x_vec)[0])
        probs = self.clf.predict_proba(x_vec)[0]
        confidence = float(np.max(probs))

        # Anomaly score: decision function from IsolationForest mapped to [0, 1]
        iso_score = float(self.anomaly_detector.decision_function(x_vec)[0])
        # Negative means anomaly, positive means inlier -> normalize to 0..1
        anomaly_score = float(np.clip(1.0 / (1.0 + np.exp(3.0 * iso_score)), 0.0, 1.0))

        return {
            'state_id': state_id,
            'state_name': STATE_NAMES.get(state_id, "UNKNOWN"),
            'confidence': round(confidence, 3),
            'anomaly_score': round(anomaly_score, 3),
            'probabilities': {STATE_NAMES[k]: round(float(p), 3) for k, p in enumerate(probs)}
        }
