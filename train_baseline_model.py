"""
train_baseline_model.py
------------------------
Trains and validates a simple baseline model on final_dataset.csv
(produced by build_final_dataset.py), using subject-level (leave-one-
subject-out) cross-validation -- important with only ~22 subjects, since
a random row-level split would leak the same subject's data into both
train and test and make results look better than they are.

Two tasks are set up out of the box:
  1. Activity classification (walk / rest / etc.) from motion + physio features
  2. Heart-rate regression (predicting label_hr_ref from non-ECG-derived features,
     as a sanity check that other channels carry cardiovascular information)

USAGE
-----
    pip install pandas scikit-learn
    python train_baseline_model.py --csv final_dataset.csv --task activity
    python train_baseline_model.py --csv final_dataset.csv --task heart_rate
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import joblib

FEATURE_PREFIXES = (
    "ecg_", "ppg_", "accel_", "gyro_", "motion_", "temp_",
    "contact_pressure_", "gsr_synthetic_",
)


def select_features(df: pd.DataFrame, exclude_prefix=None) -> list:
    cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    if exclude_prefix:
        cols = [c for c in cols if not c.startswith(exclude_prefix)]
    return cols


def run_activity_task(df: pd.DataFrame):
    feature_cols = select_features(df)
    df = df.dropna(subset=feature_cols + ["activity"])
    X = df[feature_cols].values
    y = df["activity"].values
    groups = df["subject_id"].values

    logo = LeaveOneGroupOut()
    accs, f1s = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced")
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))

    print(f"Activity classification, leave-one-subject-out over {logo.get_n_splits(X, y, groups)} subjects:")
    print(f"  mean accuracy: {np.mean(accs):.3f}  (std {np.std(accs):.3f})")
    print(f"  mean macro-F1: {np.mean(f1s):.3f}  (std {np.std(f1s):.3f})")

    # Final model trained on everything, for downstream use / export
    final_model = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced")
    final_model.fit(X, y)
    return final_model, feature_cols


def run_heart_rate_task(df: pd.DataFrame):
    # Exclude ECG-derived features so the model has to infer HR from
    # other channels (PPG, motion, temp) -- otherwise it's a trivial task.
    feature_cols = select_features(df, exclude_prefix="ecg_")
    df = df.dropna(subset=feature_cols + ["label_hr_ref"])
    X = df[feature_cols].values
    y = df["label_hr_ref"].astype(float).values
    groups = df["subject_id"].values

    logo = LeaveOneGroupOut()
    maes = []
    for train_idx, test_idx in logo.split(X, y, groups):
        reg = RandomForestRegressor(n_estimators=200, random_state=0)
        reg.fit(X[train_idx], y[train_idx])
        pred = reg.predict(X[test_idx])
        maes.append(mean_absolute_error(y[test_idx], pred))

    print(f"Heart-rate regression (non-ECG features), leave-one-subject-out over {logo.get_n_splits(X, y, groups)} subjects:")
    print(f"  mean MAE: {np.mean(maes):.2f} bpm  (std {np.std(maes):.2f})")

    final_model = RandomForestRegressor(n_estimators=200, random_state=0)
    final_model.fit(X, y)
    return final_model, feature_cols


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="Path to final_dataset.csv")
    ap.add_argument("--task", choices=["activity", "heart_rate"], default="activity")
    ap.add_argument("--model_out", default=None, help="Path to save the trained model (.joblib). Defaults to <task>_model.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "subject_id" not in df.columns:
        sys.exit("Expected a subject_id column -- did you generate this CSV with build_final_dataset.py?")

    if args.task == "activity":
        model, feature_cols = run_activity_task(df)
    else:
        model, feature_cols = run_heart_rate_task(df)

    out_path = args.model_out or f"{args.task}_model.joblib"
    joblib.dump({"model": model, "feature_columns": feature_cols}, out_path)
    print(f"\nSaved trained model + feature list to {out_path}")


if __name__ == "__main__":
    main()
