"""
train_on_synthetic.py
----------------------
Phase 2: Train PhysiologicalStateDetector on the synthetic dataset.

Uses the 1 Hz synthetic CSV (training_all.csv) via the SyntheticLoader adapter,
then saves the trained model to models/state_detector.joblib.

USAGE
-----
    python train_on_synthetic.py
    python train_on_synthetic.py --csv training_all.csv --out models/state_detector.joblib
"""

import argparse
from pathlib import Path
import pandas as pd

from src.data.synthetic_loader import load_synthetic_csv
from src.ml.state_detector import PhysiologicalStateDetector


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="training_all.csv",
                    help="Path to training_all.csv (default: training_all.csv)")
    ap.add_argument("--out", default="models/state_detector.joblib",
                    help="Where to save trained model (default: models/state_detector.joblib)")
    ap.add_argument("--window", type=int, default=30,
                    help="Rolling window in seconds at 1 Hz (default: 30)")
    ap.add_argument("--hop",    type=int, default=10,
                    help="Hop between windows in seconds (default: 10)")
    args = ap.parse_args()

    print("=" * 70)
    print("  TRON — Phase 2: Training on Synthetic Dataset")
    print("=" * 70)
    print(f"  CSV        : {args.csv}")
    print(f"  Window     : {args.window} s   Hop: {args.hop} s")
    print(f"  Model out  : {args.out}")
    print()

    # Step 1: Load & window the synthetic CSV
    dataset = load_synthetic_csv(args.csv, window_s=args.window, hop_s=args.hop)

    # Step 2: Train + LOGO cross-validate
    detector = PhysiologicalStateDetector(model_path=args.out)
    payload  = detector.train_and_evaluate(dataset)

    metrics = payload["metrics"]
    print()
    print("=" * 70)
    print(f"  LOGO CV Accuracy : {metrics['accuracy']:.3f}")
    print(f"  LOGO CV Macro-F1 : {metrics['macro_f1']:.3f}")
    print(f"  Model saved to   : {args.out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
