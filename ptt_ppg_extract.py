"""
ptt_ppg_extract.py
-------------------
Converts the PhysioNet "Pulse Transit Time PPG Dataset" (WFDB .hea/.dat
records) into clean, per-record CSV files, and maps each recorded channel
onto the wearable sensor it stands in for in our project.

Dataset home page:
  https://physionet.org/content/pulse-transit-time-ppg/1.1.0/

USAGE
-----
1. Install the reader library (already available in most environments):
       pip install wfdb pandas

2. Download the dataset. Either:
   a) Use wfdb's built-in downloader (grabs the whole database):
          import wfdb
          wfdb.dl_database('pulse-transit-time-ppg', dl_dir='ptt_ppg_raw')
      or
   b) rsync (recommended by PhysioNet, resumable):
          rsync -Cavz physionet.org::pulse-transit-time-ppg ptt_ppg_raw

3. Run this script:
       python ptt_ppg_extract.py --raw_dir ptt_ppg_raw --out_dir ptt_ppg_clean

It will produce, inside --out_dir:
   - one CSV per recording (e.g. s01_walk.csv) with a `time_s` column plus
     one column per channel, renamed to the sensor it emulates
   - one metadata CSV (record_metadata.csv) with the per-record annotations
     that PhysioNet embeds in the header comment line (activity, gender,
     height, weight, age, BP, HR, SpO2 at start/end of the recording)
   - one combined long-format CSV (all_records_long.csv) with a
     `record_id` column, convenient for feeding straight into a
     pandas groupby / ML pipeline

CHANNEL -> SENSOR MAPPING
--------------------------
The raw dataset does not have a 1:1 match with our BOM, so channels are
relabelled to the sensor they best emulate for pipeline-development
purposes (NOT a claim that the hardware is identical):

    Raw channel     Emulates              Our sensor
    -----------     ------------------    ------------------------------
    ecg             ECG                   AD8232
    pleth_1..6      PPG (multi-site)      MAX30102 (channel 1 is the
                                           closest single-site analogue;
                                           2-6 are extra PPG sites you
                                           can ignore or use for
                                           robustness experiments)
    lc_1, lc_2      Sensor contact/       Stand-in for the stretch
                    attachment pressure   sensor (physically different
                                          signal, but same idea: a
                                          contact/deformation channel)
    temp_1..3       Skin/sensor temp      TMP117
    a_x, a_y, a_z   Accelerometer         MPU6050 (accel axes)
    g_x, g_y, g_z   Gyroscope             MPU6050 (gyro axes)

    Header metadata (per recording, not a time series):
    activity, gender, height, weight, age,
    bp_sys_start/end, bp_dia_start/end,
    hr_1_start/end, hr_2_start/end, spo2_start/end
        -> useful as static features or as labels/targets
           (e.g. predict BP or HR band from the waveform window)

NOTE: GSR has no analogue in this dataset. You'll still need a synthetic
or separately-sourced GSR channel, as already discussed.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

try:
    import wfdb
except ImportError:
    sys.exit("Missing dependency. Run: pip install wfdb pandas")

# Raw channel name -> (new column name, sensor it emulates)
CHANNEL_MAP = {
    "ecg": ("ecg_mV", "AD8232 (ECG)"),
    "pleth_1": ("ppg_site1", "MAX30102 (PPG)"),
    "pleth_2": ("ppg_site2", "MAX30102 (PPG, extra site)"),
    "pleth_3": ("ppg_site3", "MAX30102 (PPG, extra site)"),
    "pleth_4": ("ppg_site4", "MAX30102 (PPG, extra site)"),
    "pleth_5": ("ppg_site5", "MAX30102 (PPG, extra site)"),
    "pleth_6": ("ppg_site6", "MAX30102 (PPG, extra site)"),
    "lc_1": ("contact_pressure_1", "Stretch-sensor proxy"),
    "lc_2": ("contact_pressure_2", "Stretch-sensor proxy"),
    "temp_1": ("temp_1_C", "TMP117"),
    "temp_2": ("temp_2_C", "TMP117 (secondary site)"),
    "temp_3": ("temp_3_C", "TMP117 (secondary site)"),
    "a_x": ("accel_x", "MPU6050 (accel)"),
    "a_y": ("accel_y", "MPU6050 (accel)"),
    "a_z": ("accel_z", "MPU6050 (accel)"),
    "g_x": ("gyro_x", "MPU6050 (gyro)"),
    "g_y": ("gyro_y", "MPU6050 (gyro)"),
    "g_z": ("gyro_z", "MPU6050 (gyro)"),
}

# Matches key:value pairs like <activity>: walk in the header comment line
META_RE = re.compile(r"<([a-zA-Z0-9_]+)>:\s*([^\s<]+)")


def find_records(raw_dir: Path):
    """Return sorted list of record base names (without extension)."""
    heas = sorted(raw_dir.glob("*.hea"))
    return [h.stem for h in heas]


def parse_metadata(record) -> dict:
    """Pull the <key>: value annotations out of the header comments."""
    meta = {}
    for comment in getattr(record, "comments", []):
        for key, val in META_RE.findall(comment):
            meta[key] = val
    return meta


def extract_record(raw_dir: Path, record_name: str, out_dir: Path):
    record = wfdb.rdrecord(str(raw_dir / record_name))
    fs = record.fs
    sig_names = record.sig_name
    signals = record.p_signal  # shape: (n_samples, n_channels)

    # Build renamed column list; keep unmapped channels under their raw name
    columns = []
    for name in sig_names:
        mapped = CHANNEL_MAP.get(name)
        columns.append(mapped[0] if mapped else name)

    out_path = out_dir / f"{record_name}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s"] + columns)
        for i, row in enumerate(signals):
            t = i / fs
            writer.writerow([f"{t:.4f}"] + [f"{v:.6f}" for v in row])

    meta = parse_metadata(record)
    meta["record_id"] = record_name
    meta["fs_hz"] = fs
    meta["n_samples"] = signals.shape[0]
    meta["duration_s"] = round(signals.shape[0] / fs, 2)
    return out_path, meta


def build_long_format(out_dir: Path, record_csvs, combined_path: Path):
    """Stack all per-record CSVs into one long-format file with record_id."""
    header_written = False
    with open(combined_path, "w", newline="") as out_f:
        writer = None
        for record_name, csv_path in record_csvs:
            with open(csv_path, newline="") as in_f:
                reader = csv.reader(in_f)
                header = next(reader)
                if not header_written:
                    writer = csv.writer(out_f)
                    writer.writerow(["record_id"] + header)
                    header_written = True
                for row in reader:
                    writer.writerow([record_name] + row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", required=True, help="Folder with the downloaded .hea/.dat files")
    ap.add_argument("--out_dir", required=True, help="Folder to write cleaned CSVs into")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = find_records(raw_dir)
    if not records:
        sys.exit(f"No .hea files found in {raw_dir}. Did the download finish?")

    print(f"Found {len(records)} records.")

    record_csvs = []
    all_meta = []
    for name in records:
        try:
            path, meta = extract_record(raw_dir, name, out_dir)
            record_csvs.append((name, path))
            all_meta.append(meta)
            print(f"  wrote {path.name}")
        except Exception as e:
            print(f"  SKIPPED {name}: {e}", file=sys.stderr)

    # Metadata table
    meta_path = out_dir / "record_metadata.csv"
    if all_meta:
        fieldnames = sorted({k for m in all_meta for k in m.keys()})
        with open(meta_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_meta)
        print(f"Wrote metadata: {meta_path.name}")

    # Combined long-format table
    combined_path = out_dir / "all_records_long.csv"
    build_long_format(out_dir, record_csvs, combined_path)
    print(f"Wrote combined dataset: {combined_path.name}")


if __name__ == "__main__":
    main()
