# Synthetic Wearable Dataset (temporary, SYNTHETIC)

Stand-in for a real sensor stream in the TRON Programming Contest health-monitor prototype.
Not clinical data. Regenerate with `python generate_synthetic_dataset.py` (needs numpy, pandas).

## Files (data/)
- `subject_001..010.csv` - 10 virtual subjects, 30 min each, 1 Hz (for training/evaluation)
- `training_all.csv` - all subjects concatenated (18,000 rows)
- `demo_stream.csv` - scripted 8-min stream for the live demo (subject 99)
- `metadata.json` - rates, row counts, label counts

## Columns
timestamp, subject_id, heart_rate (bpm), spo2 (%), temperature (deg C, skin-contact),
acc_x/y/z (g, gravity on z), gyro_x/y/z (deg/s), activity (REST/WALK/RUN),
state_label (NORMAL/WARNING/CRITICAL), event_type, source (always SYNTHETIC)

## Sensor mapping (emulated, not measured)
heart_rate <- AD8232 ECG / PPG pulse rate | spo2 <- MAX30102 | temperature <- TMP117 | acc/gyro <- MPU6050

## Labels
Ground truth = injected events, not medical thresholds. Events: TACHYCARDIA_AT_REST, LOW_SPO2,
HIGH_TEMPERATURE, FALL (always CRITICAL). Magnitudes are prototype values (see EVENTS in the script).
Activity raises HR/motion but is labeled NORMAL, so a model must use context (HR vs activity).

## Demo timeline (demo_stream.csv, 1 row/s)
00:00 NORMAL rest | 01:03-01:40 WARNING low SpO2 | 05:00-05:45 WARNING high temp | 06:40-07:09 CRITICAL fall

## Limits
Simulated at 1 Hz: no raw ECG/PPG waveforms, so R-peak/HRV features are not available from this data.
Swap in real data later by writing the same columns from the loader; downstream code is unchanged.
