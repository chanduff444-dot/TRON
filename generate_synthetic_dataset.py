"""
Synthetic wearable-sensor dataset generator (TRON Programming Contest prototype).

PURPOSE: temporary stand-in for a real sensor stream until real data / ESP32 sensors exist.
This data is SYNTHETIC. It is not clinical data and must never be presented as such.

Sensor mapping (emulated, not measured):
  heart_rate   <- ECG front end (AD8232) / PPG-derived pulse rate
  spo2         <- PPG oximeter (MAX30102)
  temperature  <- skin-contact temperature sensor (TMP117)
  acc_*, gyro_*<- IMU (MPU6050)

Labels: NORMAL / WARNING / CRITICAL come from INJECTED events (known ground truth),
not from medical thresholds. Event magnitudes are prototype values chosen to be
clearly distinguishable from normal variation; they are NOT clinical criteria.
"""
import json, numpy as np, pandas as pd
from pathlib import Path

FS_HZ = 1.0                      # output rate (samples/second); change to taste
START = pd.Timestamp("2026-10-01 09:00:00")
STATES = {0: "NORMAL", 1: "WARNING", 2: "CRITICAL"}

# activity -> (HR offset above resting bpm, acc std g, gyro std deg/s)
ACT = {"REST": (0, 0.02, 3.0), "WALK": (28, 0.25, 40.0), "RUN": (75, 0.70, 120.0)}

# event -> (warning magnitude, critical magnitude)
EVENTS = {
    "TACHYCARDIA_AT_REST": (35, 60),     # + bpm (only meaningful during REST)
    "LOW_SPO2":            (-4.5, -9.5), # delta percentage points
    "HIGH_TEMPERATURE":    (1.2, 2.5),   # delta deg C
    "FALL":                (None, None), # impact spike then stillness -> CRITICAL
}

def generate(subject_id, seed, schedule, events, resting_hr=None):
    """schedule: [(activity, seconds)], events: [(type, start_s, dur_s, level 1|2)]"""
    rng = np.random.default_rng(seed)
    acts = np.concatenate([[a] * int(d * FS_HZ) for a, d in schedule])
    n = len(acts)
    t = np.arange(n) / FS_HZ
    rhr = resting_hr or rng.normal(68, 6)
    spo2_base = rng.normal(97.5, 0.5)
    temp_base = rng.normal(33.8, 0.4)          # skin temperature, not core

    # heart rate: first-order lag toward activity target + noise
    target = np.array([rhr + ACT[a][0] for a in acts])
    hr = np.empty(n); hr[0] = target[0]
    alpha = 1 - np.exp(-1 / (15 * FS_HZ))      # ~15 s time constant
    for i in range(1, n):
        hr[i] = hr[i-1] + alpha * (target[i] - hr[i-1])
    hr += rng.normal(0, 1.2, n)
    spo2 = spo2_base + rng.normal(0, 0.35, n)
    temp = temp_base + np.cumsum(rng.normal(0, 0.005, n)) * 0.3 + rng.normal(0, 0.03, n)
    temp += 0.4 * np.array([a != "REST" for a in acts]) * (1 - np.exp(-t / 300))

    acc_std = np.array([ACT[a][1] for a in acts]); gyro_std = np.array([ACT[a][2] for a in acts])
    acc = rng.normal(0, 1, (n, 3)) * acc_std[:, None] + np.array([0.0, 0.0, 1.0])  # gravity on z
    gyro = rng.normal(0, 1, (n, 3)) * gyro_std[:, None]

    label = np.zeros(n, int); etype = np.array([""] * n, dtype=object)
    for typ, s, d, lvl in events:
        i0, i1 = int(s * FS_HZ), min(n, int((s + d) * FS_HZ))
        ramp = np.clip(np.minimum(np.arange(i1 - i0), np.arange(i1 - i0)[::-1]) / (5 * FS_HZ), 0, 1)
        if typ == "FALL":
            acc[i0] = [rng.normal(0, .5), rng.normal(0, .5), 3.5]      # impact
            gyro[i0] = rng.normal(0, 250, 3)
            acc[i0+1:i1] = np.array([0.0, 0.0, 1.0]) + rng.normal(0, 0.01, (i1-i0-1, 3))  # lying still
            gyro[i0+1:i1] = rng.normal(0, 1, (i1-i0-1, 3))
            hr[i0:i1] += 15 * ramp; lvl = 2
        else:
            mag = EVENTS[typ][lvl - 1]
            if typ == "TACHYCARDIA_AT_REST": hr[i0:i1] = np.maximum(hr[i0:i1], hr[i0:i1] + mag * ramp)
            elif typ == "LOW_SPO2": spo2[i0:i1] += mag * ramp
            elif typ == "HIGH_TEMPERATURE": temp[i0:i1] += mag * ramp
        active = np.arange(i0, i1)[ramp > 0.5] if typ != "FALL" else np.arange(i0, i1)
        label[active] = np.maximum(label[active], lvl); etype[active] = typ

    return pd.DataFrame({
        "timestamp": (START + pd.to_timedelta(t, unit="s")).strftime("%Y-%m-%dT%H:%M:%S"),
        "subject_id": subject_id,
        "heart_rate": hr.round(1), "spo2": np.clip(spo2, 70, 100).round(1),
        "temperature": temp.round(2),
        "acc_x": acc[:, 0].round(3), "acc_y": acc[:, 1].round(3), "acc_z": acc[:, 2].round(3),
        "gyro_x": gyro[:, 0].round(1), "gyro_y": gyro[:, 1].round(1), "gyro_z": gyro[:, 2].round(1),
        "activity": acts, "state_label": [STATES[v] for v in label], "event_type": etype,
        "source": "SYNTHETIC",
    })

def random_schedule(rng, total=1800):
    sched, used = [], 0
    while used < total:
        a = rng.choice(["REST", "WALK", "RUN"], p=[0.55, 0.32, 0.13])
        d = int(min(total - used, rng.integers(60, 240))); sched.append((a, d)); used += d
    return sched

def random_events(rng, schedule, total=1800):
    evs, occupied = [], []
    for _ in range(rng.integers(3, 6)):
        typ = rng.choice(list(EVENTS)); s = int(rng.integers(60, total - 90)); d = int(rng.integers(30, 60))
        if any(abs(s - o) < 90 for o in occupied): continue
        if typ == "TACHYCARDIA_AT_REST":               # only inject when subject is actually resting
            act = np.concatenate([[a] * dd for a, dd in schedule])
            if act[s:s+d].size == 0 or not (act[s:s+d] == "REST").all(): continue
        occupied.append(s); evs.append((typ, s, d, int(rng.choice([1, 2], p=[0.6, 0.4]))))
    return evs

def main(out="data"):
    out = Path(out); out.mkdir(exist_ok=True)
    frames = []
    for sid in range(1, 11):                              # 10 virtual subjects x 30 min
        rng = np.random.default_rng(1000 + sid)
        sch = random_schedule(rng); ev = random_events(rng, sch)
        df = generate(sid, 1000 + sid, sch, ev); df.to_csv(out / f"subject_{sid:03d}.csv", index=False); frames.append(df)
    pd.concat(frames).to_csv(out / "training_all.csv", index=False)

    # scripted 8-minute demo: NORMAL -> WARNING -> recovery -> CRITICAL -> recovery
    demo_sched = [("REST", 150), ("WALK", 90), ("REST", 240)]
    demo_events = [("LOW_SPO2", 60, 40, 1),              # WARNING  @ 01:00-01:40 (resting)
                   ("HIGH_TEMPERATURE", 300, 45, 1),      # WARNING  @ 05:00
                   ("FALL", 400, 30, 2)]                  # CRITICAL @ 06:40
    demo = generate(99, 4242, demo_sched, demo_events, resting_hr=70)
    demo.to_csv(out / "demo_stream.csv", index=False)

    allf = pd.concat(frames + [demo])
    meta = {"generated_by": "generate_synthetic_dataset.py", "synthetic": True, "sampling_hz": FS_HZ,
            "rows": {"training_all": int(sum(len(f) for f in frames)), "demo_stream": len(demo)},
            "label_counts": allf["state_label"].value_counts().to_dict(),
            "columns": list(demo.columns)}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
