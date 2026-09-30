"""
simulate_hardware_connect.py
----------------------------
Demonstrates physical sensor auto-detection and real-time live ingestion.
Sends live physical sensor samples to the running prototype via POST /api/hardware/stream,
triggering auto-detection of AD8232, MAX30102, MPU6050, and TMP117.
"""

import time
import urllib.request
import json
import numpy as np

def run_hardware_simulation(duration_s=8.0):
    url = "http://127.0.0.1:8005/api/hardware/stream"
    print("=" * 75)
    print(" SIMULATING PHYSICAL ESP32 HARDWARE CONNECTION TO SYSTEM")
    print(f" Target Endpoint: {url}")
    print("=" * 75)
    
    t_start = time.time()
    step = 0
    while time.time() - t_start < duration_s:
        t = time.time() - t_start
        # Simulated live physical sensor signals
        ecg_sample = 32500.0 + 1200.0 * np.sin(2 * np.pi * 1.25 * t)
        ppg_sample = 72000.0 + 350.0 * (np.sin(2 * np.pi * 1.25 * t) ** 2)
        ax = 0.05 * np.sin(2 * np.pi * 0.5 * t)
        ay = 0.03 * np.cos(2 * np.pi * 0.5 * t)
        az = 0.98 + 0.02 * np.sin(2 * np.pi * 1.0 * t)
        temp_sample = 36.68 + 0.02 * np.sin(0.1 * t)

        payload = {
            "device_id": "ESP32_PHYSICAL_NODE_01",
            "sensors": {
                "AD8232": True,
                "MAX30102": True,
                "MPU6050": True,
                "TMP117": True
            },
            "ecg_raw": round(ecg_sample, 1),
            "ppg_raw": round(ppg_sample, 1),
            "a_x": round(ax, 3),
            "a_y": round(ay, 3),
            "a_z": round(az, 3),
            "temp": round(temp_sample, 2),
            "contact": 120.0
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        res = urllib.request.urlopen(req)
        resp_data = json.loads(res.read().decode('utf-8'))

        if step % 20 == 0:
            print(f"[Live ESP32 Stream] Mode: {resp_data['mode']} | Detected Sensors: {resp_data['detected_sensors']} | ECG: {ecg_sample:.1f} | Temp: {temp_sample:.2f}?C")

        step += 1
        time.sleep(0.04) # 25 Hz live transmission rate

    print("\n[?] Hardware streaming test completed.")

if __name__ == '__main__':
    run_hardware_simulation()
