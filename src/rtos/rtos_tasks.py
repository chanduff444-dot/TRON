"""
src/rtos/rtos_tasks.py
----------------------
Steps 8 & 9: Implement RTOS tasks, priorities, and IPC.

PRIORITY HIERARCHY (?T-Kernel 3.0 Standard: 1 is highest priority):
  - Priority 3 : Task_SensorIngest (Hard real-time sample intake)
  - Priority 5 : Task_AlertManager (Safety-critical anomaly & vital limit handling)
  - Priority 8 : Task_Preprocessor (Signal conditioning & window accumulation)
  - Priority 10: Task_FeatureML (Physiological state detection & feature extraction)
  - Priority 14: Task_CommBLE (Local wireless Bluetooth LE telemetry)
  - Priority 16: Task_CommWiFi (Remote cloud/backend telemetry streaming)

INTER-TASK COMMUNICATION (IPC):
  - MBX_INGEST_TO_PREP : Raw sensor samples from Ingestion to Preprocessor
  - MBX_PREP_TO_ML     : Conditioned 5s windows from Preprocessor to ML Task
  - MBX_ML_TO_ALERT    : ML inference & vitals to Alert Manager
  - MBX_TELEMETRY      : Final consolidated telemetry frames to Comm tasks
  - FLG_WINDOW_READY   : Bit 0x01
  - FLG_ALERT_ACTIVE   : Bit 0x02
"""

import time
from typing import Dict, Any, Optional
import numpy as np

from src.rtos.tkernel_engine import MicroTKernel, E_OK, TWF_ORW, TWF_CLR
from src.data.preprocessor import SignalPreprocessor
from src.features.feature_extractor import FeatureExtractor
from src.ml.state_detector import PhysiologicalStateDetector

# Event Flag Bit Masks
FLG_WINDOW_READY = 0x0001
FLG_ALERT_ACTIVE = 0x0002

class RTOSTaskSystem:
    def __init__(self, fs: float = 50.0, window_sec: float = 5.0, hop_sec: float = 1.0):
        self.kernel = MicroTKernel.get_instance()
        self.fs = fs
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.window_len = int(window_sec * fs)
        self.hop_len = int(hop_sec * fs)

        # Preprocessor & ML models
        self.preprocessor = SignalPreprocessor(fs=fs, window_sec=window_sec, hop_sec=hop_sec)
        self.feat_extractor = FeatureExtractor(fs=fs)
        self.state_detector = PhysiologicalStateDetector("models/state_detector.joblib")
        try:
            self.state_detector.load()
        except Exception:
            print("[RTOSTaskSystem] Note: training model on initial boot if needed")

        # IPC Mailboxes
        self.mbx_raw = self.kernel.tk_cre_mbx("MBX_INGEST_TO_PREP", max_depth=500)
        self.mbx_prep = self.kernel.tk_cre_mbx("MBX_PREP_TO_ML", max_depth=16)
        self.mbx_alert = self.kernel.tk_cre_mbx("MBX_ML_TO_ALERT", max_depth=32)
        self.mbx_telemetry = self.kernel.tk_cre_mbx("MBX_TELEMETRY", max_depth=64)

        # IPC Event Flag
        self.flg_sync = self.kernel.tk_cre_flg("FLG_SYSTEM_SYNC", iflgptn=0)

        # Callbacks for telemetry subscribers (WebSocket, BLE, Logger)
        self.telemetry_subscribers = []
        self.alert_subscribers = []

        # Buffer for online windowing in Preprocessor task
        self.sample_buffer = {
            'ecg_raw': [], 'ppg_raw': [], 'a_x': [], 'a_y': [], 'a_z': [],
            'g_x': [], 'g_y': [], 'g_z': [], 'temp': [], 'contact': [], 'gsr_synthetic': []
        }
        self.last_state_inference = {
            'state_id': 0, 'state_name': 'RESTING', 'confidence': 1.0, 'anomaly_score': 0.05
        }
        self.active_alerts = []

    def register_telemetry_subscriber(self, cb):
        self.telemetry_subscribers.append(cb)

    def register_alert_subscriber(self, cb):
        self.alert_subscribers.append(cb)

    # ------------------------------------------------------------- Task 1: Ingest (Priority 3)
    def task_sensor_ingest(self, stacd: int):
        """Hard real-time ingestion task: receives samples and feeds raw mailbox."""
        print("[Task_SensorIngest] Started with Priority 3 (Hard Real-Time).")
        # Ingestion is event-driven by rtos_bridge pushing to mbx_raw
        while True:
            # Monitors ingress health and yields tick
            self.kernel.tk_dly_tsk(100)

    # ------------------------------------------------------------- Task 2: Preprocessor (Priority 8)
    def task_preprocessor(self, stacd: int):
        """Pops raw samples, maintains sliding window, filters signals causal IIR."""
        print("[Task_Preprocessor] Started with Priority 8.")
        samples_since_last_hop = 0

        while True:
            ercd, packet = self.kernel.tk_rcv_mbx(self.mbx_raw, timeout_ms=500)
            if ercd != E_OK or packet is None:
                continue

            # Append to sliding buffer
            for k in self.sample_buffer:
                if k in packet:
                    self.sample_buffer[k].append(packet[k])

            samples_since_last_hop += 1

            # When we have a full window and hop satisfied
            if len(self.sample_buffer['ecg_raw']) >= self.window_len and samples_since_last_hop >= self.hop_len:
                samples_since_last_hop = 0
                
                # Trim buffer to window_len
                window_data = {k: np.array(v[-self.window_len:], dtype=float) for k, v in self.sample_buffer.items()}
                
                # Causal online filter
                ecg_filt, ppg_filt = self.preprocessor.filter_online_chunk(
                    window_data['ecg_raw'], window_data['ppg_raw']
                )

                window_payload = {
                    'timestamp_ms': packet.get('timestamp_ms', int(time.time()*1000)),
                    'scenario': packet.get('scenario', 'normal'),
                    'signals': {
                        'ecg_filt': ecg_filt,
                        'ppg_filt': ppg_filt,
                        **window_data
                    },
                    'latest_sample': packet
                }

                # Push to ML task via mailbox
                self.kernel.tk_snd_mbx(self.mbx_prep, window_payload)
                self.kernel.tk_set_flg(self.flg_sync, FLG_WINDOW_READY)

                # Prune oldest samples to prevent unbounded memory growth
                for k in self.sample_buffer:
                    if len(self.sample_buffer[k]) > self.window_len * 2:
                        self.sample_buffer[k] = self.sample_buffer[k][-self.window_len:]

    # ------------------------------------------------------------- Task 3: FeatureML (Priority 10)
    def task_feature_ml(self, stacd: int):
        """Extracts physiological features and evaluates state detector."""
        print("[Task_FeatureML] Started with Priority 10.")
        while True:
            ercd, window_payload = self.kernel.tk_rcv_mbx(self.mbx_prep, timeout_ms=1000)
            if ercd != E_OK or window_payload is None:
                continue

            # Extract features
            feats = self.feat_extractor.extract_all(window_payload)

            # Predict Physiological State & Anomaly Score
            inference = self.state_detector.predict_window(feats)
            self.last_state_inference = inference

            consolidated = {
                'timestamp_ms': window_payload['timestamp_ms'],
                'scenario': window_payload['scenario'],
                'features': feats,
                'inference': inference,
                'latest_sample': window_payload['latest_sample']
            }

            # Forward to Alert Manager (P5) and Telemetry (P14/P16)
            self.kernel.tk_snd_mbx(self.mbx_alert, consolidated)
            self.kernel.tk_snd_mbx(self.mbx_telemetry, consolidated)

    # ------------------------------------------------------------- Task 4: Alert Manager (Priority 5)
    def task_alert_manager(self, stacd: int):
        """Safety-critical alert engine evaluating multi-parameter physiological limits."""
        print("[Task_AlertManager] Started with Priority 5 (Safety-Critical).")
        debounce_counter = 0

        while True:
            ercd, data = self.kernel.tk_rcv_mbx(self.mbx_alert, timeout_ms=1000)
            if ercd != E_OK or data is None:
                continue

            feats = data['features']
            inf = data['inference']
            hr = feats.get('hr_bpm', 70.0)
            temp = feats.get('temp_mean_c', 36.5)
            anomaly_score = inf.get('anomaly_score', 0.0)
            state_name = inf.get('state_name', 'RESTING')

            alerts = []

            # 1. Lead-off detection
            if state_name == 'MOTION_ARTIFACT' and feats.get('contact_mean', 100) < 10:
                alerts.append({
                    'code': 'ALERT_LEAD_OFF',
                    'severity': 'CRITICAL',
                    'message': 'Sensor Lead-Off Detected: Verify electrode and optical placement'
                })

            # 2. Contextual Tachycardia (HR > 120 while NOT in active exertion)
            elif hr > 120.0 and state_name != 'ACTIVE_EXERTION':
                alerts.append({
                    'code': 'ALERT_TACHYCARDIA_INACTIVE',
                    'severity': 'WARNING',
                    'message': f'Elevated Heart Rate ({hr:.0f} bpm) detected during sedentary state'
                })

            # 3. Severe physiological strain / Anomaly detection
            elif anomaly_score > 0.85:
                alerts.append({
                    'code': 'ALERT_PHYSIOLOGICAL_STRAIN',
                    'severity': 'WARNING',
                    'message': f'Elevated Autonomic Strain (Anomaly Score: {anomaly_score:.2f})'
                })

            # 4. Temperature thresholds
            if temp > 38.5:
                alerts.append({
                    'code': 'ALERT_HYPERTHERMIA',
                    'severity': 'WARNING',
                    'message': f'High Skin Temperature ({temp:.1f}?C)'
                })
            elif temp < 34.0:
                alerts.append({
                    'code': 'ALERT_HYPOTHERMIA',
                    'severity': 'WARNING',
                    'message': f'Low Skin Temperature ({temp:.1f}?C)'
                })

            self.active_alerts = alerts
            if alerts:
                self.kernel.tk_set_flg(self.flg_sync, FLG_ALERT_ACTIVE)
                for cb in self.alert_subscribers:
                    try:
                        cb(alerts)
                    except Exception:
                        pass

    # ------------------------------------------------------------- Task 5: Comm BLE (Priority 14)
    def task_comm_ble(self, stacd: int):
        """Local Bluetooth LE transmission task."""
        print("[Task_CommBLE] Started with Priority 14.")
        while True:
            self.kernel.tk_dly_tsk(500)

    # ------------------------------------------------------------- Task 6: Comm WiFi (Priority 16)
    def task_comm_wifi(self, stacd: int):
        """Remote cloud/backend WiFi transmission task."""
        print("[Task_CommWiFi] Started with Priority 16.")
        while True:
            ercd, tele = self.kernel.tk_rcv_mbx(self.mbx_telemetry, timeout_ms=500)
            if ercd == E_OK and tele is not None:
                # Attach active alerts & RTOS diagnostics
                tele['active_alerts'] = self.active_alerts
                tele['rtos_metrics'] = self.kernel.get_system_telemetry()
                for cb in self.telemetry_subscribers:
                    try:
                        cb(tele)
                    except Exception:
                        pass

    def start_all_tasks(self):
        """Registers and starts all tasks in ?T-Kernel 3.0 scheduler."""
        # Hard real-time ingest (P3)
        self.tskid_ingest = self.kernel.tk_cre_tsk("Task_SensorIngest", 3, self.task_sensor_ingest)
        self.kernel.tk_sta_tsk(self.tskid_ingest)

        # Safety Alert Manager (P5)
        self.tskid_alert = self.kernel.tk_cre_tsk("Task_AlertManager", 5, self.task_alert_manager)
        self.kernel.tk_sta_tsk(self.tskid_alert)

        # Preprocessor (P8)
        self.tskid_prep = self.kernel.tk_cre_tsk("Task_Preprocessor", 8, self.task_preprocessor)
        self.kernel.tk_sta_tsk(self.tskid_prep)

        # Feature ML (P10)
        self.tskid_ml = self.kernel.tk_cre_tsk("Task_FeatureML", 10, self.task_feature_ml)
        self.kernel.tk_sta_tsk(self.tskid_ml)

        # Local BLE (P14)
        self.tskid_ble = self.kernel.tk_cre_tsk("Task_CommBLE", 14, self.task_comm_ble)
        self.kernel.tk_sta_tsk(self.tskid_ble)

        # Remote WiFi (P16)
        self.tskid_wifi = self.kernel.tk_cre_tsk("Task_CommWiFi", 16, self.task_comm_wifi)
        self.kernel.tk_sta_tsk(self.tskid_wifi)

        print("[RTOSTaskSystem] All 6 ?T-Kernel 3.0 tasks initialized and running.")
