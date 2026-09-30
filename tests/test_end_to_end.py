"""
tests/test_end_to_end.py
------------------------
Step 16: Automated verification test suite for all 16 subsystem components.
"""

import time
import unittest
import numpy as np
from pathlib import Path

from src.data.data_loader import generate_benchmark_record, parse_metadata, synthesize_gsr
from src.data.preprocessor import SignalPreprocessor
from src.features.feature_extractor import FeatureExtractor
from src.ml.state_detector import PhysiologicalStateDetector, derive_ground_truth_label
from src.streamer.sensor_streamer import SensorStreamer
from src.rtos.tkernel_engine import MicroTKernel, E_OK, TWF_ORW, TWF_CLR
from src.rtos.rtos_tasks import RTOSTaskSystem
from src.rtos.rtos_bridge import RTOSDataBridge
from src.engine.alert_engine import AlertEngine
from src.comm.ble_abstraction import BLEGATTServer

class TestWearableSystem(unittest.TestCase):
    def test_01_data_loader(self):
        rec = generate_benchmark_record(activity='sit', duration_s=5.0, fs=500.0)
        self.assertEqual(rec['fs'], 500.0)
        self.assertEqual(rec['n_samples'], 2500)
        self.assertIn('ecg', rec['signals'])
        self.assertIn('ppg', rec['signals'])
        self.assertIn('a_x', rec['signals'])
        self.assertIn('gsr_synthetic', rec['signals'])
        print("[?] Test 1: Data loader validated.")

    def test_02_preprocessor(self):
        prep = SignalPreprocessor(fs=500.0, window_sec=5.0, hop_sec=1.0)
        t = np.arange(2500) / 500.0
        ecg_raw = 32500.0 + 1000.0 * np.sin(2 * np.pi * 1.5 * t)
        ppg_raw = 72000.0 + 400.0 * np.sin(2 * np.pi * 1.5 * t)
        ecg_filt, ppg_filt = prep.filter_batch(ecg_raw, ppg_raw)
        # DC offset should be eliminated (mean close to 0)
        self.assertLess(abs(np.mean(ecg_filt)), 25.0)
        self.assertLess(abs(np.mean(ppg_filt)), 10.0)
        print("[?] Test 2: Preprocessing and bandpass filtering validated.")

    def test_03_feature_extractor(self):
        rec = generate_benchmark_record(activity='sit', duration_s=5.0, fs=500.0)
        prep = SignalPreprocessor(fs=500.0, window_sec=5.0, hop_sec=1.0)
        feat_extractor = FeatureExtractor(fs=500.0)

        for w in prep.segment_windows(rec['signals'], rec['meta']):
            feats = feat_extractor.extract_all(w)
            self.assertTrue(50 <= feats['hr_bpm'] <= 100)
            self.assertIn('ecg_rr_std_ms', feats)
            self.assertIn('motion_sma', feats)
            self.assertIn('temp_mean_c', feats)
            break
        print("[?] Test 3: Feature extraction validated.")

    def test_04_ml_state_detector(self):
        detector = PhysiologicalStateDetector("models/state_detector.joblib")
        detector.load()

        prep = SignalPreprocessor(fs=500.0, window_sec=5.0, hop_sec=1.0)
        feat_extractor = FeatureExtractor(fs=500.0)

        # 1. Test resting window
        rec_sit = generate_benchmark_record(activity='sit', duration_s=5.0, fs=500.0)
        for w in prep.segment_windows(rec_sit['signals'], rec_sit['meta']):
            feats_sit = feat_extractor.extract_all(w)
            res_sit = detector.predict_window(feats_sit)
            self.assertEqual(res_sit['state_name'], 'RESTING')
            self.assertLess(res_sit['anomaly_score'], 0.6)
            break

        # 2. Test active exertion window
        rec_run = generate_benchmark_record(activity='run', duration_s=5.0, fs=500.0)
        for w in prep.segment_windows(rec_run['signals'], rec_run['meta']):
            feats_run = feat_extractor.extract_all(w)
            res_run = detector.predict_window(feats_run)
            self.assertEqual(res_run['state_name'], 'ACTIVE_EXERTION')
            break

        print("[?] Test 4: ML physiological state and anomaly detector validated.")

    def test_05_tkernel_rtos_engine(self):
        kernel = MicroTKernel()
        mbxid = kernel.tk_cre_mbx("TEST_MBX")
        flgid = kernel.tk_cre_flg("TEST_FLG")

        # Test mailbox
        ercd = kernel.tk_snd_mbx(mbxid, {"data": 42})
        self.assertEqual(ercd, E_OK)
        ercd, msg = kernel.tk_rcv_mbx(mbxid, timeout_ms=100)
        self.assertEqual(ercd, E_OK)
        self.assertEqual(msg['data'], 42)

        # Test event flag
        kernel.tk_set_flg(flgid, 0x01)
        ercd = kernel.tk_wai_flg(flgid, 0x01, wfmode=TWF_ORW|TWF_CLR, timeout_ms=100)
        self.assertEqual(ercd, E_OK)

        print("[?] Test 5: ?T-Kernel 3.0 RTOS Engine (mailboxes & flags) validated.")

    def test_06_alert_engine(self):
        alert_engine = AlertEngine(debounce_threshold=1)
        # Contextual tachycardia: HR 125 while resting
        feats = {'hr_bpm': 125.0, 'temp_mean_c': 36.6, 'contact_mean': 100.0}
        inf = {'state_name': 'RESTING', 'anomaly_score': 0.90}
        alerts = alert_engine.evaluate(feats, inf)
        self.assertTrue(any(a['code'] == 'ALERT_TACHYCARDIA_INACTIVE' for a in alerts))
        print("[?] Test 6: Alert engine debouncing and contextual logic validated.")

    def test_07_ble_abstraction(self):
        ble = BLEGATTServer()
        hr_bytes = ble.encode_heart_rate(75.0)
        self.assertEqual(len(hr_bytes), 2)
        temp_bytes = ble.encode_temperature(36.8)
        self.assertEqual(len(temp_bytes), 5)
        packet = ble.encode_custom_telemetry({
            'features': {'hr_bpm': 72.0, 'motion_sma': 0.05, 'temp_mean_c': 36.6},
            'inference': {'state_id': 0, 'state_name': 'RESTING', 'anomaly_score': 0.05}
        })
        self.assertEqual(packet['char_uuid'], '0xffe1')
        print("[?] Test 7: Bluetooth SIG GATT profile encoding validated.")

    def test_08_end_to_end_streaming_pipeline(self):
        print("[+] Running End-to-End Pipeline test (Streamer -> RTOS -> Alerts -> Comm)...")
        streamer = SensorStreamer(speed=2.0, emit_hz=50.0)
        rtos = RTOSTaskSystem(fs=50.0, window_sec=2.0, hop_sec=0.5)
        bridge = RTOSDataBridge(streamer, rtos)

        received_telemetry = []
        rtos.register_telemetry_subscriber(lambda t: received_telemetry.append(t))

        rtos.start_all_tasks()
        streamer.start()

        # Let pipeline run for 3 seconds
        time.sleep(3.0)

        streamer.stop()

        self.assertGreater(bridge.total_packets_received, 50)
        self.assertGreater(bridge.total_packets_forwarded, 50)
        self.assertGreater(len(received_telemetry), 0)
        latest = received_telemetry[-1]
        self.assertIn('features', latest)
        self.assertIn('inference', latest)
        self.assertIn('rtos_metrics', latest)
        print(f"[?] Test 8: End-to-End streaming pipeline successfully verified ({len(received_telemetry)} telemetry frames processed).")

if __name__ == '__main__':
    unittest.main()
