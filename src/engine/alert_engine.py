"""
src/engine/alert_engine.py
--------------------------
Step 11: Implement alert engine.
Multi-parameter physiological alert manager with temporal debouncing and hysteresis.
Prevents false alarms by checking multi-modal context (e.g. ignoring elevated HR during exertion).
"""

import time
from typing import Dict, Any, List, Optional

class AlertEngine:
    def __init__(self, debounce_threshold: int = 2):
        self.debounce_threshold = debounce_threshold
        self.consecutive_violations: Dict[str, int] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []

    def evaluate(self, features: Dict[str, Any], inference: Dict[str, Any]) -> List[Dict[str, Any]]:
        hr = features.get('hr_bpm', 70.0)
        temp = features.get('temp_mean_c', 36.5)
        contact = features.get('contact_mean', 100.0)
        state_name = inference.get('state_name', 'RESTING')
        anomaly_score = inference.get('anomaly_score', 0.0)

        potential_alerts = {}

        # 1. Lead-off detection
        if contact < 10.0 or state_name == 'MOTION_ARTIFACT':
            potential_alerts['ALERT_LEAD_OFF'] = {
                'code': 'ALERT_LEAD_OFF',
                'severity': 'CRITICAL',
                'title': 'Sensor Lead-Off Detected',
                'message': 'Electrodes or optical sensor detached or excessive motion artifact.'
            }

        # 2. Contextual Tachycardia (HR > 115 bpm while resting/sedentary)
        elif hr > 115.0 and state_name != 'ACTIVE_EXERTION':
            potential_alerts['ALERT_TACHYCARDIA_INACTIVE'] = {
                'code': 'ALERT_TACHYCARDIA_INACTIVE',
                'severity': 'WARNING',
                'title': 'Contextual Tachycardia',
                'message': f'Elevated Heart Rate ({hr:.0f} bpm) detected while in inactive/sedentary state.'
            }

        # 3. Bradycardia (HR < 45 bpm)
        elif hr < 45.0 and hr > 0:
            potential_alerts['ALERT_BRADYCARDIA'] = {
                'code': 'ALERT_BRADYCARDIA',
                'severity': 'WARNING',
                'title': 'Bradycardia Warning',
                'message': f'Low Heart Rate ({hr:.0f} bpm) detected.'
            }

        # 4. Physiological Strain Anomaly (Anomaly score > 0.85)
        if anomaly_score > 0.85 and 'ALERT_LEAD_OFF' not in potential_alerts:
            potential_alerts['ALERT_PHYSIOLOGICAL_STRAIN'] = {
                'code': 'ALERT_PHYSIOLOGICAL_STRAIN',
                'severity': 'WARNING',
                'title': 'High Physiological Strain',
                'message': f'Elevated autonomic/cardiovascular strain (Anomaly Score: {anomaly_score:.2f}).'
            }

        # 5. Core/Skin Temperature
        if temp > 38.5:
            potential_alerts['ALERT_HYPERTHERMIA'] = {
                'code': 'ALERT_HYPERTHERMIA',
                'severity': 'WARNING',
                'title': 'Hyperthermia Warning',
                'message': f'Elevated surface temperature ({temp:.1f}?C).'
            }
        elif temp < 33.5 and 'ALERT_LEAD_OFF' not in potential_alerts:
            potential_alerts['ALERT_HYPOTHERMIA'] = {
                'code': 'ALERT_HYPOTHERMIA',
                'severity': 'WARNING',
                'title': 'Hypothermia Warning',
                'message': f'Low surface temperature ({temp:.1f}?C).'
            }

        # Temporal debouncing (requires persistence to trigger)
        newly_active = []
        for code, alert_info in potential_alerts.items():
            self.consecutive_violations[code] = self.consecutive_violations.get(code, 0) + 1
            if self.consecutive_violations[code] >= self.debounce_threshold:
                if code not in self.active_alerts:
                    alert_info['timestamp_ms'] = int(time.time() * 1000)
                    self.active_alerts[code] = alert_info
                    self.alert_history.append(alert_info)
                    newly_active.append(alert_info)

        # Clear resolved alerts
        for code in list(self.active_alerts.keys()):
            if code not in potential_alerts:
                self.consecutive_violations[code] = 0
                del self.active_alerts[code]

        return list(self.active_alerts.values())
