"""
src/comm/ble_abstraction.py
---------------------------
Step 12: Implement Bluetooth abstraction.
Simulates Bluetooth Low Energy (BLE) GATT profile server with standard
Health Services and custom multi-sensor streaming characteristics.
Ready to be swapped to ESP32 NimBLE / Bluedroid C APIs in physical deployment.
"""

import struct
from typing import Dict, Any, List

# Standard GATT Service & Characteristic UUIDs
UUID_SERVICE_HEART_RATE        = 0x180D
UUID_CHAR_HR_MEASUREMENT       = 0x2A37
UUID_SERVICE_HEALTH_THERM      = 0x1809
UUID_CHAR_TEMP_MEASUREMENT     = 0x2A1C
UUID_SERVICE_CUSTOM_WEARABLE   = 0xFFE0
UUID_CHAR_CUSTOM_TELEMETRY     = 0xFFE1
UUID_CHAR_CUSTOM_STATE         = 0xFFE2

class BLEGATTServer:
    def __init__(self, device_name: str = "TRON-uTK3-Wearable"):
        self.device_name = device_name
        self.is_connected = True
        self.is_advertising = False
        self.notifications_enabled = True
        self.packet_count = 0

    def encode_heart_rate(self, hr_bpm: float) -> bytes:
        flags = 0x00  # UINT8 format, no sensor contact
        hr_val = int(max(0, min(255, hr_bpm)))
        return struct.pack("<BB", flags, hr_val)

    def encode_temperature(self, temp_c: float) -> bytes:
        flags = 0x00  # Celsius
        mantissa = int(temp_c * 10)
        temp_bytes = struct.pack("<i", mantissa)[:3] + bytes([0xFE]) # 10^-2 exponent
        return bytes([flags]) + temp_bytes

    def encode_custom_telemetry(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        self.packet_count += 1
        feats = telemetry.get('features', {})
        inf = telemetry.get('inference', {})
        
        # Binary frame: [Seq:2B][HR:2B][MotionSMA:2B][Temp:2B][State:1B][Anomaly:1B]
        seq = self.packet_count & 0xFFFF
        hr = int(feats.get('hr_bpm', 70.0) * 10)
        sma = int(feats.get('motion_sma', 0.0) * 100)
        temp = int(feats.get('temp_mean_c', 36.5) * 10)
        state_id = int(inf.get('state_id', 0))
        anomaly_byte = int(inf.get('anomaly_score', 0.0) * 100)

        raw_bytes = struct.pack("<HHHhBB", seq, hr, sma, temp, state_id, anomaly_byte)

        return {
            'device_name': self.device_name,
            'service_uuid': hex(UUID_SERVICE_CUSTOM_WEARABLE),
            'char_uuid': hex(UUID_CHAR_CUSTOM_TELEMETRY),
            'payload_hex': raw_bytes.hex(),
            'payload_len': len(raw_bytes),
            'fields': {
                'seq': seq,
                'hr_bpm': feats.get('hr_bpm', 70.0),
                'state': inf.get('state_name', 'RESTING'),
                'anomaly_score': inf.get('anomaly_score', 0.0)
            }
        }
