"""
src/rtos/rtos_bridge.py
-----------------------
Step 10: Connect data stream to RTOS.
Bridges between the real-time SensorStreamer and ?T-Kernel 3.0 Ingestion Mailbox.
Tracks buffer latency, packet counters, and dropped packet statistics.
"""

import time
from typing import Dict, Any, Optional
from src.streamer.sensor_streamer import SensorStreamer
from src.rtos.rtos_tasks import RTOSTaskSystem
from src.rtos.tkernel_engine import E_OK, E_QOVR

class RTOSDataBridge:
    def __init__(self, streamer: SensorStreamer, rtos: RTOSTaskSystem):
        self.streamer = streamer
        self.rtos = rtos
        self.total_packets_received = 0
        self.total_packets_forwarded = 0
        self.total_packets_dropped = 0
        self.last_packet_timestamp = 0

        # Wire callback from streamer
        self.streamer.register_callback(self._on_sensor_packet)

    def _on_sensor_packet(self, packet: Dict[str, Any]):
        self.total_packets_received += 1
        self.last_packet_timestamp = packet.get('timestamp_ms', int(time.time() * 1000))

        # Push directly into ?T-Kernel 3.0 raw ingestion mailbox
        ercd = self.rtos.kernel.tk_snd_mbx(self.rtos.mbx_raw, packet, timeout_ms=10)
        if ercd == E_OK:
            self.total_packets_forwarded += 1
        elif ercd == E_QOVR:
            self.total_packets_dropped += 1
            if self.total_packets_dropped % 100 == 1:
                print(f"[RTOSDataBridge] Warning: RTOS Ingestion Mailbox full! Dropped {self.total_packets_dropped} packets.")

    def get_bridge_stats(self) -> Dict[str, Any]:
        return {
            'received': self.total_packets_received,
            'forwarded': self.total_packets_forwarded,
            'dropped': self.total_packets_dropped,
            'drop_rate_pct': round((self.total_packets_dropped / max(1, self.total_packets_received)) * 100.0, 2),
            'streamer_scenario': self.streamer.current_scenario,
            'streamer_speed': self.streamer.speed
        }
