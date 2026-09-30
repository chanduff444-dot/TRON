"""
src/streamer/hardware_bridge.py
-------------------------------
Hardware Auto-Detection & Real-Time Sensor Ingestion Bridge.
Detects physical hardware connections via:
1. USB Serial (Auto-detects COM ports, probes baud rates 115200/9600, handshakes)
2. Network / Wi-Fi (HTTP & WebSocket push from ESP32 / Arduino / STM32)

Identifies sensor modules by signature / WHO_AM_I probe:
- AD8232 (Single-lead ECG)
- MAX30102 (Optical PPG Pulse Oximeter)
- MPU6050 (6-DOF Accelerometer & Gyroscope)
- TMP117 (High-precision clinical skin temperature)
- Synthetic / Analog GSR (Skin conductance)

Dynamically routes physical samples into ?T-Kernel 3.0 Task_SensorIngest (Priority 3).
"""

import time
import json
import threading
from typing import Dict, Any, List, Optional, Callable

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

class HardwareBridge:
    def __init__(self, on_sensor_sample: Callable[[Dict[str, Any]], None]):
        self.on_sensor_sample = on_sensor_sample
        self.is_running = False
        self.active_mode = "EMULATION" # "EMULATION" or "PHYSICAL_HARDWARE"
        self.connected_port: Optional[str] = None
        self.serial_conn = None
        self.detected_sensors: Dict[str, bool] = {
            "AD8232": False,
            "MAX30102": False,
            "MPU6050": False,
            "TMP117": False,
            "GSR": False
        }
        self.last_hardware_packet_time = 0
        self._scan_thread: Optional[threading.Thread] = None
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []

    def register_status_subscriber(self, cb: Callable[[Dict[str, Any]], None]):
        self._subscribers.append(cb)

    def start_auto_detection(self):
        if self.is_running:
            return
        self.is_running = True
        self._scan_thread = threading.Thread(target=self._com_scan_loop, daemon=True, name="HW_AutoDetect")
        self._scan_thread.start()

    def stop(self):
        self.is_running = False
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except Exception:
                pass

    def _com_scan_loop(self):
        """Continuously checks for plugged-in USB COM ports (e.g. ESP32, Arduino)."""
        while self.is_running:
            if self.active_mode != "PHYSICAL_HARDWARE" or not (self.serial_conn and self.serial_conn.is_open):
                if serial:
                    ports = list(serial.tools.list_ports.comports())
                    for p in ports:
                        # Check typical USB-to-UART chips (CP210x, CH340, FTDI, STM VCP)
                        desc_lower = (p.description + " " + p.manufacturer if p.manufacturer else p.description).lower()
                        if any(k in desc_lower for k in ["ch340", "cp210", "ftdi", "usb", "serial", "stmicroelectronics", "esp32"]):
                            self._try_connect_serial(p.device)
                            if self.active_mode == "PHYSICAL_HARDWARE":
                                break

            time.sleep(2.0)

    def _try_connect_serial(self, port_name: str):
        try:
            print(f"[HardwareBridge] Probing detected COM port: {port_name} ...")
            ser = serial.Serial(port_name, baudrate=115200, timeout=1.0)
            time.sleep(1.5) # Wait for microcontroller reboot
            ser.write(b"WHO_AM_I\n")
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            # Attempt to parse handshake or JSON frame
            sensors_found = self._parse_handshake_or_frame(line)
            if sensors_found:
                self.serial_conn = ser
                self.connected_port = port_name
                self.active_mode = "PHYSICAL_HARDWARE"
                print(f"[HardwareBridge] Successfully connected to {port_name}! Sensors: {self.detected_sensors}")
                self._notify_status()
                # Launch reader loop
                threading.Thread(target=self._serial_read_loop, daemon=True, name=f"HW_Reader_{port_name}").start()
            else:
                ser.close()
        except Exception as e:
            # Port busy or unable to connect
            pass

    def _parse_handshake_or_frame(self, line: str) -> bool:
        if not line:
            return False
        try:
            # Check JSON format
            data = json.loads(line)
            if "sensors" in data or "ecg_raw" in data or "ppg_raw" in data:
                self._identify_sensors_from_payload(data)
                return True
        except Exception:
            # Check CSV formatted line: ECG,PPG,AX,AY,AZ,TEMP
            parts = line.split(',')
            if len(parts) >= 4:
                self.detected_sensors["AD8232"] = True
                self.detected_sensors["MAX30102"] = True
                self.detected_sensors["MPU6050"] = True
                if len(parts) >= 6:
                    self.detected_sensors["TMP117"] = True
                return True
        return False

    def _identify_sensors_from_payload(self, data: Dict[str, Any]):
        if "sensors" in data and isinstance(data["sensors"], dict):
            for s_name in self.detected_sensors:
                if s_name in data["sensors"]:
                    self.detected_sensors[s_name] = True
        else:
            if "ecg_raw" in data or "ecg" in data: self.detected_sensors["AD8232"] = True
            if "ppg_raw" in data or "ppg" in data or "red" in data: self.detected_sensors["MAX30102"] = True
            if "a_x" in data or "accel" in data: self.detected_sensors["MPU6050"] = True
            if "temp" in data or "temp_c" in data: self.detected_sensors["TMP117"] = True
            if "gsr" in data: self.detected_sensors["GSR"] = True

    def _serial_read_loop(self):
        while self.is_running and self.serial_conn and self.serial_conn.is_open:
            try:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    self.ingest_external_sample(line)
            except Exception as e:
                print(f"[HardwareBridge] Serial read disconnected: {e}")
                break

        self.active_mode = "EMULATION"
        self.connected_port = None
        self._notify_status()

    def ingest_external_sample(self, raw_input: Any):
        """
        Ingests a live sample packet from physical hardware (Serial or Wi-Fi HTTP POST),
        normalizes it, and sends it to ?T-Kernel 3.0 Task_SensorIngest.
        """
        now_ms = int(time.time() * 1000)
        self.last_hardware_packet_time = now_ms
        self.active_mode = "PHYSICAL_HARDWARE"

        data = {}
        if isinstance(raw_input, str):
            try:
                data = json.loads(raw_input)
            except Exception:
                # Comma-separated: ecg, ppg, ax, ay, az, gx, gy, gz, temp
                parts = [p.strip() for p in raw_input.split(',')]
                try:
                    vals = [float(v) for v in parts]
                    data = {
                        "ecg_raw": vals[0] if len(vals) > 0 else 32500.0,
                        "ppg_raw": vals[1] if len(vals) > 1 else 72000.0,
                        "a_x": vals[2] if len(vals) > 2 else 0.0,
                        "a_y": vals[3] if len(vals) > 3 else 0.0,
                        "a_z": vals[4] if len(vals) > 4 else 1.0,
                        "temp": vals[8] if len(vals) > 8 else 36.6,
                    }
                except ValueError:
                    return
        elif isinstance(raw_input, dict):
            data = raw_input

        # Update detected sensors
        self._identify_sensors_from_payload(data)

        # Build standardized hardware packet for ?T-Kernel 3.0
        packet = {
            'timestamp_ms': now_ms,
            'source': 'PHYSICAL_HARDWARE',
            'scenario': 'live_physical_sensors',
            'ecg_raw': float(data.get('ecg_raw', data.get('ecg', 32500.0))),
            'ppg_raw': float(data.get('ppg_raw', data.get('ppg', 72000.0))),
            'a_x': float(data.get('a_x', 0.0)),
            'a_y': float(data.get('a_y', 0.0)),
            'a_z': float(data.get('a_z', 1.0)),
            'g_x': float(data.get('g_x', 0.0)),
            'g_y': float(data.get('g_y', 0.0)),
            'g_z': float(data.get('g_z', 0.0)),
            'temp': float(data.get('temp', data.get('temp_c', 36.6))),
            'contact': float(data.get('contact', 100.0)),
            'gsr_synthetic': float(data.get('gsr', 3.5)),
            'meta': {'activity': 'live_user'}
        }

        # Dispatch into ?T-Kernel 3.0
        self.on_sensor_sample(packet)

    def _notify_status(self):
        status = self.get_status()
        for cb in self._subscribers:
            try:
                cb(status)
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        # If no physical hardware packet in last 3 seconds, mark offline
        is_live = (time.time() * 1000 - self.last_hardware_packet_time) < 3000
        return {
            "mode": "PHYSICAL_HARDWARE" if is_live else "DATASET_EMULATION",
            "is_hardware_connected": is_live,
            "connected_port": self.connected_port,
            "detected_sensors": {k: (v if is_live else False) for k, v in self.detected_sensors.items()},
            "sensors_summary": [k for k, v in self.detected_sensors.items() if (v and is_live)]
        }
