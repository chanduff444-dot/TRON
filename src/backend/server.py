"""
src/backend/server.py
---------------------
FastAPI and WebSocket telematics server with:
- Dual-Mode Real-Time Sensor Ingestion:
  1. PhysioNet Dataset Emulation (Default)
  2. Physical Hardware Live Auto-Detection (USB Serial COM ports & Wi-Fi HTTP stream)
- Dynamic Sensor Probing & Handshake (AD8232, MAX30102, MPU6050, TMP117)
- ?T-Kernel 3.0 RTOS Priority Preemption & Telemetry
- Bluetooth Low Energy GATT profile server
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.streamer.sensor_streamer import SensorStreamer
from src.streamer.hardware_bridge import HardwareBridge
from src.streamer.demo_replay import DemoReplay
from src.rtos.rtos_tasks import RTOSTaskSystem
from src.rtos.rtos_bridge import RTOSDataBridge
from src.engine.alert_engine import AlertEngine
from src.comm.ble_abstraction import BLEGATTServer

app = FastAPI(title="?T-Kernel 3.0 Wearable Health Prototype Backend")

# Core subsystem instances
streamer = SensorStreamer(speed=1.0, emit_hz=50.0)
rtos_system = RTOSTaskSystem(fs=50.0, window_sec=5.0, hop_sec=1.0)
bridge = RTOSDataBridge(streamer, rtos_system)
alert_engine = AlertEngine(debounce_threshold=2)
ble_server = BLEGATTServer()

# Hardware Auto-Detection Bridge
def on_hardware_sample(packet: Dict[str, Any]):
    # Forward physical sample straight into ?T-Kernel 3.0 Ingestion Mailbox (Priority 3)
    rtos_system.kernel.tk_snd_mbx(rtos_system.mbx_raw, packet, timeout_ms=5)

hw_bridge = HardwareBridge(on_sensor_sample=on_hardware_sample)

# Demo stream replay (demo_stream.csv → live dashboard)
demo_replay = DemoReplay(csv_path="demo_stream.csv", speed=1.0, loop=False)

active_connections: List[WebSocket] = []
latest_telemetry: Dict[str, Any] = {}
latest_raw_sample: Dict[str, Any] = {}

class ScenarioRequest(BaseModel):
    scenario: str
    speed: float = 1.0

@app.on_event("startup")
def startup_event():
    print("[Backend] Booting ?T-Kernel 3.0 Real-Time Wearable Subsystems...")
    rtos_system.start_all_tasks()
    streamer.start()
    hw_bridge.start_auto_detection()

    # Wire telemetry callback from RTOS CommWiFi task
    def on_rtos_telemetry(frame: Dict[str, Any]):
        global latest_telemetry
        alerts = alert_engine.evaluate(frame['features'], frame['inference'])
        frame['active_alerts'] = alerts
        frame['ble_packet'] = ble_server.encode_custom_telemetry(frame)
        latest_telemetry = frame

    rtos_system.register_telemetry_subscriber(on_rtos_telemetry)

    # Wire raw sample stream callback
    def on_raw_sample(packet: Dict[str, Any]):
        global latest_raw_sample
        latest_raw_sample = packet

    streamer.register_callback(on_raw_sample)
    # Wire demo replay into the same callback so dashboard sees demo packets
    demo_replay.register_callback(on_raw_sample)
    print("[Backend] Subsystems and Hardware Auto-Detector running.")

@app.on_event("shutdown")
def shutdown_event():
    streamer.stop()
    hw_bridge.stop()

@app.get("/api/status")
def get_status():
    rtos_meta = rtos_system.kernel.get_system_telemetry()
    bridge_stats = bridge.get_bridge_stats()
    hw_status = hw_bridge.get_status()
    return {
        "status": "ONLINE",
        "timestamp_ms": int(time.time() * 1000),
        "hardware": hw_status,
        "rtos": rtos_meta,
        "bridge": bridge_stats,
        "ble": {
            "device_name": ble_server.device_name,
            "connected": ble_server.is_connected,
            "packets_sent": ble_server.packet_count
        }
    }

@app.post("/api/scenario")
def set_scenario(req: ScenarioRequest):
    streamer.set_speed(req.speed)
    streamer.inject_scenario(req.scenario)
    return {
        "success": True,
        "current_scenario": streamer.current_scenario,
        "speed": streamer.speed
    }

@app.post("/api/demo")
def control_demo(action: str = "start", speed: float = 1.0, loop: bool = False):
    """
    Control the scripted 8-minute demo stream replay.
    action: 'start' | 'stop'
    speed:  playback speed multiplier (default 1.0 = real-time)
    loop:   whether to loop the replay (default False)
    """
    global demo_replay
    if action == "start":
        demo_replay.stop()
        demo_replay = DemoReplay(
            csv_path="demo_stream.csv",
            speed=speed,
            loop=loop,
            callbacks=[lambda pkt: None],  # placeholder; real cb registered at startup
        )
        demo_replay.register_callback(lambda pkt: setattr(
            __import__('builtins'), '_latest_demo', pkt))
        # Re-register the raw-sample callback
        from src.backend import server as _srv
        demo_replay.register_callback(lambda pkt: _srv.__dict__.update(
            latest_raw_sample=pkt))
        demo_replay.start()
        return {"status": "DEMO_STARTED", "speed": speed, "loop": loop}
    elif action == "stop":
        demo_replay.stop()
        return {"status": "DEMO_STOPPED"}
    return {"error": "unknown action"}

@app.post("/api/hardware/stream")
async def ingest_physical_hardware(request: Request):
    """
    Direct Wi-Fi / LAN ingestion endpoint for ESP32 / Arduino / Microcontrollers.
    Accepts raw JSON sensor readings or comma-separated CSV samples.
    Auto-detects connected sensors and updates RTOS in real time.
    """
    body_bytes = await request.body()
    try:
        data = json.loads(body_bytes.decode('utf-8'))
    except Exception:
        data = body_bytes.decode('utf-8', errors='ignore').strip()

    hw_bridge.ingest_external_sample(data)
    return {
        "status": "INGESTED",
        "mode": "PHYSICAL_HARDWARE",
        "detected_sensors": hw_bridge.get_status()["sensors_summary"]
    }

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            payload = {
                "timestamp_ms": int(time.time() * 1000),
                "raw": latest_raw_sample,
                "telemetry": latest_telemetry,
                "hardware": hw_bridge.get_status(),
                "rtos": rtos_system.kernel.get_system_telemetry(),
                "bridge": bridge.get_bridge_stats(),
                "ble": getattr(latest_telemetry, 'ble_packet', {})
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)

dashboard_path = Path("src/dashboard").resolve()
app.mount("/static", StaticFiles(directory=str(dashboard_path)), name="static")

@app.get("/")
def get_index():
    return FileResponse(dashboard_path / "index.html")
