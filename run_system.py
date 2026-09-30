"""
run_system.py
-------------
Step 15: Integrate everything.
Master orchestrator for the Real-Time Wearable Health-Monitoring Prototype on ?T-Kernel 3.0.
Boots the ?T-Kernel 3.0 Real-Time Execution Engine, real-time sensor streamer,
alert engine, BLE abstraction, and FastAPI/WebSocket telematics server.
"""

import sys
import os
import time
import argparse
import webbrowser
import uvicorn
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="?T-Kernel 3.0 Wearable Health Prototype")
    parser.add_argument("--port", type=int, default=8005, help="Port to run the telematics server on (default: 8005)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    print("=" * 80)
    print("  ?T-KERNEL 3.0 REAL-TIME WEARABLE HEALTH-MONITORING PROTOTYPE")
    print("=" * 80)
    print("  Official Technical Description:")
    print("  'The proposed system is a real-time wearable health-monitoring prototype based")
    print("   on ?T-Kernel 3.0. Instead of relying on physical sensors during the initial")
    print("   prototype phase, openly accessible physiological datasets are used to emulate")
    print("   real-time sensor streams. The system performs preprocessing, feature extraction")
    print("   and lightweight AI-based physiological state detection while ?T-Kernel 3.0")
    print("   manages real-time tasks, priorities and inter-task communication. A hybrid")
    print("   communication architecture provides Bluetooth-based local monitoring and")
    print("   Wi-Fi/Internet-based remote monitoring. The architecture is designed so that")
    print("   the simulated data source can later be replaced by actual ESP32-connected sensors")
    print("   without redesigning the core system.'")
    print("=" * 80)

    url = f"http://{args.host}:{args.port}/"
    print(f"\n[+] Starting FastAPI & WebSocket Telematics Server on {url} ...")
    print(f"[+] ?T-Kernel 3.0 RTOS Priority Preemption & Mailbox IPC: ACTIVE")
    
    if not args.no_browser:
        print(f"[+] Opening Clinical & Wearable Monitoring Dashboard in default browser...")
        try:
            webbrowser.open(url)
        except Exception:
            pass

    print(f"[?] Dashboard accessible at: {url}")
    print(f"    Press Ctrl+C to terminate system.\n")

    uvicorn.run("src.backend.server:app", host=args.host, port=args.port, log_level="info")

if __name__ == '__main__':
    main()
