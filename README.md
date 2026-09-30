# AI-Powered Real-Time Personal Health Monitoring System
### TRON Programming Contest 2026 — μT-Kernel 3.0

---

## 1. Project Overview

This project presents an AI-powered real-time personal health monitoring system designed using **μT-Kernel 3.0** and the **STM32N6570-DK** microcontroller board.

The proposed system is designed for a portable personal monitoring device capable of processing physiological information, detecting abnormal physiological patterns, generating alerts, and providing local and remote monitoring.

For the initial prototype, openly accessible physiological data is used to emulate the real-time sensor data that would eventually be obtained from wearable sensors.

The system combines:

- Real-time operating system processing using μT-Kernel 3.0
- Physiological signal processing
- Lightweight machine learning
- Real-time anomaly/state detection
- Priority-based alert handling
- Bluetooth-based local monitoring
- Wi-Fi/Internet-based remote monitoring
- A web-based monitoring dashboard

The architecture is designed so that the dataset-based input can later be replaced by real sensor data without redesigning the core processing architecture.

---

# 2. Main Objective

The main objective is to demonstrate how an embedded real-time operating system can coordinate physiological data processing and AI inference while maintaining predictable task scheduling and priority-based alert handling.

The system aims to demonstrate:

1. Physiological data acquisition simulation
2. Signal preprocessing
3. Feature extraction
4. Lightweight AI inference
5. Real-time task management
6. Priority-based alert processing
7. Local Bluetooth communication
8. Remote Wi-Fi communication
9. Monitoring and visualization

---

# 3. System Architecture

```text
                 OPEN PHYSIOLOGICAL DATASET
                            |
                            v
                  +----------------------+
                  |    Dataset Loader    |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Data Preprocessing   |
                  | Filtering            |
                  | Normalization        |
                  | Windowing             |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Feature Extraction   |
                  | ECG / PPG / SpO2     |
                  | Temperature / Motion |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |   AI/ML Inference    |
                  | Physiological State  |
                  | / Anomaly Detection  |
                  +----------+-----------+
                             |
                             v
              +--------------------------------+
              |       μT-Kernel 3.0            |
              |                                |
              | Data Task                      |
              | Processing Task                |
              | Feature Task                   |
              | AI Task                        |
              | Alert Task                     |
              | Communication Task              |
              | Logging Task                   |
              +---------------+----------------+
                              |
                   +----------+----------+
                   |                     |
                   v                     v
              Bluetooth                Wi-Fi
                   |                     |
                   v                     v
             Local Phone             Internet
                                         |
                                         v
                                  Remote Dashboard
