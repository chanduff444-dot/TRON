/*
 * wearable_tasks.h
 * Real-Time Wearable Health-Monitoring Prototype on ?T-Kernel 3.0
 * Task Definitions, Priorities, IPC Mailboxes, and Data Structures
 */

#ifndef WEARABLE_TASKS_H_
#define WEARABLE_TASKS_H_

#include <tk/tkernel.h>
#include <tm/tmonitor.h>

/* Task Priorities (?T-Kernel 3.0: 1 is highest priority) */
#define PRI_SENSOR_INGEST    3   /* Hard real-time sensor intake */
#define PRI_ALERT_MANAGER    5   /* Safety-critical alert engine */
#define PRI_PREPROCESSOR     8   /* Signal conditioning & windowing */
#define PRI_FEATURE_ML      10   /* Feature extraction & ML state detection */
#define PRI_COMM_BLE        14   /* Local Bluetooth LE telemetry */
#define PRI_COMM_WIFI       16   /* Remote WiFi cloud telematics */

/* Event Flag Bits */
#define FLG_WINDOW_READY    0x0001
#define FLG_ALERT_ACTIVE    0x0002

/* Physiological States */
typedef enum {
    STATE_RESTING            = 0,
    STATE_ACTIVE_EXERTION    = 1,
    STATE_PHYSIOLOGICAL_STRAIN = 2,
    STATE_MOTION_ARTIFACT    = 3
} PhysiologicalState_t;

/* Alert Codes */
typedef enum {
    ALERT_NONE               = 0,
    ALERT_TACHYCARDIA        = 1,
    ALERT_BRADYCARDIA        = 2,
    ALERT_STRAIN_ANOMALY     = 3,
    ALERT_LEAD_OFF           = 4,
    ALERT_TEMP_ABNORMAL      = 5
} AlertCode_t;

/* Raw Sensor Frame Structure */
typedef struct {
    UW  timestamp_ms;
    UW  sample_idx;
    float ecg_raw;
    float ppg_raw;
    float a_x, a_y, a_z;
    float g_x, g_y, g_z;
    float temp;
    float contact;
    float gsr_synthetic;
} RawSensorPacket_t;

/* Extracted Physiological Features */
typedef struct {
    UW  timestamp_ms;
    float hr_bpm;
    float rr_std_ms;
    float ppg_pulse_rate_bpm;
    float ppg_amplitude;
    float motion_sma;
    float motion_energy;
    float motion_jerk_std;
    float temp_mean_c;
    float contact_mean;
    float gsr_synthetic_mean;
} PhysiologicalFeatures_t;

/* ML State & Telemetry Frame */
typedef struct {
    UW  timestamp_ms;
    PhysiologicalState_t state;
    float confidence;
    float anomaly_score;
    PhysiologicalFeatures_t features;
    AlertCode_t active_alert;
} TelemetryFrame_t;

#endif /* WEARABLE_TASKS_H_ */
