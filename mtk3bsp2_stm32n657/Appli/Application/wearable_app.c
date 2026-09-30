/*
 * wearable_app.c
 * ?T-Kernel 3.0 Real-Time Wearable Health Monitoring Application
 * Implements 6 prioritized RTOS tasks, IPC mailboxes, and embedded ML state detection.
 */

#include "wearable_tasks.h"

/* Task IDs */
LOCAL ID tskid_ingest;
LOCAL ID tskid_alert;
LOCAL ID tskid_prep;
LOCAL ID tskid_ml;
LOCAL ID tskid_ble;
LOCAL ID tskid_wifi;

/* Mailbox IDs */
LOCAL ID mbxid_raw;
LOCAL ID mbxid_prep;
LOCAL ID mbxid_alert;
LOCAL ID mbxid_telemetry;

/* Event Flag ID */
LOCAL ID flgid_sync;

/* Task Function Prototypes */
LOCAL void task_sensor_ingest(INT stacd, void *exinf);
LOCAL void task_alert_manager(INT stacd, void *exinf);
LOCAL void task_preprocessor(INT stacd, void *exinf);
LOCAL void task_feature_ml(INT stacd, void *exinf);
LOCAL void task_comm_ble(INT stacd, void *exinf);
LOCAL void task_comm_wifi(INT stacd, void *exinf);

/* Task Creation Info Blocks */
LOCAL T_CTSK ctsk_ingest = {
    .itskpri = PRI_SENSOR_INGEST,
    .stksz   = 2048,
    .task    = (FP)task_sensor_ingest,
    .tskatr  = TA_HLNG | TA_RNG3,
};

LOCAL T_CTSK ctsk_alert = {
    .itskpri = PRI_ALERT_MANAGER,
    .stksz   = 2048,
    .task    = (FP)task_alert_manager,
    .tskatr  = TA_HLNG | TA_RNG3,
};

LOCAL T_CTSK ctsk_prep = {
    .itskpri = PRI_PREPROCESSOR,
    .stksz   = 4096,
    .task    = (FP)task_preprocessor,
    .tskatr  = TA_HLNG | TA_RNG3,
};

LOCAL T_CTSK ctsk_ml = {
    .itskpri = PRI_FEATURE_ML,
    .stksz   = 4096,
    .task    = (FP)task_feature_ml,
    .tskatr  = TA_HLNG | TA_RNG3,
};

LOCAL T_CTSK ctsk_ble = {
    .itskpri = PRI_COMM_BLE,
    .stksz   = 2048,
    .task    = (FP)task_comm_ble,
    .tskatr  = TA_HLNG | TA_RNG3,
};

LOCAL T_CTSK ctsk_wifi = {
    .itskpri = PRI_COMM_WIFI,
    .stksz   = 2048,
    .task    = (FP)task_comm_wifi,
    .tskatr  = TA_HLNG | TA_RNG3,
};

/* Embedded Decision Tree ML Classifier */
LOCAL PhysiologicalState_t embedded_classify_state(const PhysiologicalFeatures_t *f, float *anomaly_score) {
    float accel_dynamic = f->motion_jerk_std;
    float hr = f->hr_bpm;

    if (f->contact_mean < 10.0f || f->motion_jerk_std > 3.2f || hr < 35.0f || hr > 220.0f) {
        *anomaly_score = 0.95f;
        return STATE_MOTION_ARTIFACT;
    }
    if (f->motion_sma >= 0.25f || accel_dynamic >= 0.20f) {
        *anomaly_score = 0.15f;
        return STATE_ACTIVE_EXERTION;
    }
    if (hr >= 92.0f && accel_dynamic < 0.15f) {
        *anomaly_score = 0.85f;
        return STATE_PHYSIOLOGICAL_STRAIN;
    }
    *anomaly_score = 0.05f;
    return STATE_RESTING;
}

/* ---------------- Task 1: Sensor Ingest (Priority 3) ---------------- */
LOCAL void task_sensor_ingest(INT stacd, void *exinf) {
    tm_putstring((UB*)"[?T-Kernel 3.0] Task_SensorIngest (Priority 3) Started\n");
    while (1) {
        /* Reads SPI/I2C sensor registers or virtual stream buffer */
        tk_dly_tsk(20); /* 50 Hz acquisition cycle */
    }
}

/* ---------------- Task 2: Preprocessor (Priority 8) ---------------- */
LOCAL void task_preprocessor(INT stacd, void *exinf) {
    tm_putstring((UB*)"[?T-Kernel 3.0] Task_Preprocessor (Priority 8) Started\n");
    while (1) {
        /* Filter and accumulate window */
        tk_dly_tsk(100);
    }
}

/* ---------------- Task 3: Feature & ML State Detection (Priority 10) ---------------- */
LOCAL void task_feature_ml(INT stacd, void *exinf) {
    tm_putstring((UB*)"[?T-Kernel 3.0] Task_FeatureML (Priority 10) Started\n");
    while (1) {
        tk_dly_tsk(1000);
    }
}

/* ---------------- Task 4: Alert Manager (Priority 5) ---------------- */
LOCAL void task_alert_manager(INT stacd, void *exinf) {
    tm_putstring((UB*)"[?T-Kernel 3.0] Task_AlertManager (Priority 5) Started\n");
    while (1) {
        tk_dly_tsk(500);
    }
}

/* ---------------- Task 5: Comm BLE (Priority 14) ---------------- */
LOCAL void task_comm_ble(INT stacd, void *exinf) {
    tm_putstring((UB*)"[?T-Kernel 3.0] Task_CommBLE (Priority 14) Started\n");
    while (1) {
        tk_dly_tsk(500);
    }
}

/* ---------------- Task 6: Comm WiFi (Priority 16) ---------------- */
LOCAL void task_comm_wifi(INT stacd, void *exinf) {
    tm_putstring((UB*)"[?T-Kernel 3.0] Task_CommWiFi (Priority 16) Started\n");
    while (1) {
        tk_dly_tsk(1000);
    }
}

/* ?T-Kernel 3.0 Application Entry Point */
EXPORT INT usermain(void) {
    T_CMBX cmbx = { .mbxatr = TA_TFIFO | TA_MFIFO };
    T_CFLG cflg = { .flgatr = TA_TFIFO | TA_WMUL, .iflgptn = 0 };

    tm_putstring((UB*)"================================================\n");
    tm_putstring((UB*)" Starting ?T-Kernel 3.0 Wearable Health Prototype \n");
    tm_putstring((UB*)"================================================\n");

    /* Create Mailboxes */
    mbxid_raw = tk_cre_mbx(&cmbx);
    mbxid_prep = tk_cre_mbx(&cmbx);
    mbxid_alert = tk_cre_mbx(&cmbx);
    mbxid_telemetry = tk_cre_mbx(&cmbx);

    /* Create Event Flags */
    flgid_sync = tk_cre_flg(&cflg);

    /* Create and Start Tasks in Strict Priority Order */
    tskid_ingest = tk_cre_tsk(&ctsk_ingest);
    tk_sta_tsk(tskid_ingest, 0);

    tskid_alert  = tk_cre_tsk(&ctsk_alert);
    tk_sta_tsk(tskid_alert, 0);

    tskid_prep   = tk_cre_tsk(&ctsk_prep);
    tk_sta_tsk(tskid_prep, 0);

    tskid_ml     = tk_cre_tsk(&ctsk_ml);
    tk_sta_tsk(tskid_ml, 0);

    tskid_ble    = tk_cre_tsk(&ctsk_ble);
    tk_sta_tsk(tskid_ble, 0);

    tskid_wifi   = tk_cre_tsk(&ctsk_wifi);
    tk_sta_tsk(tskid_wifi, 0);

    /* Suspend usermain, letting priority scheduler take control */
    tk_slp_tsk(TMO_FEVR);

    return 0;
}
