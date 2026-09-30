/**
 * wearable_app.c
 * --------------
 * TRON Contest — STM32N6570-DK / µT-Kernel 3.0
 *
 * usermain() — the µT-Kernel application entry point.
 * Called by the BSP after the RTOS scheduler is running.
 *
 * Creates all 6 wearable tasks and their IPC objects (mailboxes + event flag),
 * then suspends itself. The scheduler drives the rest.
 *
 * Task priority hierarchy (lower number = higher precedence in µT-Kernel):
 *   Pri  3  Task_SensorIngest  — hard real-time ADC/I2C capture
 *   Pri  5  Task_AlertManager  — safety-critical threshold evaluation
 *   Pri  8  Task_Preprocessor  — causal IIR filter + sliding window
 *   Pri 10  Task_FeatureML     — feature extraction + AI state inference
 *   Pri 14  Task_CommBLE       — BLE GATT broadcast (local monitoring)
 *   Pri 16  Task_CommWiFi      — UART/Wi-Fi telemetry (remote monitoring)
 */

#include <tk/tkernel.h>     /* µT-Kernel 3.0 API                            */
#include <tm/tmonitor.h>    /* tm_printf debug output (STM32 UART)          */
#include <string.h>

#include "wearable_sensors.h"
#include "sensor_hal_stm32n6.h"

/* ── Stack sizes ────────────────────────────────────────────────────────── */
#define STACK_INGEST    2048
#define STACK_ALERT     2048
#define STACK_PREPROC   4096
#define STACK_FEATURE   4096
#define STACK_BLE       2048
#define STACK_WIFI      2048

/* ── IPC object IDs (global so tasks can reference them) ────────────────── */
ID mbx_ingest_to_prep  = 0;   /* RawSensorPacket_t queue                   */
ID mbx_prep_to_ml      = 0;   /* windowed signal buffer queue              */
ID mbx_ml_to_alert     = 0;   /* FeatureResult_t queue                     */
ID mbx_telemetry       = 0;   /* FeatureResult_t to comms tasks            */
ID flg_system_sync     = 0;   /* bit 0 = window ready, bit 1 = alert active*/

/* ── Task function prototypes ───────────────────────────────────────────── */
void Task_SensorIngest (INT stacd, void *exinf);
void Task_AlertManager (INT stacd, void *exinf);
void Task_Preprocessor (INT stacd, void *exinf);
void Task_FeatureML    (INT stacd, void *exinf);
void Task_CommBLE      (INT stacd, void *exinf);
void Task_CommWiFi     (INT stacd, void *exinf);

/* ── Simple message envelope for mailboxes ──────────────────────────────── */
typedef struct _MsgHdr { T_MSG msg; } MsgHdr_t;

/* ══════════════════════════════════════════════════════════════════════════
 *  IPC helpers
 * ══════════════════════════════════════════════════════════════════════════ */
static ID create_mailbox(void)
{
    T_CMBX cmbx = {
        .mbxatr = TA_TFIFO | TA_MFIFO,
        .maxmpri = 0,
        .msprior = 0
    };
    return tk_cre_mbx(&cmbx);
}

static ID create_event_flag(void)
{
    T_CFLG cflg = {
        .flgatr  = TA_TFIFO | TA_WMUL,
        .iflgptn = 0x00000000u
    };
    return tk_cre_flg(&cflg);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Task_SensorIngest  (Priority 3 — Hard real-time)
 *
 *  Reads one RawSensorPacket_t every 2 ms (500 Hz ECG governs).
 *  Pushes packet into mbx_ingest_to_prep.
 *  On hardware: this would be triggered by the ADC DMA half-complete IRQ.
 * ══════════════════════════════════════════════════════════════════════════ */
void Task_SensorIngest(INT stacd, void *exinf)
{
    (void)stacd; (void)exinf;
    tm_printf((UB*)"[Ingest] Started — 500 Hz sensor capture\n");

    static RawSensorPacket_t pkt;

    for (;;) {
        HAL_Sensors_ReadPacket(&pkt);

        /* Post to preprocessor mailbox */
        T_MSG *pmsg = (T_MSG *)&pkt;  /* reuse static buffer (single-task producer) */
        tk_snd_mbx(mbx_ingest_to_prep, pmsg);

        tk_dly_tsk(2);   /* 2 ms = 500 Hz */
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Task_Preprocessor  (Priority 8)
 *
 *  Accumulates 2500 ECG samples (5 s) into a window buffer.
 *  Applies causal IIR bandpass (emulated here; on-target use CMSIS-DSP biquad).
 *  Signals Task_FeatureML via mbx_prep_to_ml + sets FLG_WINDOW_READY.
 * ══════════════════════════════════════════════════════════════════════════ */

#define FLG_WINDOW_READY    0x01u
#define FLG_ALERT_ACTIVE    0x02u

/* Simple first-order IIR high-pass: y[n] = α*(y[n-1] + x[n] - x[n-1])   */
static float iir_ecg_hp(float xn, float *xprev, float *yprev)
{
    const float alpha = 0.995f;   /* fc ≈ 0.4 Hz at 500 Hz                 */
    float yn = alpha * (*yprev + xn - *xprev);
    *xprev = xn;
    *yprev = yn;
    return yn;
}

void Task_Preprocessor(INT stacd, void *exinf)
{
    (void)stacd; (void)exinf;
    tm_printf((UB*)"[Preproc] Started — 5 s sliding window, IIR filter\n");

    static int16_t ecg_win[SENSOR_WINDOW_SAMPLES];
    static float   ax_win[SENSOR_WINDOW_SAMPLES];
    static float   ay_win[SENSOR_WINDOW_SAMPLES];
    static float   az_win[SENSOR_WINDOW_SAMPLES];
    static float   spo2_acc;
    static float   temp_acc;

    float xp = 0.0f, yp = 0.0f;   /* IIR state */
    uint32_t idx = 0;

    for (;;) {
        T_MSG *pmsg;
        tk_rcv_mbx(mbx_ingest_to_prep, &pmsg, TMO_FEVR);
        RawSensorPacket_t *pkt = (RawSensorPacket_t *)pmsg;

        /* Accumulate into window */
        float ecg_f = iir_ecg_hp((float)pkt->ecg_raw, &xp, &yp);
        ecg_win[idx] = (int16_t)ecg_f;
        ax_win[idx]  = (float)pkt->accel_x_raw / 8192.0f;  /* ±4g LSB */
        ay_win[idx]  = (float)pkt->accel_y_raw / 8192.0f;
        az_win[idx]  = (float)pkt->accel_z_raw / 8192.0f;
        spo2_acc    += pkt->spo2_pct;
        temp_acc    += pkt->skin_temp_c;
        idx++;

        if (idx >= SENSOR_WINDOW_SAMPLES) {
            /* Window full — forward to ML task (reuse static buffer ptr) */
            T_MSG *fwd = (T_MSG *)ecg_win;
            tk_snd_mbx(mbx_prep_to_ml, fwd);
            tk_set_flg(flg_system_sync, FLG_WINDOW_READY);

            /* Reset accumulator */
            idx = 0;
            spo2_acc = 0.0f;
            temp_acc = 0.0f;
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Task_FeatureML  (Priority 10)
 *
 *  Waits for a window, computes lightweight features, runs the Decision-Tree
 *  state classifier (exported from Python as C thresholds), fills
 *  FeatureResult_t, and pushes to both mbx_ml_to_alert and mbx_telemetry.
 * ══════════════════════════════════════════════════════════════════════════ */

/* Lightweight decision tree thresholds (export from state_detector.py).
 * These are prototype values — regenerate via DecisionTreeClassifier
 * export_text() after model training.                                      */
static uint8_t classify_state(float hr, float accel_std, float temp)
{
    /* Leaf 0: RESTING          */
    /* Leaf 1: ACTIVE_EXERTION  */
    /* Leaf 2: STRAIN           */
    /* Leaf 3: ARTIFACT         */
    if (accel_std > 3.2f)          return 3;   /* fall / severe artifact  */
    if (accel_std > 0.20f)         return 1;   /* locomotion              */
    if (hr > 92.0f && accel_std < 0.15f) return 2;  /* resting strain     */
    return 0;                                   /* resting baseline        */
}

void Task_FeatureML(INT stacd, void *exinf)
{
    (void)stacd; (void)exinf;
    tm_printf((UB*)"[FeatureML] Started — state classifier active\n");

    static FeatureResult_t result;

    for (;;) {
        /* Wait for window-ready flag, then clear it */
        FLGPTN pattern;
        tk_wai_flg(flg_system_sync, FLG_WINDOW_READY,
                   TWF_ANDW | TWF_BITCLR, &pattern, TMO_FEVR);

        T_MSG *pmsg;
        tk_rcv_mbx(mbx_prep_to_ml, &pmsg, TMO_FEVR);
        int16_t *ecg_win = (int16_t *)pmsg;

        /* ── Simplified feature extraction (prototype) ────────────────
         * On-target: replace with CMSIS-DSP arm_rms_f32, arm_std_f32.
         * Full Python feature set takes ~8 ms on Cortex-M55 @ 600 MHz.  */
        float acc_sum = 0.0f, acc_sq = 0.0f;
        float ecg_sum = 0.0f, ecg_sq = 0.0f;
        for (uint32_t i = 0; i < SENSOR_WINDOW_SAMPLES; i++) {
            float e = (float)ecg_win[i];
            ecg_sum += e; ecg_sq += e * e;
        }
        float ecg_rms = sqrtf(ecg_sq / SENSOR_WINDOW_SAMPLES);

        /* Placeholder HR from ECG RMS (replace with R-peak detector)    */
        float hr_est   = 70.0f + ecg_rms * 0.01f;
        float accel_std = 0.15f;   /* replace with CMSIS arm_std_f32     */
        float temp_c    = 34.0f;

        result.hr_bpm         = hr_est;
        result.spo2_pct       = 97.5f;
        result.temp_mean_c    = temp_c;
        result.accel_x_std    = accel_std;
        result.motion_energy  = ecg_rms;
        result.anomaly_score  = 0.0f;
        result.state_id       = classify_state(hr_est, accel_std, temp_c);
        result.window_start_ms = tk_get_utim();

        /* Forward to alert engine and telemetry */
        tk_snd_mbx(mbx_ml_to_alert, (T_MSG *)&result);
        tk_snd_mbx(mbx_telemetry,   (T_MSG *)&result);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Task_AlertManager  (Priority 5 — Safety critical)
 *
 *  Evaluates thresholds on FeatureResult_t, raises FLG_ALERT_ACTIVE,
 *  prints critical alerts to UART (tm_printf) and toggles board LED.
 * ══════════════════════════════════════════════════════════════════════════ */

/* STM32N6570-DK: LD1 = green (normal), LD2 = red (alert) on PE1/PF1 */
#define LED_NORMAL_PORT  GPIOE
#define LED_NORMAL_PIN   GPIO_PIN_1
#define LED_ALERT_PORT   GPIOF
#define LED_ALERT_PIN    GPIO_PIN_1

static void led_normal(void)
{
    HAL_GPIO_WritePin(LED_NORMAL_PORT, LED_NORMAL_PIN, GPIO_PIN_SET);
    HAL_GPIO_WritePin(LED_ALERT_PORT,  LED_ALERT_PIN,  GPIO_PIN_RESET);
}
static void led_alert(void)
{
    HAL_GPIO_WritePin(LED_NORMAL_PORT, LED_NORMAL_PIN, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(LED_ALERT_PORT,  LED_ALERT_PIN,  GPIO_PIN_SET);
}

void Task_AlertManager(INT stacd, void *exinf)
{
    (void)stacd; (void)exinf;
    tm_printf((UB*)"[Alert] Started — threshold watchdog active\n");
    led_normal();

    static const char *STATE_NAMES[] = {
        "RESTING", "ACTIVE_EXERTION", "PHYSIOLOGICAL_STRAIN", "MOTION_ARTIFACT"
    };

    for (;;) {
        T_MSG *pmsg;
        tk_rcv_mbx(mbx_ml_to_alert, &pmsg, TMO_FEVR);
        FeatureResult_t *r = (FeatureResult_t *)pmsg;

        AlertLevel_t level = ALERT_NONE;

        /* WARNING conditions */
        if (r->spo2_pct < 94.0f)                    level = ALERT_WARNING;
        if (r->hr_bpm > 100.0f && r->state_id == 0) level = ALERT_WARNING;  /* rest tachy */
        if (r->temp_mean_c > 37.5f)                 level = ALERT_WARNING;

        /* CRITICAL conditions */
        if (r->spo2_pct < 90.0f)                    level = ALERT_CRITICAL;
        if (r->state_id == 3)                        level = ALERT_CRITICAL; /* fall */
        if (r->hr_bpm > 150.0f && r->state_id == 0) level = ALERT_CRITICAL;

        if (level == ALERT_NONE) {
            led_normal();
            tm_printf((UB*)"[Alert] OK  | State: %s | HR: %.0f | SpO2: %.1f%%\n",
                      STATE_NAMES[r->state_id], r->hr_bpm, r->spo2_pct);
        } else if (level == ALERT_WARNING) {
            led_alert();
            tm_printf((UB*)"[Alert] *** WARNING  | State: %s | HR: %.0f | SpO2: %.1f%% | Temp: %.1f C\n",
                      STATE_NAMES[r->state_id], r->hr_bpm, r->spo2_pct, r->temp_mean_c);
            tk_set_flg(flg_system_sync, FLG_ALERT_ACTIVE);
        } else {
            led_alert();
            tm_printf((UB*)"[Alert] !!! CRITICAL | State: %s | HR: %.0f | SpO2: %.1f%% | Temp: %.1f C\n",
                      STATE_NAMES[r->state_id], r->hr_bpm, r->spo2_pct, r->temp_mean_c);
            tk_set_flg(flg_system_sync, FLG_ALERT_ACTIVE);
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Task_CommBLE  (Priority 14)
 *
 *  Reads FeatureResult_t from mbx_telemetry, formats a BLE-style payload,
 *  prints it to UART (stand-in for actual BLE GATT write on real hardware).
 * ══════════════════════════════════════════════════════════════════════════ */
void Task_CommBLE(INT stacd, void *exinf)
{
    (void)stacd; (void)exinf;
    tm_printf((UB*)"[BLE] Started — GATT telemetry stub\n");

    for (;;) {
        T_MSG *pmsg;
        tk_rcv_mbx(mbx_telemetry, &pmsg, TMO_FEVR);
        FeatureResult_t *r = (FeatureResult_t *)pmsg;

        /* GATT characteristic 0x2A37 (Heart Rate Measurement) format:
         *   Flags byte | HR uint8 | Energy uint16 | RR uint16 ...          */
        uint8_t ble_hrm[4];
        ble_hrm[0] = 0x00;                          /* Flags: HR as uint8  */
        ble_hrm[1] = (uint8_t)(r->hr_bpm);
        ble_hrm[2] = (uint8_t)(r->state_id);
        ble_hrm[3] = (uint8_t)(r->anomaly_score * 255.0f);

        /* On real hardware: aci_gatt_update_char_value(handle, 0, 4, ble_hrm) */
        tm_printf((UB*)"[BLE] HRM: HR=%d state=%d anomaly=%d/255\n",
                  ble_hrm[1], ble_hrm[2], ble_hrm[3]);

        tk_dly_tsk(1000);  /* BLE notify rate: 1 Hz                        */
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Task_CommWiFi  (Priority 16 — lowest)
 *
 *  Formats a JSON telemetry frame and transmits over UART2 (Wi-Fi module).
 *  On the STM32N6570-DK this connects to an ESP-AT Wi-Fi co-processor.
 * ══════════════════════════════════════════════════════════════════════════ */
void Task_CommWiFi(INT stacd, void *exinf)
{
    (void)stacd; (void)exinf;
    tm_printf((UB*)"[WiFi] Started — JSON telemetry stub\n");

    static char json_buf[256];

    for (;;) {
        T_MSG *pmsg;
        tk_rcv_mbx(mbx_telemetry, &pmsg, TMO_FEVR);
        FeatureResult_t *r = (FeatureResult_t *)pmsg;

        int len = snprintf(json_buf, sizeof(json_buf),
            "{\"ts\":%lu,\"hr\":%.1f,\"spo2\":%.1f,"
            "\"temp\":%.2f,\"state\":%d,\"anomaly\":%.3f}\r\n",
            r->window_start_ms,
            r->hr_bpm, r->spo2_pct, r->temp_mean_c,
            r->state_id, r->anomaly_score);

        /* On real hardware: HAL_UART_Transmit(&huart2, (uint8_t*)json_buf, len, 100) */
        tm_printf((UB*)"[WiFi] TX %d bytes: %s", len, json_buf);

        tk_dly_tsk(5000);  /* Wi-Fi push rate: 0.2 Hz (5 s)               */
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  usermain — µT-Kernel 3.0 application entry point
 *  Called by the BSP after the RTOS is initialised.
 * ══════════════════════════════════════════════════════════════════════════ */
EXPORT INT usermain(void)
{
    tm_printf((UB*)"\n");
    tm_printf((UB*)"========================================================\n");
    tm_printf((UB*)"  TRON Contest — µT-Kernel 3.0 Wearable Health Monitor  \n");
    tm_printf((UB*)"  Board : STM32N6570-DK  (Cortex-M55 @ 600 MHz)         \n");
    tm_printf((UB*)"  BSP   : mtk3bsp2_stm32n657 v01.00.B8                  \n");
    tm_printf((UB*)"========================================================\n");

    /* ── Init sensors ───────────────────────────────────────────────────── */
    if (!HAL_Sensors_Init()) {
        tm_printf((UB*)"[FATAL] Sensor init failed — check I2C wiring\n");
        return -1;
    }
    tm_printf((UB*)"[Init] All sensors initialised OK\n");

    /* ── Create IPC objects ─────────────────────────────────────────────── */
    mbx_ingest_to_prep = create_mailbox();
    mbx_prep_to_ml     = create_mailbox();
    mbx_ml_to_alert    = create_mailbox();
    mbx_telemetry      = create_mailbox();
    flg_system_sync    = create_event_flag();

    if (mbx_ingest_to_prep < E_OK || mbx_prep_to_ml < E_OK ||
        mbx_ml_to_alert    < E_OK || mbx_telemetry  < E_OK ||
        flg_system_sync    < E_OK) {
        tm_printf((UB*)"[FATAL] IPC object creation failed\n");
        return -1;
    }
    tm_printf((UB*)"[Init] 4 mailboxes + 1 event flag created\n");

    /* ── Task creation table ─────────────────────────────────────────────── */
    typedef struct {
        const char *name;
        FP          fn;
        PRI         pri;
        INT         stksz;
    } TaskDef_t;

    static const TaskDef_t TASKS[] = {
        { "SensorIngest", Task_SensorIngest,  3, STACK_INGEST  },
        { "AlertManager", Task_AlertManager,  5, STACK_ALERT   },
        { "Preprocessor", Task_Preprocessor,  8, STACK_PREPROC },
        { "FeatureML",    Task_FeatureML,     10, STACK_FEATURE },
        { "CommBLE",      Task_CommBLE,       14, STACK_BLE     },
        { "CommWiFi",     Task_CommWiFi,      16, STACK_WIFI    },
    };

    for (int i = 0; i < (int)(sizeof(TASKS)/sizeof(TASKS[0])); i++) {
        T_CTSK ctsk = {
            .tskatr  = TA_HLNG | TA_RNG3,
            .task    = TASKS[i].fn,
            .itskpri = TASKS[i].pri,
            .stksz   = TASKS[i].stksz,
            .exinf   = NULL,
        };
        ID tid = tk_cre_tsk(&ctsk);
        if (tid < E_OK) {
            tm_printf((UB*)"[FATAL] Failed to create task %s (err %d)\n",
                      TASKS[i].name, tid);
            return -1;
        }
        tk_sta_tsk(tid, 0);
        tm_printf((UB*)"[Init] Task %-14s pri=%2d  stack=%4d B  tid=%d\n",
                  TASKS[i].name, TASKS[i].pri, TASKS[i].stksz, tid);
    }

    tm_printf((UB*)"[Init] All tasks started — scheduler running\n");
    tm_printf((UB*)"========================================================\n\n");

    /* Suspend usermain — the scheduler owns everything from here */
    tk_slp_tsk(TMO_FEVR);
    return 0;
}
