/**
 * wearable_sensors.h
 * ------------------
 * TRON Contest — STM32N6570-DK / µT-Kernel 3.0
 *
 * Shared sensor packet struct and constants used across all RTOS tasks.
 * Sensor BOM:
 *   AD8232  — ECG front-end  (ADC1, DMA, 500 Hz)
 *   MAX30102 — PPG oximeter  (I2C1, interrupt-driven)
 *   MPU6050  — 6-DOF IMU     (I2C1, FIFO 100 Hz)
 *   TMP117   — Skin temp     (I2C1, 1 Hz polling)
 */

#ifndef WEARABLE_SENSORS_H
#define WEARABLE_SENSORS_H

#include <stdint.h>
#include <stdbool.h>

/* ── Sampling constants ─────────────────────────────────────────────────── */
#define SENSOR_ECG_FS_HZ        500u    /* AD8232 via ADC1 DMA              */
#define SENSOR_PPG_FS_HZ        100u    /* MAX30102 FIFO readout rate        */
#define SENSOR_IMU_FS_HZ        100u    /* MPU6050 FIFO output rate          */
#define SENSOR_TEMP_FS_HZ       1u      /* TMP117 one-shot, 1 s period       */
#define SENSOR_WINDOW_SAMPLES   2500u   /* 5 s × 500 Hz (ECG governs)        */

/* ── I2C addresses ──────────────────────────────────────────────────────── */
#define MAX30102_I2C_ADDR       0x57u   /* 7-bit                             */
#define MPU6050_I2C_ADDR        0x68u   /* 7-bit, AD0=GND                    */
#define TMP117_I2C_ADDR         0x48u   /* 7-bit, ADD0=GND                   */

/* ── MAX30102 register map (subset) ─────────────────────────────────────── */
#define MAX30102_REG_INT_STATUS1    0x00u
#define MAX30102_REG_FIFO_WR_PTR    0x04u
#define MAX30102_REG_FIFO_RD_PTR    0x06u
#define MAX30102_REG_FIFO_DATA      0x07u
#define MAX30102_REG_MODE_CFG       0x09u
#define MAX30102_REG_SPO2_CFG       0x0Au
#define MAX30102_REG_LED1_PA        0x0Cu   /* Red LED pulse amplitude       */
#define MAX30102_REG_LED2_PA        0x0Du   /* IR  LED pulse amplitude       */

/* ── MPU6050 register map (subset) ─────────────────────────────────────── */
#define MPU6050_REG_SMPLRT_DIV      0x19u
#define MPU6050_REG_CONFIG          0x1Au
#define MPU6050_REG_GYRO_CFG        0x1Bu
#define MPU6050_REG_ACCEL_CFG       0x1Cu
#define MPU6050_REG_FIFO_EN         0x23u
#define MPU6050_REG_INT_ENABLE      0x38u
#define MPU6050_REG_ACCEL_XOUT_H    0x3Bu
#define MPU6050_REG_USER_CTRL       0x6Au
#define MPU6050_REG_PWR_MGMT_1      0x6Bu
#define MPU6050_REG_FIFO_COUNT_H    0x72u
#define MPU6050_REG_FIFO_R_W        0x74u

/* ── TMP117 register map ────────────────────────────────────────────────── */
#define TMP117_REG_TEMP_RESULT      0x00u   /* 16-bit, 1 LSB = 7.8125e-3 °C */
#define TMP117_REG_CONFIG           0x01u
#define TMP117_REG_DEVICE_ID        0x0Fu   /* should read 0x0117            */
#define TMP117_TEMP_RESOLUTION      0.0078125f  /* °C per LSB                */

/* ── Raw sensor packet (produced by Task_SensorIngest) ─────────────────── */
typedef struct {
    uint32_t timestamp_ms;          /* tk_get_tim() snapshot                 */

    /* AD8232 — ECG (raw 12-bit ADC counts, signed baseline-shifted)        */
    int16_t  ecg_raw;

    /* MAX30102 — PPG (18-bit red + IR, stored as uint32_t)                 */
    uint32_t ppg_red;
    uint32_t ppg_ir;
    float    spo2_pct;              /* computed by MAX30102 algorithm        */

    /* MPU6050 — IMU (raw 16-bit counts, ±4g / ±500°/s full-scale)         */
    int16_t  accel_x_raw;
    int16_t  accel_y_raw;
    int16_t  accel_z_raw;
    int16_t  gyro_x_raw;
    int16_t  gyro_y_raw;
    int16_t  gyro_z_raw;

    /* TMP117 — skin temperature (°C, converted on read)                    */
    float    skin_temp_c;

    bool     valid;                 /* false if any sensor read failed       */
} RawSensorPacket_t;

/* ── Processed feature vector (produced by Task_FeatureML) ─────────────── */
typedef struct {
    float    hr_bpm;
    float    rr_std_ms;
    float    rmssd_ms;
    float    spo2_pct;
    float    accel_x_std;
    float    accel_y_std;
    float    accel_z_std;
    float    motion_energy;
    float    motion_jerk_std;
    float    gyro_x_std;
    float    gyro_y_std;
    float    gyro_z_std;
    float    temp_mean_c;
    float    temp_trend;            /* °C / min                              */
    uint8_t  state_id;             /* 0=REST 1=ACTIVE 2=STRAIN 3=ARTIFACT   */
    float    anomaly_score;        /* 0.0 … 1.0                             */
    uint32_t window_start_ms;
} FeatureResult_t;

/* ── Alert severity ─────────────────────────────────────────────────────── */
typedef enum {
    ALERT_NONE     = 0,
    ALERT_WARNING  = 1,
    ALERT_CRITICAL = 2
} AlertLevel_t;

typedef struct {
    AlertLevel_t level;
    char         message[64];
    uint32_t     timestamp_ms;
} AlertEvent_t;

#endif /* WEARABLE_SENSORS_H */
