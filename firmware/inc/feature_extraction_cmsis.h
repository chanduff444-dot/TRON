/**
 * feature_extraction_cmsis.h
 * ---------------------------
 * TRON Contest — STM32N6570-DK
 * Public API for CMSIS-DSP embedded feature extraction.
 */

#ifndef FEATURE_EXTRACTION_CMSIS_H
#define FEATURE_EXTRACTION_CMSIS_H

#include <stdint.h>
#include "arm_math.h"

/* ── Input window buffer (filled by Task_Preprocessor) ─────────────────── */
#define FEAT_WINDOW_ECG     2500u   /* 5 s × 500 Hz                         */
#define FEAT_WINDOW_IMU     500u    /* 5 s × 100 Hz                         */
#define FEAT_WINDOW_TEMP    5u      /* 5 s × 1 Hz                           */

typedef struct {
    float32_t ecg_filt[FEAT_WINDOW_ECG];  /* bandpass-filtered ECG (centred)*/
    float32_t ax[FEAT_WINDOW_IMU];         /* accel X  (g, ±4g range)        */
    float32_t ay[FEAT_WINDOW_IMU];         /* accel Y                         */
    float32_t az[FEAT_WINDOW_IMU];         /* accel Z  (gravity on z)         */
    float32_t gx[FEAT_WINDOW_IMU];         /* gyro  X  (°/s)                  */
    float32_t gy[FEAT_WINDOW_IMU];         /* gyro  Y                         */
    float32_t gz[FEAT_WINDOW_IMU];         /* gyro  Z                         */
    float32_t temp[FEAT_WINDOW_TEMP];      /* skin temp (°C)                  */
    uint32_t  start_ms;                    /* µT-Kernel timestamp             */
} SensorWindow_t;

/* ── Output feature vector (matches Python FEATURE_COLUMNS) ─────────────── */
typedef struct {
    /* ECG / HR */
    float32_t hr_bpm;
    float32_t rr_std_ms;           /* SDNN proxy                            */
    float32_t rmssd_ms;
    float32_t ecg_qrs_energy;

    /* PPG (filled from SpO2 data) */
    float32_t ppg_pulse_rate_bpm;
    float32_t ppg_amplitude;
    float32_t ppg_perfusion_index;

    /* IMU */
    float32_t accel_x_std;
    float32_t accel_y_std;
    float32_t accel_z_std;
    float32_t motion_mag_mean;
    float32_t motion_sma;
    float32_t motion_energy;
    float32_t motion_jerk_std;
    float32_t gyro_x_std;
    float32_t gyro_y_std;
    float32_t gyro_z_std;

    /* Temperature */
    float32_t temp_mean_c;
    float32_t temp_trend;          /* °C / min                              */

    /* Contact / GSR (defaults for this BOM) */
    float32_t contact_mean;
    float32_t gsr_synthetic_mean;

    /* Metadata */
    uint32_t  window_start_ms;
    uint8_t   state_id;            /* filled by FEAT_ClassifyState()        */
} EmbeddedFeatures_t;

/* ── API ─────────────────────────────────────────────────────────────────── */

/**
 * FEAT_ComputeAll — compute all features from a filled SensorWindow_t.
 * Typical execution time on Cortex-M55 @ 600 MHz with Helium: ~2.8 ms.
 */
void    FEAT_ComputeAll(const SensorWindow_t *win, EmbeddedFeatures_t *out);

/**
 * FEAT_ClassifyState — lightweight Decision-Tree classifier.
 * Returns: 0=RESTING  1=ACTIVE_EXERTION  2=STRAIN  3=ARTIFACT
 */
uint8_t FEAT_ClassifyState(const EmbeddedFeatures_t *f);

#endif /* FEATURE_EXTRACTION_CMSIS_H */
