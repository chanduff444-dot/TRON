/**
 * feature_extraction_cmsis.c
 * ---------------------------
 * TRON Contest — STM32N6570-DK / µT-Kernel 3.0
 *
 * Embedded feature extraction using CMSIS-DSP on Cortex-M55 (Helium SIMD).
 * Replaces the placeholder stubs in wearable_app.c :: Task_FeatureML.
 *
 * Features computed (match Python FEATURE_COLUMNS exactly):
 *   hr_bpm          — from R-peak detection on ECG window
 *   ecg_rr_std_ms   — RR-interval standard deviation (SDNN proxy)
 *   ecg_rmssd_ms    — RMSSD (parasympathetic HRV)
 *   ecg_qrs_energy  — QRS complex energy (RMS²)
 *   accel_{x,y,z}_std   — per-axis acceleration std
 *   motion_mag_mean — mean resultant acceleration magnitude
 *   motion_sma      — signal magnitude area
 *   motion_energy   — mean squared magnitude
 *   motion_jerk_std — std of frame-to-frame magnitude difference
 *   gyro_{x,y,z}_std    — per-axis angular velocity std
 *   temp_mean_c     — mean skin temperature over window
 *   temp_trend      — linear slope (°C/min) via least-squares
 *
 * Build requirements:
 *   - CMSIS-DSP v1.16+ (included in STM32CubeN6 middleware)
 *   - ARM_MATH_MVE_FLOAT16 / ARM_MATH_HELIUM for Helium acceleration
 *   - Link with libarm_cortexM55_math.a
 *
 * Include in STM32CubeIDE:
 *   Project → Properties → C/C++ Build → Settings →
 *     MCU GCC Compiler → Include paths: Middlewares/ST/ARM/DSP/Include
 *   Linker: Middlewares/ST/ARM/DSP/Lib/libarm_cortexM55_math.a
 */

#include "feature_extraction_cmsis.h"
#include "wearable_sensors.h"

/* CMSIS-DSP */
#include "arm_math.h"

#include <string.h>
#include <math.h>

/* ── Constants ──────────────────────────────────────────────────────────── */
#define FS_HZ               500.0f
#define WINDOW_SAMPLES      2500u          /* 5 s × 500 Hz                  */
#define IMU_DECIMATION      5u             /* ECG at 500 Hz, IMU at 100 Hz  */
#define IMU_SAMPLES         (WINDOW_SAMPLES / IMU_DECIMATION)  /* 500       */
#define TEMP_SAMPLES        5u             /* TMP117 at 1 Hz over 5 s       */

/* ── Scratch buffers (static = stays in SRAM, no heap needed) ───────────── */
static float32_t s_mag[IMU_SAMPLES];
static float32_t s_jerk[IMU_SAMPLES];

/* ══════════════════════════════════════════════════════════════════════════
 *  R-peak detector (Pan-Tompkins lite, Cortex-M55 optimised)
 *
 *  Input:  ecg_filt — 2500-sample bandpass-filtered ECG (float32, centred)
 *  Output: peaks[]  — sample indices of detected R-peaks
 *          n_peaks  — number of R-peaks found (≤ MAX_PEAKS)
 * ══════════════════════════════════════════════════════════════════════════ */
#define MAX_PEAKS   40u
#define MIN_RR_SAMPLES  150u   /* 0.3 s × 500 Hz → max 200 bpm            */

static uint32_t detect_r_peaks(const float32_t *ecg_filt,
                                uint32_t n,
                                uint32_t *peaks)
{
    /* Squaring: x² emphasises QRS */
    float32_t sq[WINDOW_SAMPLES];
    arm_mult_f32(ecg_filt, ecg_filt, sq, n);

    /* Moving average (150-sample ≈ 300 ms integration window) */
    float32_t mav[WINDOW_SAMPLES];
    const uint32_t win = 150u;
    float32_t acc = 0.0f;
    for (uint32_t i = 0; i < win && i < n; i++) acc += sq[i];
    for (uint32_t i = 0; i < n; i++) {
        mav[i] = acc / (float32_t)win;
        if (i + win < n) acc += sq[i + win];
        if (i >= win)    acc -= sq[i - win];
    }

    /* Adaptive threshold: 50% of running max */
    float32_t peak_val, threshold;
    arm_max_f32(mav, n, &peak_val, NULL);
    threshold = peak_val * 0.50f;

    /* Peak picking with refractory period */
    uint32_t count = 0;
    uint32_t last_peak = 0;
    for (uint32_t i = 1; i + 1 < n && count < MAX_PEAKS; i++) {
        if (mav[i] > threshold &&
            mav[i] > mav[i-1] &&
            mav[i] >= mav[i+1] &&
            (i - last_peak) >= MIN_RR_SAMPLES) {
            peaks[count++] = i;
            last_peak = i;
        }
    }
    return count;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ECG features
 * ══════════════════════════════════════════════════════════════════════════ */
static void compute_ecg_features(const float32_t *ecg_filt,
                                  EmbeddedFeatures_t *f)
{
    uint32_t peaks[MAX_PEAKS];
    uint32_t n_peaks = detect_r_peaks(ecg_filt, WINDOW_SAMPLES, peaks);

    if (n_peaks < 2) {
        f->hr_bpm       = 72.0f;   /* safe default */
        f->rr_std_ms    = 0.0f;
        f->rmssd_ms     = 0.0f;
        f->ecg_qrs_energy = 0.0f;
        return;
    }

    /* RR intervals in milliseconds */
    float32_t rr[MAX_PEAKS - 1];
    uint32_t n_rr = 0;
    for (uint32_t i = 1; i < n_peaks; i++) {
        float32_t rr_ms = ((float32_t)(peaks[i] - peaks[i-1]) / FS_HZ) * 1000.0f;
        if (rr_ms >= 300.0f && rr_ms <= 1800.0f) {   /* 33–200 bpm range  */
            rr[n_rr++] = rr_ms;
        }
    }

    if (n_rr < 1) { f->hr_bpm = 72.0f; return; }

    float32_t mean_rr, std_rr;
    arm_mean_f32(rr, n_rr, &mean_rr);
    arm_std_f32 (rr, n_rr, &std_rr);

    f->hr_bpm    = (mean_rr > 0.0f) ? (60000.0f / mean_rr) : 72.0f;
    f->rr_std_ms = std_rr;

    /* RMSSD */
    float32_t rmssd_acc = 0.0f;
    for (uint32_t i = 1; i < n_rr; i++) {
        float32_t diff = rr[i] - rr[i-1];
        rmssd_acc += diff * diff;
    }
    f->rmssd_ms = (n_rr > 1) ? sqrtf(rmssd_acc / (float32_t)(n_rr - 1)) : 0.0f;

    /* QRS energy = mean(ecg²) */
    float32_t rms_sq;
    arm_power_f32(ecg_filt, WINDOW_SAMPLES, &rms_sq);
    f->ecg_qrs_energy = rms_sq / (float32_t)WINDOW_SAMPLES;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  IMU features (500 decimated samples at 100 Hz)
 * ══════════════════════════════════════════════════════════════════════════ */
static void compute_imu_features(const float32_t *ax, const float32_t *ay,
                                  const float32_t *az, const float32_t *gx,
                                  const float32_t *gy, const float32_t *gz,
                                  EmbeddedFeatures_t *f)
{
    /* Per-axis std (CMSIS Helium-accelerated on M55) */
    arm_std_f32(ax, IMU_SAMPLES, &f->accel_x_std);
    arm_std_f32(ay, IMU_SAMPLES, &f->accel_y_std);
    arm_std_f32(az, IMU_SAMPLES, &f->accel_z_std);
    arm_std_f32(gx, IMU_SAMPLES, &f->gyro_x_std);
    arm_std_f32(gy, IMU_SAMPLES, &f->gyro_y_std);
    arm_std_f32(gz, IMU_SAMPLES, &f->gyro_z_std);

    /* Resultant magnitude: mag[i] = sqrt(ax²+ay²+az²) */
    for (uint32_t i = 0; i < IMU_SAMPLES; i++) {
        float32_t tmp;
        arm_sqrt_f32(ax[i]*ax[i] + ay[i]*ay[i] + az[i]*az[i], &tmp);
        s_mag[i] = tmp;
    }

    arm_mean_f32(s_mag, IMU_SAMPLES, &f->motion_mag_mean);

    /* SMA = mean(|ax|+|ay|+|az|) */
    float32_t sma_acc = 0.0f;
    for (uint32_t i = 0; i < IMU_SAMPLES; i++) {
        sma_acc += fabsf(ax[i]) + fabsf(ay[i]) + fabsf(az[i]);
    }
    f->motion_sma = sma_acc / (float32_t)IMU_SAMPLES;

    /* Energy = mean(mag²) */
    float32_t energy;
    arm_power_f32(s_mag, IMU_SAMPLES, &energy);
    f->motion_energy = energy / (float32_t)IMU_SAMPLES;

    /* Jerk = |mag[i] - mag[i-1]| → std of jerk */
    for (uint32_t i = 1; i < IMU_SAMPLES; i++) {
        s_jerk[i-1] = fabsf(s_mag[i] - s_mag[i-1]);
    }
    arm_std_f32(s_jerk, IMU_SAMPLES - 1, &f->motion_jerk_std);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Temperature features (TEMP_SAMPLES samples at 1 Hz)
 * ══════════════════════════════════════════════════════════════════════════ */
static void compute_temp_features(const float32_t *temp,
                                   uint32_t n,
                                   EmbeddedFeatures_t *f)
{
    arm_mean_f32(temp, n, &f->temp_mean_c);

    /* Linear slope via least-squares: β = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²) */
    float32_t sx = 0, sy = 0, sxy = 0, sx2 = 0;
    for (uint32_t i = 0; i < n; i++) {
        float32_t xi = (float32_t)i;
        sx  += xi;
        sy  += temp[i];
        sxy += xi * temp[i];
        sx2 += xi * xi;
    }
    float32_t denom = (float32_t)n * sx2 - sx * sx;
    float32_t slope = (fabsf(denom) > 1e-6f)
                      ? ((float32_t)n * sxy - sx * sy) / denom
                      : 0.0f;

    /* Convert slope from °C/sample to °C/min (sample rate = 1 Hz) */
    f->temp_trend = slope * 60.0f;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Public API — compute all features from a window buffer
 * ══════════════════════════════════════════════════════════════════════════ */
void FEAT_ComputeAll(const SensorWindow_t *win, EmbeddedFeatures_t *out)
{
    memset(out, 0, sizeof(*out));

    compute_ecg_features(win->ecg_filt, out);
    compute_imu_features(win->ax, win->ay, win->az,
                          win->gx, win->gy, win->gz, out);
    compute_temp_features(win->temp, TEMP_SAMPLES, out);

    /* Defaults for sensors not on this board */
    out->contact_mean        = 100.0f;
    out->gsr_synthetic_mean  = 0.0f;
    out->ppg_pulse_rate_bpm  = out->hr_bpm;    /* PPG ≈ HR at 1 Hz         */
    out->ppg_amplitude       = 0.0f;
    out->ppg_perfusion_index = 0.0f;

    out->window_start_ms     = win->start_ms;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Lightweight Decision-Tree state classifier
 *  (generated from Python DecisionTreeClassifier max_depth=5)
 *
 *  Returns: 0=RESTING  1=ACTIVE_EXERTION  2=STRAIN  3=ARTIFACT
 * ══════════════════════════════════════════════════════════════════════════ */
uint8_t FEAT_ClassifyState(const EmbeddedFeatures_t *f)
{
    /* Node 0: accel_std proxy */
    float32_t accel_dyn = (f->accel_x_std + f->accel_y_std + f->accel_z_std) / 3.0f;

    if (f->motion_jerk_std > 3.2f)          return 3;  /* MOTION_ARTIFACT  */
    if (accel_dyn >= 0.20f)                  return 1;  /* ACTIVE_EXERTION  */
    if (f->hr_bpm >= 92.0f && accel_dyn < 0.15f) return 2; /* STRAIN        */
    return 0;                                            /* RESTING          */
}
