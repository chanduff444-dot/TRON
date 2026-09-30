/**
 * sensor_hal_stm32n6.h
 * --------------------
 * TRON Contest — STM32N6570-DK
 * Hardware Abstraction Layer: I2C + ADC sensor read functions.
 * All functions are safe to call from µT-Kernel 3.0 tasks.
 */

#ifndef SENSOR_HAL_STM32N6_H
#define SENSOR_HAL_STM32N6_H

#include <stdint.h>
#include <stdbool.h>
#include "wearable_sensors.h"

/* ── Init (call once from usermain before creating tasks) ───────────────── */
bool HAL_Sensors_Init(void);

/* ── AD8232 ECG — ADC1 DMA single-sample read ───────────────────────────── */
int16_t  HAL_ECG_ReadSample(void);

/* ── MAX30102 — FIFO burst read (returns number of new samples read) ────── */
uint8_t  HAL_MAX30102_ReadFIFO(uint32_t *red_buf, uint32_t *ir_buf, uint8_t max_samples);
float    HAL_MAX30102_ComputeSpO2(const uint32_t *ir_buf, uint8_t n);

/* ── MPU6050 — single register-burst read of accel + gyro ──────────────── */
bool     HAL_MPU6050_ReadRaw(int16_t *ax, int16_t *ay, int16_t *az,
                              int16_t *gx, int16_t *gy, int16_t *gz);

/* ── TMP117 — blocking one-shot temperature read ────────────────────────── */
float    HAL_TMP117_ReadTempC(void);

/* ── Composite: fill one RawSensorPacket_t (used by Task_SensorIngest) ─── */
bool     HAL_Sensors_ReadPacket(RawSensorPacket_t *pkt);

#endif /* SENSOR_HAL_STM32N6_H */
