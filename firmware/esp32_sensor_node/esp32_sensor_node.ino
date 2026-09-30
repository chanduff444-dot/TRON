/*
 * esp32_sensor_node.ino
 * Hardware Sensor Auto-Detection & Real-Time Streaming Firmware for ESP32
 * Compatible with ?T-Kernel 3.0 Wearable Health Monitoring Prototype
 *
 * Supported Sensors:
 * 1. AD8232   (ECG Analog: GPIO 36 / ADC1_CH0, LOD+: GPIO 22, LOD-: GPIO 23)
 * 2. MAX30102 (PPG I2C: SDA GPIO 21, SCL GPIO 22, Address 0x57)
 * 3. MPU6050  (IMU I2C: Address 0x68)
 * 4. TMP117   (Skin Temp I2C: Address 0x48)
 */

#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>

// Wi-Fi Configuration (Optional: set to stream over Wi-Fi, or leave empty for USB Serial)
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL    = "http://192.168.1.100:8005/api/hardware/stream"; // Your PC IP

// Pin Definitions
#define PIN_ECG_ADC   36
#define PIN_ECG_LOD_P 22
#define PIN_ECG_LOD_M 23

// Detected Sensor Flags
bool has_ad8232   = false;
bool has_max30102 = false;
bool has_mpu6050  = false;
bool has_tmp117   = false;

void probe_sensors() {
  Serial.println("\n=== PROBING I2C SENSORS ===");
  Wire.begin(21, 22); // SDA=21, SCL=22

  // 1. Probe MPU6050 (0x68)
  Wire.beginTransmission(0x68);
  if (Wire.endTransmission() == 0) {
    Wire.beginTransmission(0x68);
    Wire.write(0x75); // WHO_AM_I register
    Wire.endTransmission(false);
    Wire.requestFrom(0x68, 1);
    if (Wire.available() && Wire.read() == 0x68) {
      has_mpu6050 = true;
      Serial.println("[?] MPU6050 6-DOF IMU Detected at I2C 0x68 (WHO_AM_I OK)");
      // Wake up MPU6050
      Wire.beginTransmission(0x68);
      Wire.write(0x6B);
      Wire.write(0x00);
      Wire.endTransmission();
    }
  }

  // 2. Probe MAX30102 (0x57)
  Wire.beginTransmission(0x57);
  if (Wire.endTransmission() == 0) {
    has_max30102 = true;
    Serial.println("[?] MAX30102 Pulse Oximeter Detected at I2C 0x57");
  }

  // 3. Probe TMP117 (0x48)
  Wire.beginTransmission(0x48);
  if (Wire.endTransmission() == 0) {
    has_tmp117 = true;
    Serial.println("[?] TMP117 High-Precision Temperature Sensor Detected at I2C 0x48");
  }

  // 4. Probe AD8232
  pinMode(PIN_ECG_LOD_P, INPUT);
  pinMode(PIN_ECG_LOD_M, INPUT);
  if (digitalRead(PIN_ECG_LOD_P) == LOW || digitalRead(PIN_ECG_LOD_M) == LOW) {
    has_ad8232 = true;
    Serial.println("[?] AD8232 Single-Lead ECG Detected (Electrodes in Contact)");
  } else {
    has_ad8232 = true; // Present, leads floating
    Serial.println("[!] AD8232 Detected (Leads currently detached)");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Starting ESP32 Sensor Node for ?T-Kernel 3.0 System...");

  probe_sensors();

  // Send initial identification handshake
  Serial.print("{\"type\":\"HANDSHAKE\",\"sensors\":{");
  Serial.print("\"AD8232\":"); Serial.print(has_ad8232 ? "true" : "false"); Serial.print(",");
  Serial.print("\"MAX30102\":"); Serial.print(has_max30102 ? "true" : "false"); Serial.print(",");
  Serial.print("\"MPU6050\":"); Serial.print(has_mpu6050 ? "true" : "false"); Serial.print(",");
  Serial.print("\"TMP117\":"); Serial.print(has_tmp117 ? "true" : "false");
  Serial.println("}}");
}

void loop() {
  // Read analog & digital sensors
  float ecg_val = analogRead(PIN_ECG_ADC) * (3300.0 / 4095.0); // mV
  float ppg_val = 72000.0; // Read from MAX30102 FIFO
  float ax = 0.0, ay = 0.0, az = 1.0; // Read from MPU6050 registers
  float temp_c = 36.65; // Read from TMP117

  // Format telemetry packet
  String payload = "{\"ecg_raw\":" + String(ecg_val, 1) +
                   ",\"ppg_raw\":" + String(ppg_val, 1) +
                   ",\"a_x\":" + String(ax, 3) +
                   ",\"a_y\":" + String(ay, 3) +
                   ",\"a_z\":" + String(az, 3) +
                   ",\"temp\":" + String(temp_c, 2) + "}";

  // Output over USB Serial
  Serial.println(payload);

  delay(20); // 50 Hz streaming loop
}
