/*
  Combo: MPU‑6050 IMU  +  Two FSRs (resistance only)
  FSR wiring (per sensor):

        5 V ── FSR ──+── A0 / A1
                     |
                  10 kΩ
                     |
                    GND
  Change VIN to 3.3 if you run the divider from 3 V logic.
*/

#include <Wire.h>

/* ----------  MPU‑6050 section ---------- */
#define MPU_ADDR          0x68
#define REG_PWR_MGMT_1    0x6B
#define REG_ACCEL_XOUT_H  0x3B
#define REG_CONFIG        0x1A
#define REG_GYRO_CONFIG   0x1B
#define REG_ACCEL_CONFIG  0x1C

const float G_SI   = 9.80665f;        // m / s²
const float DEG2RAD = 0.01745329252f; // rad / deg

int16_t axRaw, ayRaw, azRaw, gxRaw, gyRaw, gzRaw;
int16_t axOff=0, ayOff=0, azOff=0, gxOff=0, gyOff=0, gzOff=0;
float   ax_ms2, ay_ms2, az_ms2, gx_rad, gy_rad, gz_rad;

void initMPU() {
  Wire.beginTransmission(MPU_ADDR); Wire.write(REG_PWR_MGMT_1); Wire.write(0); Wire.endTransmission(true);
  Wire.beginTransmission(MPU_ADDR); Wire.write(REG_GYRO_CONFIG);  Wire.write(0x00); Wire.endTransmission(true); // ±250 °/s
  Wire.beginTransmission(MPU_ADDR); Wire.write(REG_ACCEL_CONFIG); Wire.write(0x00); Wire.endTransmission(true); // ±2 g
  Wire.beginTransmission(MPU_ADDR); Wire.write(REG_CONFIG);       Wire.write(0x03); Wire.endTransmission(true); // DLPF ≈43 Hz
}

void calibrateMPU() {
  const int N = 500;
  long axSum=0, aySum=0, azSum=0, gxSum=0, gySum=0, gzSum=0;

  Serial.println(F("Keep IMU still... calibrating (≈2 s)"));
  delay(2000);
  for (int i=0;i<N;i++) {
    Wire.beginTransmission(MPU_ADDR); Wire.write(REG_ACCEL_XOUT_H); Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR,14,true);
    int16_t ax = Wire.read()<<8 | Wire.read();
    int16_t ay = Wire.read()<<8 | Wire.read();
    int16_t az = Wire.read()<<8 | Wire.read();
    Wire.read(); Wire.read(); // skip temp
    int16_t gx = Wire.read()<<8 | Wire.read();
    int16_t gy = Wire.read()<<8 | Wire.read();
    int16_t gz = Wire.read()<<8 | Wire.read();

    axSum+=ax; aySum+=ay; azSum+=(az-16384); // subtract 1 g on Z
    gxSum+=gx; gySum+=gy; gzSum+=gz;
    delay(2);
  }
  axOff=axSum/N; ayOff=aySum/N; azOff=azSum/N;
  gxOff=gxSum/N; gyOff=gySum/N; gzOff=gzSum/N;
}

void readMPU() {
  Wire.beginTransmission(MPU_ADDR); Wire.write(REG_ACCEL_XOUT_H); Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR,14,true);

  axRaw = Wire.read()<<8 | Wire.read();
  ayRaw = Wire.read()<<8 | Wire.read();
  azRaw = Wire.read()<<8 | Wire.read();
  Wire.read(); Wire.read();
  gxRaw = Wire.read()<<8 | Wire.read();
  gyRaw = Wire.read()<<8 | Wire.read();
  gzRaw = Wire.read()<<8 | Wire.read();

  axRaw -= axOff; ayRaw -= ayOff; azRaw -= azOff;
  gxRaw -= gxOff; gyRaw -= gyOff; gzRaw -= gzOff;

  ax_ms2 = (axRaw/16384.0f)*G_SI;
  ay_ms2 = (ayRaw/16384.0f)*G_SI;
  az_ms2 = (azRaw/16384.0f)*G_SI;
  gx_rad = (gxRaw/131.0f)*DEG2RAD;
  gy_rad = (gyRaw/131.0f)*DEG2RAD;
  gz_rad = (gzRaw/131.0f)*DEG2RAD;
}

/* ----------  FSR section ---------- */
const int FSR1_PIN = A0;
const int FSR2_PIN = A1;
const float R_PULL = 10000.0f; // 10 kΩ
const float VIN    = 5.0f;     // change to 3.3 if needed

float getResistance(int raw) {
  if (raw == 0) return -1.0f;            // open / infinite
  float vout = raw * (VIN / 1023.0f);
  return (vout * R_PULL) / (VIN - vout); // divider formula
}

/* ----------  Setup & loop ---------- */
void setup() {
  Serial.begin(115200);
  Wire.begin();

  initMPU();
  calibrateMPU();

  pinMode(FSR1_PIN, INPUT);
  pinMode(FSR2_PIN, INPUT);

  Serial.println(F("FSR‑Resistance + IMU monitor ready"));
}

void loop() {
  /* Read sensors */
  int fsr1Raw = analogRead(FSR1_PIN);
  int fsr2Raw = analogRead(FSR2_PIN);
  float R1 = getResistance(fsr1Raw);
  float R2 = getResistance(fsr2Raw);
  readMPU();

  /* Print */
  Serial.print(F("FSR1: "));
  (R1<0) ? Serial.print(F("Open")) : Serial.print(R1,1);
  Serial.print(F(" Ω   FSR2: "));
  (R2<0) ? Serial.print(F("Open")) : Serial.print(R2,1);
  Serial.println(F(" Ω"));

  Serial.print(F("Accel [m/s²] X=")); Serial.print(ax_ms2,2);
  Serial.print(F(" Y=")); Serial.print(ay_ms2,2);
  Serial.print(F(" Z=")); Serial.print(az_ms2,2);

  Serial.print(F(" | Gyro [rad/s] X=")); Serial.print(gx_rad,3);
  Serial.print(F(" Y=")); Serial.print(gy_rad,3);
  Serial.print(F(" Z=")); Serial.println(gz_rad,3);

  Serial.println(F("-------------------------------------------"));
  delay(250); // 4 Hz print‑rate
}
