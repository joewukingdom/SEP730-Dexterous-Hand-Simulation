#include <Arduino.h>

// HC-SR04 style ultrasonic sensor
constexpr int TRIG_PIN = 10;
constexpr int ECHO_PIN = 18;

// ~4 m max range -> round trip ~23.3 ms; 30 ms timeout
constexpr unsigned long ECHO_TIMEOUT_US = 30000;

// Returns distance in cm, or -1 if no echo received.
float readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);
  if (duration == 0) return -1.0f;

  // speed of sound ~343 m/s = 0.0343 cm/us, divide by 2 for round trip
  return duration * 0.0343f / 2.0f;
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);
  Serial.println("Ultrasonic sensor ready (TRIG=10, ECHO=18)");
}

void loop() {
  float d = readDistanceCm();
  if (d < 0) {
    Serial.println("Out of range / no echo");
  } else {
    Serial.printf("Distance: %.1f cm\n", d);
  }
  delay(100);  // >=60 ms between pings to avoid echo overlap
}
