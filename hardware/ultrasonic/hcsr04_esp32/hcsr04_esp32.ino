// HC-SR04 distance sensor on an ESP32 -> one line per reading over USB serial.
//
// Wiring (HC-SR04 needs 5 V; its ECHO pin outputs 5 V, the ESP32 is 3.3 V only):
//   HC-SR04 VCC  -> ESP32 5V / VIN
//   HC-SR04 GND  -> ESP32 GND
//   HC-SR04 TRIG -> GPIO 5                    (3.3 V is enough to trigger it)
//   HC-SR04 ECHO -> 1k -> GPIO 18 -> 2k -> GND (voltage divider, 5 V -> 3.3 V)
// (An HC-SR04P / 3.3 V variant can be powered from 3V3 and wired without the divider.)
//
// Output, 115200 baud, ~20 lines/s:
//   dist_mm,<median of 5 pings in mm, or -1 if no echo>
// Read it with: python hardware/ultrasonic/read_ultrasonic.py COMx

const int TRIG_PIN = 5;
const int ECHO_PIN = 18;
const unsigned long ECHO_TIMEOUT_US = 25000;  // ~4.3 m round trip; longer = nothing in range
const int SAMPLES = 5;

float pingMm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(4);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  unsigned long us = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);
  if (us == 0) return -1.0f;
  return us * 0.1715f;  // speed of sound 343 m/s, halved for the round trip -> mm per us
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);
  delay(100);
  Serial.println("# hcsr04_esp32 ready");
}

void loop() {
  float v[SAMPLES];
  int n = 0;
  for (int i = 0; i < SAMPLES; i++) {
    float d = pingMm();
    if (d > 0) v[n++] = d;
    delay(10);  // let echoes from the previous ping die out
  }
  if (n == 0) {
    Serial.println("dist_mm,-1");
    return;
  }
  // insertion sort, then take the median of the valid pings
  for (int i = 1; i < n; i++) {
    float key = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; j--; }
    v[j + 1] = key;
  }
  Serial.print("dist_mm,");
  Serial.println(v[n / 2], 1);
}
