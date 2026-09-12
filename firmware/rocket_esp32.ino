// ============================================================================
//  Rocket stabilization + evasion controller - ESP32-S flight sketch
//  (ESP32 / ESP32-S2 / ESP32-S3, Arduino core 3.x)
// ============================================================================
//
//  Same brain as the Teensy sketch (rocket_evasion.ino) - identical networks,
//  identical controllers - but with the platform bits swapped for ESP32:
//    * servo PWM via the LEDC peripheral (no Servo library needed)
//    * timing via micros()
//    * cycle counting via the Xtensa/RISC-V cycle counter when available
//
//  Chip notes:
//    * ESP32 / ESP32-S3: single-precision hardware FPU - inference runs in a
//      few tens of microseconds. Plenty for a 100 Hz (or 1 kHz) loop.
//    * ESP32-S2: NO hardware FPU - float math is software-emulated. Inference
//      still fits a 100 Hz loop with huge margin (~hundreds of us), but prefer
//      an S3 if you're buying hardware.
//
//  INTEGRATION POINTS (same contract as the Teensy sketch):
//    1. read_attitude()      - your IMU / attitude filter output
//    2. read_threat_sensor() - your object tracker (optional; leave invalid
//                              and the vehicle just flies stabilization)
// ============================================================================

#include "evasion_controller.h"
#include "stabilization_controller.h"

// ---- Pins (any LEDC-capable GPIO; avoid strapping pins 0/45/46 on S3) ------
static const int PIN_FIN_PITCH = 4;
static const int PIN_FIN_YAW   = 5;
static const int PIN_ARM       = 6;    // arming switch (HIGH = armed)

// ---- Servo PWM via LEDC ------------------------------------------------------
// Standard hobby-servo signal: 50 Hz frame, 1000-2000 us pulse.
static const int      SERVO_FREQ_HZ   = 50;
static const int      SERVO_RES_BITS  = 16;
static const int      SERVO_CENTER_US = 1500;
static const int      SERVO_TRAVEL_US = 450;   // +/- us at full deflection
static const uint32_t SERVO_MAX_DUTY  = (1UL << SERVO_RES_BITS) - 1;

static inline void servo_write_us(int pin, int us) {
    // duty = us / frame_period(20000 us) scaled to the timer resolution
    uint32_t duty = (uint32_t)((uint64_t)us * SERVO_FREQ_HZ * SERVO_MAX_DUTY / 1000000ULL);
    ledcWrite(pin, duty);
}

// ---- Control rates -----------------------------------------------------------
static const float LOOP_HZ = 250.0f;   // outer loop (evasion + servo update)
static const float STAB_HZ = 100.0f;   // stabilization net rate (training rate)

// Evasion blend + engage gate, same semantics as the Teensy sketch.
static const float EVASION_AUTHORITY = 1.0f;
static const float ENGAGE_RANGE_M    = 500.0f;

// Burn time (s) of the motor you fly (see rocketnn/motors.py for the six
// trained motors, e.g. Estes F15: 3.45, AeroTech F24W: 2.13, Cesaroni F70: 0.816).
static const float MOTOR_BURN_TIME_S = 3.45f;

// ---- Globals -----------------------------------------------------------------
EvasionState evasion;
StabState    stab;
static uint32_t last_loop_us = 0;
static uint32_t last_stab_us = 0;
static float    t_ignition_s = -1.0f;   // set at launch detect
static float    stab_pitch = 0.0f, stab_yaw = 0.0f;

// Cycle counter for latency telemetry (0 on cores without one exposed).
static inline uint32_t cycles_now() {
#if defined(__XTENSA__)
    uint32_t c; __asm__ __volatile__("rsr %0, ccount" : "=a"(c)); return c;
#else
    return (uint32_t)micros();  // fallback: microseconds instead of cycles
#endif
}

// ===========================================================================
//  INTEGRATION POINT 1 - attitude sensing for the stabilization network
// ===========================================================================
struct AttitudeReading {
    bool  valid;
    float tilt_pitch_rad;   // tilt from vertical, pitch-servo plane
    float rate_pitch_rps;
    float tilt_yaw_rad;     // tilt from vertical, yaw-servo plane
    float rate_yaw_rps;
    float airspeed_mps;
};

AttitudeReading read_attitude() {
    AttitudeReading a;
    a.valid = false;        // <-- set true once your Madgwick/Kalman filter runs
    a.tilt_pitch_rad = 0.0f;
    a.rate_pitch_rps = 0.0f;
    a.tilt_yaw_rad = 0.0f;
    a.rate_yaw_rps = 0.0f;
    a.airspeed_mps = 0.0f;
    // e.g. a = imu_attitude();  a.airspeed_mps = baro_vertical_speed();
    return a;
}

// ===========================================================================
//  INTEGRATION POINT 2 - your threat sensor (optional)
// ===========================================================================
struct ThreatReading {
    bool  valid;
    float range_m;
    float closing_mps;
    float bearing_rad;
    float elevation_rad;
    float own_speed_mps;
};

ThreatReading read_threat_sensor() {
    ThreatReading t;
    t.valid = false;        // leave false to fly pure stabilization
    t.range_m = 0.0f;
    t.closing_mps = 0.0f;
    t.bearing_rad = 0.0f;
    t.elevation_rad = 0.0f;
    t.own_speed_mps = 0.0f;
    return t;
}

// ---------------------------------------------------------------------------
static inline int fin_to_us(float cmd) {
    cmd = rnn_clampf(cmd, -1.0f, 1.0f);
    return SERVO_CENTER_US + (int)(cmd * SERVO_TRAVEL_US);
}

void setup() {
    Serial.begin(115200);
    pinMode(PIN_ARM, INPUT_PULLDOWN);
    ledcAttach(PIN_FIN_PITCH, SERVO_FREQ_HZ, SERVO_RES_BITS);
    ledcAttach(PIN_FIN_YAW, SERVO_FREQ_HZ, SERVO_RES_BITS);
    servo_write_us(PIN_FIN_PITCH, SERVO_CENTER_US);
    servo_write_us(PIN_FIN_YAW, SERVO_CENTER_US);
    evasion_init(&evasion);
    stab_init(&stab, MOTOR_BURN_TIME_S);
    last_loop_us = micros();
    last_stab_us = micros();
}

void loop() {
    const uint32_t now = micros();
    if ((now - last_loop_us) < (uint32_t)(1e6f / LOOP_HZ)) return;
    const float dt = (now - last_loop_us) * 1e-6f;
    last_loop_us = now;

    const bool armed = digitalRead(PIN_ARM) == HIGH;

    // 1) Stabilization net at its trained 100 Hz rate; hold between ticks.
    AttitudeReading a = read_attitude();
    if (a.valid && (now - last_stab_us) >= (uint32_t)(1e6f / STAB_HZ)) {
        last_stab_us = now;
        // Launch detect: first time we see real airspeed, start the clock.
        if (t_ignition_s < 0.0f && a.airspeed_mps > 5.0f) {
            t_ignition_s = now * 1e-6f;
        }
        const float t_since = (t_ignition_s >= 0.0f)
                                  ? (now * 1e-6f - t_ignition_s)
                                  : MOTOR_BURN_TIME_S;   // burn_frac=1 on pad
        StabOutput s = stab_step(&stab,
                                 a.tilt_pitch_rad, a.rate_pitch_rps,
                                 a.tilt_yaw_rad, a.rate_yaw_rps,
                                 a.airspeed_mps, t_since);
        stab_pitch = s.fin_pitch;
        stab_yaw = s.fin_yaw;
    }
    float fin_pitch = stab_pitch;
    float fin_yaw = stab_yaw;

    // 2) Evasion net, blended on top when something real is inbound.
    ThreatReading t = read_threat_sensor();
    int threat_class = 0;
    float threat_conf = 0.0f;
    uint32_t infer_cycles = 0;
    if (t.valid && t.range_m <= ENGAGE_RANGE_M) {
        const uint32_t c0 = cycles_now();
        EvasionOutput e = evasion_step(&evasion,
                                       t.range_m, t.closing_mps,
                                       t.bearing_rad, t.elevation_rad,
                                       t.own_speed_mps, dt, /*gate=*/1);
        infer_cycles = cycles_now() - c0;
        threat_class = e.threat_class;
        threat_conf = e.threat_conf;
        if (e.evading) {
            fin_pitch = (1.0f - EVASION_AUTHORITY) * fin_pitch
                        + EVASION_AUTHORITY * e.fin_pitch;
            fin_yaw = (1.0f - EVASION_AUTHORITY) * fin_yaw
                      + EVASION_AUTHORITY * e.fin_yaw;
        }
    }

    // 3) Servos (center when disarmed).
    if (!armed) { fin_pitch = 0.0f; fin_yaw = 0.0f; }
    servo_write_us(PIN_FIN_PITCH, fin_to_us(fin_pitch));
    servo_write_us(PIN_FIN_YAW, fin_to_us(fin_yaw));

    // 4) 1 Hz telemetry.
    static uint32_t last_print_us = 0;
    if (now - last_print_us > 1000000UL) {
        last_print_us = now;
        Serial.printf("armed=%d fin=[%.2f %.2f] class=%s conf=%.2f infer_cyc=%lu\n",
                      (int)armed, fin_pitch, fin_yaw,
                      RNN_CLASS_NAMES[threat_class], threat_conf,
                      (unsigned long)infer_cycles);
    }
}
