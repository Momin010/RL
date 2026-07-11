// ============================================================================
//  Rocket evasion controller — Teensy 4.1 flight sketch
// ============================================================================
//
//  Pipeline per loop:
//     read threat sensor  ->  neural net  ->  fin deflection commands
//                                          ->  threat classification
//     blend with your existing stabilization loop  ->  servos
//
//  The neural network was trained offline (see the repo README) and baked into
//  model_weights.h. Inference here is a fixed ~sub-microsecond forward pass on
//  the M7's FPU — the DWT cycle counter below prints the real number so you can
//  see the latency budget for yourself.
//
//  THIS SKETCH IS A HARNESS. Two things are yours to wire up, both clearly
//  marked "INTEGRATION POINT" below:
//     1. read_threat_sensor()  — feed in your radar / ToF / optical tracker
//     2. existing_control_loop() — your current stabilization + servo commands
//
//  Everything between them (feature assembly, inference, classification, the
//  safety gate) is done for you by evasion_controller.h.
// ============================================================================

#include <Servo.h>
#include "evasion_controller.h"

// ---- Configuration ---------------------------------------------------------
static const int   PIN_FIN_PITCH = 2;      // servo controlling the pitch plane
static const int   PIN_FIN_YAW   = 3;      // servo controlling the yaw plane
static const int   PIN_ARM       = 4;      // arming switch (HIGH = armed)

static const float LOOP_HZ       = 500.0f; // control loop rate
static const int   SERVO_CENTER_US = 1500;
static const int   SERVO_TRAVEL_US = 450;  // +/- microseconds at full deflection

// How strongly evasion overrides your stabilization loop when a threat is live.
// 0 = evasion ignored, 1 = evasion fully replaces stabilization while evading.
static const float EVASION_AUTHORITY = 1.0f;

// Only actually classify/evade objects inside this range (metres). Outside it
// we still run the net for telemetry but hold fire.
static const float ENGAGE_RANGE_M = 500.0f;

// ---- Globals ---------------------------------------------------------------
Servo finPitch, finYaw;
EvasionState evasion;
const float DT = 1.0f / LOOP_HZ;
elapsedMicros loopTimer;

// A raw reading from your threat tracker. Fill these in read_threat_sensor().
struct ThreatReading {
    bool  valid;          // is the tracker currently locked onto something?
    float range_m;        // slant range
    float closing_mps;    // range rate, positive = approaching
    float bearing_rad;    // horizontal angle off the nose
    float elevation_rad;  // vertical angle off the nose
    float own_speed_mps;  // your forward speed (from IMU/pitot/baro)
};

// ---------------------------------------------------------------------------
//  Cycle-accurate timing of the inference call (Cortex-M7 DWT counter).
// ---------------------------------------------------------------------------
static inline void enable_cycle_counter() {
#if defined(ARM_DWT_CYCCNT)
    ARM_DEMCR |= ARM_DEMCR_TRCENA;
    ARM_DWT_CTRL |= ARM_DWT_CTRL_CYCCNTENA;
#endif
}
static inline uint32_t cycles_now() {
#if defined(ARM_DWT_CYCCNT)
    return ARM_DWT_CYCCNT;
#else
    return 0;
#endif
}

// ===========================================================================
//  INTEGRATION POINT 1 — your threat sensor
//  Replace the body with a real read from your radar / lidar / ToF / optical
//  tracker. Return valid=false when nothing is being tracked. The controller
//  computes line-of-sight rates for you, so raw range + angles are enough.
// ===========================================================================
ThreatReading read_threat_sensor() {
    ThreatReading t;
    t.valid = false;
    t.range_m = 0.0f;
    t.closing_mps = 0.0f;
    t.bearing_rad = 0.0f;
    t.elevation_rad = 0.0f;
    t.own_speed_mps = 0.0f;
    // e.g. t = poll_radar();  t.own_speed_mps = imu_forward_speed();
    return t;
}

// ===========================================================================
//  INTEGRATION POINT 2 — your existing stabilization / servo loop
//  Return whatever fin deflections your current controller wants, each in
//  [-1, 1]. These are what fly when there is no threat; the evasion command is
//  blended on top when a threat appears.
// ===========================================================================
struct FinCommand { float pitch; float yaw; };

FinCommand existing_control_loop() {
    FinCommand c;
    c.pitch = 0.0f;   // <-- your PID / attitude controller output here
    c.yaw   = 0.0f;
    return c;
}

// ---------------------------------------------------------------------------
static inline int fin_to_us(float cmd) {
    cmd = rnn_clampf(cmd, -1.0f, 1.0f);
    return SERVO_CENTER_US + (int)(cmd * SERVO_TRAVEL_US);
}

void setup() {
    Serial.begin(115200);
    finPitch.attach(PIN_FIN_PITCH);
    finYaw.attach(PIN_FIN_YAW);
    pinMode(PIN_ARM, INPUT_PULLDOWN);
    finPitch.writeMicroseconds(SERVO_CENTER_US);
    finYaw.writeMicroseconds(SERVO_CENTER_US);
    evasion_init(&evasion);
    enable_cycle_counter();
}

void loop() {
    if (loopTimer < (unsigned)(1e6f * DT)) {
        return;                          // hold the loop rate
    }
    loopTimer = 0;

    const bool armed = digitalRead(PIN_ARM) == HIGH;

    // 1) Your stabilization loop always runs.
    FinCommand base = existing_control_loop();

    // 2) Read the threat and run the network.
    ThreatReading t = read_threat_sensor();

    float fin_pitch = base.pitch;
    float fin_yaw   = base.yaw;
    int   threat_class = 0;
    float threat_conf = 0.0f;
    uint32_t infer_cycles = 0;

    if (t.valid && t.range_m <= ENGAGE_RANGE_M) {
        uint32_t c0 = cycles_now();
        EvasionOutput e = evasion_step(&evasion,
                                       t.range_m, t.closing_mps,
                                       t.bearing_rad, t.elevation_rad,
                                       t.own_speed_mps, DT, /*gate*/1);
        infer_cycles = cycles_now() - c0;
        threat_class = e.threat_class;
        threat_conf = e.threat_conf;

        // 3) Blend evasion over stabilization. While evading, evasion takes
        //    priority up to EVASION_AUTHORITY; otherwise stabilization holds.
        if (e.evading) {
            float w = EVASION_AUTHORITY;
            fin_pitch = (1.0f - w) * base.pitch + w * e.fin_pitch;
            fin_yaw   = (1.0f - w) * base.yaw   + w * e.fin_yaw;
        }
    } else {
        // No lock: keep the controller's rate estimator from spiking on
        // reacquisition by resetting its finite-difference memory.
        evasion_init(&evasion);
    }

    // 4) Drive the servos (only when armed; otherwise hold centre).
    if (armed) {
        finPitch.writeMicroseconds(fin_to_us(fin_pitch));
        finYaw.writeMicroseconds(fin_to_us(fin_yaw));
    } else {
        finPitch.writeMicroseconds(SERVO_CENTER_US);
        finYaw.writeMicroseconds(SERVO_CENTER_US);
    }

    // 5) Telemetry at a reduced rate (every 50 loops).
    static uint32_t n = 0;
    if ((n++ % 50) == 0) {
        float us = (float)infer_cycles * 1e6f / (float)F_CPU_ACTUAL;
        Serial.printf("armed=%d lock=%d class=%s conf=%.2f "
                      "fin=(%+.2f,%+.2f) infer=%.2fus (%lu cyc)\n",
                      armed, t.valid,
                      (t.valid ? RNN_CLASS_NAMES[threat_class] : "----"),
                      threat_conf, fin_pitch, fin_yaw, us,
                      (unsigned long)infer_cycles);
    }
}
