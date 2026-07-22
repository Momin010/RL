// Fin-stabilization controller for the Teensy 4.1.
//
// Wraps the trained stabilization network (stab_model_weights.h) behind one
// call per control tick. Feature assembly here MUST mirror
// rocketnn/stab_sim.py::make_features — the parity test checks the raw
// network; this file is the contract for what you feed it.
//
//   inputs : tilt + tilt rate per axis (from your IMU / attitude filter),
//            airspeed estimate, and time since ignition
//   output : fin_pitch / fin_yaw commands in [-1, 1]
//
// Portable C/C++ — also compiles on a desktop for the parity test.
#ifndef ROCKET_STABILIZATION_CONTROLLER_H
#define ROCKET_STABILIZATION_CONTROLLER_H

#include "stab_model_weights.h"

#ifndef RNN_CLAMPF_DEFINED
#define RNN_CLAMPF_DEFINED
static inline float rnn_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
#endif

// Forward pass over the snn_* weights (same engine as nn_inference.h, but
// bound to the stabilization net so both networks can live in one image).
static inline void snn_infer(const float* raw_in, float* out) {
    float a[SNN_MAX_UNITS];
    float z[SNN_MAX_UNITS];
    for (int i = 0; i < SNN_NUM_FEATURES; ++i) {
        a[i] = (raw_in[i] - snn_in_mean[i]) / snn_in_std[i];
    }
    for (int L = 0; L < SNN_NUM_LAYERS; ++L) {
        const int nin = snn_layer_in[L];
        const int nout = snn_layer_out[L];
        const float* W = snn_W[L];
        const float* b = snn_b[L];
        for (int o = 0; o < nout; ++o) {
            const float* wrow = W + o * nin;
            float acc = b[o];
            for (int k = 0; k < nin; ++k) acc += wrow[k] * a[k];
            if (snn_layer_relu[L] && acc < 0.0f) acc = 0.0f;
            z[o] = acc;
        }
        for (int o = 0; o < nout; ++o) a[o] = z[o];
    }
    for (int o = 0; o < SNN_NUM_OUTPUTS; ++o) out[o] = a[o];
}

typedef struct {
    float fin_pitch_prev;   // last commanded, fed back as a feature
    float fin_yaw_prev;
    float burn_time_s;      // burn time of the loaded motor (set at init)
} StabState;

typedef struct {
    float fin_pitch;        // [-1, 1]
    float fin_yaw;          // [-1, 1]
} StabOutput;

static inline void stab_init(StabState* s, float burn_time_s) {
    s->fin_pitch_prev = 0.0f;
    s->fin_yaw_prev = 0.0f;
    s->burn_time_s = burn_time_s;
}

// One control tick (call at ~100 Hz once off the rail).
//   tilt_*  : rad, tilt of the body axis from vertical in each control plane
//   rate_*  : rad/s, body rates from the gyro
//   airspeed_mps : airspeed (or vertical-speed) estimate
//   t_since_ignition_s : seconds since motor ignition
static inline StabOutput stab_step(StabState* s,
                                   float tilt_pitch, float rate_pitch,
                                   float tilt_yaw, float rate_yaw,
                                   float airspeed_mps,
                                   float t_since_ignition_s) {
    const float q_dyn = 0.5f * 1.225f * airspeed_mps * airspeed_mps;
    float burn_frac = 1.0f;
    if (s->burn_time_s > 0.0f && t_since_ignition_s < s->burn_time_s) {
        burn_frac = t_since_ignition_s / s->burn_time_s;
    }

    // Mirror of rocketnn/stab_sim.py::make_features — keep in lockstep.
    float x[SNN_NUM_FEATURES];
    x[0] = tilt_pitch;
    x[1] = rate_pitch;
    x[2] = tilt_yaw;
    x[3] = rate_yaw;
    x[4] = airspeed_mps / 50.0f;
    x[5] = q_dyn / 1500.0f;
    x[6] = s->fin_pitch_prev;
    x[7] = s->fin_yaw_prev;
    x[8] = burn_frac;

    float y[SNN_NUM_OUTPUTS];
    snn_infer(x, y);

    StabOutput o;
    o.fin_pitch = rnn_clampf(y[0], -1.0f, 1.0f);
    o.fin_yaw = rnn_clampf(y[1], -1.0f, 1.0f);
    s->fin_pitch_prev = o.fin_pitch;
    s->fin_yaw_prev = o.fin_yaw;
    return o;
}

#endif  // ROCKET_STABILIZATION_CONTROLLER_H
