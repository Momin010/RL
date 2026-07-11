// High-level evasion controller: raw sensor stream in, fin commands out.
//
// This wraps the raw network (nn_inference.h) with everything the flight loop
// actually needs:
//   * assembling the 10-element feature vector in the exact order the network
//     was trained on (rocketnn/simulator.py::FEATURE_NAMES),
//   * finite-differencing the line-of-sight rates from the incoming angle
//     stream (the training sensor model does the identical thing),
//   * carrying the previous fin commands back in as proprioceptive inputs,
//   * decoding the classifier head (what is inbound + a confidence),
//   * an optional safety gate so the vehicle only throws a maneuver for a
//     genuine, closing threat.
//
// Pure C, no Arduino dependencies, no dynamic memory — so it drops into any
// loop and also compiles on a desktop for testing.
#ifndef ROCKET_EVASION_CONTROLLER_H
#define ROCKET_EVASION_CONTROLLER_H

#include <math.h>
#include "nn_inference.h"

typedef struct {
    float prev_bearing;
    float prev_elevation;
    float last_fin_pitch;
    float last_fin_yaw;
    int   initialized;
} EvasionState;

typedef struct {
    float fin_pitch;     // commanded pitch-plane fin, [-1, 1]
    float fin_yaw;       // commanded yaw-plane fin,   [-1, 1]
    int   threat_class;  // 0=none 1=ballistic 2=guided 3=debris
    float threat_conf;   // classifier confidence for that class, [0, 1]
    int   evading;       // 1 if a meaningful evasion is being commanded
} EvasionOutput;

static const char* const RNN_CLASS_NAMES[RNN_NUM_CLASSES] = {
    "none", "ballistic", "guided", "debris"
};

static inline void evasion_init(EvasionState* s) {
    s->prev_bearing = 0.0f;
    s->prev_elevation = 0.0f;
    s->last_fin_pitch = 0.0f;
    s->last_fin_yaw = 0.0f;
    s->initialized = 0;
}

static inline float rnn_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// One control step.
//   range_m       : slant range to the tracked object (m)
//   closing_mps   : range rate, positive when approaching (m/s)
//   bearing_rad   : horizontal angle off the nose (rad)
//   elevation_rad : vertical angle off the nose (rad)
//   speed_mps     : own forward speed (m/s)
//   dt_s          : loop period since the previous call (s)
//   gate          : if non-zero, suppress fins unless a real threat is closing
static inline EvasionOutput evasion_step(EvasionState* s,
                                         float range_m, float closing_mps,
                                         float bearing_rad, float elevation_rad,
                                         float speed_mps, float dt_s,
                                         int gate) {
    float brate = 0.0f, erate = 0.0f;
    if (s->initialized && dt_s > 1e-6f) {
        brate = (bearing_rad - s->prev_bearing) / dt_s;
        erate = (elevation_rad - s->prev_elevation) / dt_s;
    }
    float denom = closing_mps > 1.0f ? closing_mps : 1.0f;
    float tgo = range_m / denom;

    float feat[RNN_NUM_FEATURES];
    feat[0] = range_m;
    feat[1] = closing_mps;
    feat[2] = bearing_rad;
    feat[3] = elevation_rad;
    feat[4] = brate;
    feat[5] = erate;
    feat[6] = tgo;
    feat[7] = speed_mps;
    feat[8] = s->last_fin_pitch;
    feat[9] = s->last_fin_yaw;

    float out[RNN_NUM_OUTPUTS];
    rnn_infer(feat, out);

    EvasionOutput r;
    r.fin_pitch = rnn_clampf(out[RNN_OUT_FIN_PITCH], -1.0f, 1.0f);
    r.fin_yaw   = rnn_clampf(out[RNN_OUT_FIN_YAW], -1.0f, 1.0f);

    // Decode the classifier head (argmax + softmax confidence).
    int best = 0;
    float bestv = out[RNN_OUT_CLASS0];
    for (int c = 1; c < RNN_NUM_CLASSES; ++c) {
        if (out[RNN_OUT_CLASS0 + c] > bestv) {
            bestv = out[RNN_OUT_CLASS0 + c];
            best = c;
        }
    }
    float sum = 0.0f;
    for (int c = 0; c < RNN_NUM_CLASSES; ++c) {
        sum += expf(out[RNN_OUT_CLASS0 + c] - bestv);
    }
    r.threat_class = best;
    r.threat_conf = (sum > 0.0f) ? (1.0f / sum) : 1.0f;

    // Optional safety gate: don't waste control authority (or risk a tumble)
    // on something that is not a closing threat. class 0 == "none".
    if (gate && (best == 0 || closing_mps <= 0.0f)) {
        r.fin_pitch = 0.0f;
        r.fin_yaw = 0.0f;
    }

    float mag = sqrtf(r.fin_pitch * r.fin_pitch + r.fin_yaw * r.fin_yaw);
    r.evading = (mag > 0.05f) ? 1 : 0;

    // Persist state for next step's rates / proprioception.
    s->prev_bearing = bearing_rad;
    s->prev_elevation = elevation_rad;
    s->last_fin_pitch = r.fin_pitch;
    s->last_fin_yaw = r.fin_yaw;
    s->initialized = 1;
    return r;
}

#endif  // ROCKET_EVASION_CONTROLLER_H
