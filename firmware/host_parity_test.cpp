// Host-side parity test: proves the C++ inference engine reproduces the
// trained Python network to within a tight tolerance, before any of this runs
// on real hardware. Build and run with tools/run_all.sh (or the g++ line at
// the bottom of this file). Exits non-zero if parity fails.
//
// It also instantiates the high-level evasion controller once, so that
// evasion_controller.h gets full compile coverage on the host.

#include <cstdio>
#include <cmath>

#include "nn_inference.h"
#include "evasion_controller.h"
#include "parity_vectors.h"

int main() {
    float max_abs = 0.0f;
    int worst_row = -1, worst_col = -1;

    for (int i = 0; i < RNN_PARITY_N; ++i) {
        float out[RNN_NUM_OUTPUTS];
        rnn_infer(rnn_parity_in[i], out);
        for (int j = 0; j < RNN_NUM_OUTPUTS; ++j) {
            float d = std::fabs(out[j] - rnn_parity_out[i][j]);
            if (d > max_abs) {
                max_abs = d;
                worst_row = i;
                worst_col = j;
            }
        }
    }

    printf("parity vectors : %d\n", RNN_PARITY_N);
    printf("network        : %d features -> %d outputs, %d layers\n",
           RNN_NUM_FEATURES, RNN_NUM_OUTPUTS, RNN_NUM_LAYERS);
    printf("max abs diff   : %.3e  (row %d, col %d)\n",
           max_abs, worst_row, worst_col);

    // Exercise the full controller path once (compile + sanity check).
    EvasionState st;
    evasion_init(&st);
    // A close, fast, near-boresight closing threat: expect a real maneuver.
    EvasionOutput r = evasion_step(&st, /*range*/120.0f, /*closing*/380.0f,
                                   /*bearing*/0.04f, /*elevation*/-0.02f,
                                   /*speed*/210.0f, /*dt*/0.01f, /*gate*/1);
    // Second call so the rate finite-difference is live.
    r = evasion_step(&st, 118.0f, 380.0f, 0.055f, -0.028f, 210.0f, 0.01f, 1);
    printf("controller demo: class=%s conf=%.2f fin_pitch=%+.3f fin_yaw=%+.3f evading=%d\n",
           RNN_CLASS_NAMES[r.threat_class], r.threat_conf,
           r.fin_pitch, r.fin_yaw, r.evading);

    const float TOL = 1e-3f;
    if (max_abs > TOL) {
        printf("RESULT         : FAIL (max diff %.3e > tol %.3e)\n", max_abs, TOL);
        return 1;
    }
    printf("RESULT         : PASS (C++ matches Python within %.1e)\n", TOL);
    return 0;
}

// Manual build:
//   g++ -O2 -std=c++14 -I firmware firmware/host_parity_test.cpp -o /tmp/parity && /tmp/parity
