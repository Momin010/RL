// Host-side parity test for the stabilization network.
//
// Proves the dependency-free C++ engine (snn_infer) reproduces the trained
// Python network on real in-distribution feature vectors. Run on your desktop
// before flashing:
//
//   g++ -O2 -std=c++14 -I firmware firmware/host_stab_parity_test.cpp -o parity_stab
//   ./parity_stab
#include <cmath>
#include <cstdio>

#include "stabilization_controller.h"
#include "stab_parity_vectors.h"

int main() {
    float out[SNN_NUM_OUTPUTS];
    float worst = 0.0f;
    int fails = 0;
    for (int i = 0; i < SNN_PARITY_N; ++i) {
        snn_infer(snn_parity_in[i], out);
        for (int o = 0; o < SNN_NUM_OUTPUTS; ++o) {
            float err = std::fabs(out[o] - snn_parity_out[i][o]);
            if (err > worst) worst = err;
            if (err > 1e-3f) {
                ++fails;
                std::printf("MISMATCH vec %d out %d: C++ %.8f vs Python %.8f\n",
                            i, o, out[o], snn_parity_out[i][o]);
            }
        }
    }
    std::printf("stab parity: %d vectors, worst |err| = %.3e -> %s\n",
                SNN_PARITY_N, worst, fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
