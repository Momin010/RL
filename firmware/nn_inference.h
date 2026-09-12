// Dependency-free neural-network inference for the Teensy 4.1.
//
// This is a straight, allocation-free forward pass over the weights in
// model_weights.h. It uses only float32 (the Cortex-M7 has a hardware FPU),
// needs no libc math functions in the hot path, and touches no dynamic memory,
// so its timing is deterministic - exactly what a hard real-time control loop
// wants. The arithmetic mirrors rocketnn/nn.py bit-for-bit closely enough that
// the host parity test agrees to < 1e-3 (see host_parity_test.cpp).
//
// Portable C/C++: this header also compiles on a desktop for testing.
#ifndef ROCKET_NN_INFERENCE_H
#define ROCKET_NN_INFERENCE_H

#include "model_weights.h"

// Forward pass.
//   raw_in : RNN_NUM_FEATURES un-normalised sensor features
//   out    : RNN_NUM_OUTPUTS network outputs (fin cmds + class logits)
// The input is standardised internally using the baked-in mean/std, so the
// caller passes raw engineering units (metres, m/s, radians).
static inline void rnn_infer(const float* raw_in, float* out) {
    float a[RNN_MAX_UNITS];   // current activations
    float z[RNN_MAX_UNITS];   // next-layer pre-activations

    for (int i = 0; i < RNN_NUM_FEATURES; ++i) {
        a[i] = (raw_in[i] - rnn_in_mean[i]) / rnn_in_std[i];
    }

    for (int L = 0; L < RNN_NUM_LAYERS; ++L) {
        const int nin = rnn_layer_in[L];
        const int nout = rnn_layer_out[L];
        const float* W = rnn_W[L];    // row-major [nout][nin]
        const float* b = rnn_b[L];
        for (int o = 0; o < nout; ++o) {
            const float* wrow = W + o * nin;
            float acc = b[o];
            for (int k = 0; k < nin; ++k) {
                acc += wrow[k] * a[k];
            }
            if (rnn_layer_relu[L] && acc < 0.0f) {
                acc = 0.0f;
            }
            z[o] = acc;
        }
        for (int o = 0; o < nout; ++o) {
            a[o] = z[o];
        }
    }

    for (int o = 0; o < RNN_NUM_OUTPUTS; ++o) {
        out[o] = a[o];
    }
}

#endif  // ROCKET_NN_INFERENCE_H
