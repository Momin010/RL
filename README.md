# Rocket Evasion Neural Network — Teensy 4.1

An end-to-end TinyML pipeline for an on-board **evasion controller**: sense an
incoming object, recognise *what it is*, and command fin deflections to
**maximise miss distance** — fast enough to run inside a hard real-time flight
loop on a Teensy 4.1.

It ships as three cooperating parts:

1. a **trainer** (pure NumPy, no PyTorch/TensorFlow) that learns the policy from
   a physics simulator,
2. a **weight exporter** that bakes the trained network into a C header, and
3. **firmware** — a dependency-free C++ inference engine plus a Teensy sketch
   that drops into your existing servo loop.

---

## The one thing to understand first

**You do not train a neural network *on* a Teensy — and you don't want to.**
Training needs a dataset, gradients, and 64-bit optimizer state; a reflex that
must fire in milliseconds can't stop to do that. The correct pattern (what every
serious embedded-ML system does) is:

```
  HOST (this repo, Python)                    DEVICE (Teensy 4.1, C++)
  ────────────────────────                    ────────────────────────
  simulate engagements                         read threat sensor
  train the network      ──weights (3 KB)──▶   run the SAME network (µs)
  verify + export                              command fins, blend w/ your loop
```

The payoff for your "super fast / 100 ms" requirement: the trained network is
**766 parameters (~3 KB)** and one forward pass is a few microseconds on the
M7's FPU — so inference uses well under **1%** of even a 1 kHz control loop.
Compute is nowhere near your bottleneck; your sensor's update rate is.

---

## Results (measured, `artifacts/metrics.json`)

Closed-loop **median miss distance** over 120 held-out engagements per class —
flying straight (no evasion) vs. the trained network vs. the expert it learned
from. Bigger is better; a "hit" is a few metres.

| Inbound threat        | No evasion | **Trained net** | Expert (teacher) |
|-----------------------|-----------:|----------------:|-----------------:|
| **Guided** (tracks you) | **1.2 m** (dead hit) | **29.6 m** | 39.2 m |
| **Ballistic** (fast, unguided) | **7.1 m** (near hit) | **37.3 m** | 55.9 m |
| Debris (tumbling)     | 98.5 m | 133.8 m | 186.4 m |
| None (passing wide)   | 106.3 m | 146.4 m | 194.9 m |

The network converts a **1.2 m guaranteed intercept into a ~30 m miss**, from
*noisy, single-frame* sensor data, capturing ~70–75% of the expert's benefit.

**Threat classification:** 89.3% overall 4-way accuracy — and for the two
classes that can actually kill you it is far higher: **ballistic 95.4%**,
**guided 97.5%**. Most residual error is `none`↔`debris`, both non-collision.

**C++/Python parity:** the on-device engine reproduces the trained network to
**< 1e-5** (float32 rounding). Proven automatically by the host parity test.

---

## Quickstart

```bash
# Train, export weights to C, then build+run the parity test — all in one:
bash tools/run_all.sh
```

Individually:

```bash
python3 scripts/train.py       # -> artifacts/model.npz + metrics.json  (~2 min)
python3 scripts/export_c.py    # -> firmware/model_weights.h + parity_vectors.h
g++ -O2 -std=c++14 -I firmware firmware/host_parity_test.cpp -o /tmp/parity && /tmp/parity
```

Then open `firmware/rocket_evasion.ino` in the Arduino IDE / PlatformIO, select
Teensy 4.1, and flash.

Requirements: Python 3 + NumPy, and any C++ compiler for the parity check.

---

## Wiring it into *your* rocket

The firmware is a harness with exactly **two** things for you to fill in, both
marked `INTEGRATION POINT` in `firmware/rocket_evasion.ino`:

1. **`read_threat_sensor()`** — return range, closing speed, and bearing /
   elevation off the nose from your radar / ToF / lidar / optical tracker. The
   controller finite-differences the line-of-sight rates itself, so raw range +
   angles are enough.

2. **`existing_control_loop()`** — return your current stabilization fin
   commands. The evasion command is blended *on top* (see `EVASION_AUTHORITY`);
   when no threat is present, your loop flies the vehicle unchanged.

Everything between them — feature assembly, inference, classification, and a
safety gate that holds fire on non-threats — is handled by
`firmware/evasion_controller.h`. The one call you make each loop:

```c
EvasionOutput e = evasion_step(&state,
                               range_m, closing_mps,
                               bearing_rad, elevation_rad,
                               own_speed_mps, dt_s, /*gate=*/1);
// e.fin_pitch, e.fin_yaw  in [-1,1]
// e.threat_class, e.threat_conf, e.evading
```

---

## Repository layout

```
rocketnn/            The library (pure NumPy)
  nn.py                From-scratch MLP: forward, backprop, Adam, save/load
  simulator.py         3D engagement sim, threat models, sensor, expert evader
  data.py              Turns expert engagements into a supervised dataset
scripts/
  train.py             Train + evaluate (held-out + closed-loop), save artifacts
  export_c.py          model.npz -> firmware/model_weights.h + parity_vectors.h
firmware/
  nn_inference.h       Dependency-free C++ forward pass (the deploy target)
  evasion_controller.h Sensor stream -> fin commands + classification + gate
  model_weights.h      GENERATED: architecture, normalisation, weights
  parity_vectors.h     GENERATED: fixtures for the parity test
  host_parity_test.cpp Proves C++ == Python (run on host)
  rocket_evasion.ino   Teensy 4.1 flight sketch (your integration points)
artifacts/             Trained model.npz + metrics.json (committed)
docs/DESIGN.md         Architecture, feature contract, latency, safety, limits
tools/run_all.sh       One-shot train -> export -> verify
```

## Retraining for your hardware

The model is only as good as the simulator it learns from. To match your real
vehicle, edit the constants at the top of `rocketnn/simulator.py` (rocket max-g,
threat speeds, sensor noise, detection range) and re-run `tools/run_all.sh`.
Best of all is to replace the simulator's expert labels with **logged real
engagements** once you have them — the trainer is agnostic to where the
`(features -> action, class)` pairs come from.

See `docs/DESIGN.md` for the full feature contract, latency analysis, safety
notes, and known limitations before you fly anything.
