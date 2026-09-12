# Rocket Stabilization + Evasion Neural Networks - Teensy 4.1

Two trained networks, one firmware image:

1. **Stabilization net** (`scripts/train_stab.py`) - the always-on inner loop.
   Reads tilt + body rates from your IMU and drives the fin servos to keep the
   rocket vertical against wind. Trained on **real F-class motor thrust
   curves** pulled from thrustcurve.org certification data (six motors, from
   the 3.45 s Estes F15 to the 0.8 s Cesaroni F70), through a
   harness-data → train → validate → DAgger → **RL fine-tune** cycle.
2. **Evasion net** (`scripts/train.py`) - classifies an incoming object and
   commands an evasive fin blend on top of stabilization (details below).

## Stabilization results (`artifacts/stab_metrics.json`)

Closed-loop over 120 held-out randomised flights (6 real motors, steady wind
up to 6 m/s per axis + gusts, sensor noise, servo rate limits, 10 ms latency):

| Controller | Median max tilt | Mean RMS tilt | Loss of control |
|---|---:|---:|---:|
| Fins locked (no control) | ~23° | ~9° | ~20% of flights |
| **Trained net (100 Hz)** | **~3°** | **~1.3°** | **0%** |
| Gain-scheduled PD expert (true state) | ~3° | ~1.3° | 0% |

The net matches its teacher while running from *noisy* sensor features only -
and it costs a few microseconds per tick on the Teensy's FPU, so your 100 Hz
(10 ms) loop budget is >99% free for sensing.

Stabilization quickstart:

```bash
python3 scripts/train_stab.py      # harness data -> BC -> DAgger -> RL (~min)
python3 scripts/export_stab_c.py   # -> firmware/stab_model_weights.h
g++ -O2 -std=c++14 -I firmware firmware/host_stab_parity_test.cpp -o parity_stab && ./parity_stab
```

Then in `firmware/rocket_evasion.ino` fill in `read_attitude()` (INTEGRATION
POINT 2) with your IMU/attitude-filter output - the trained net now *is* the
`existing_control_loop()`.

---

# Part 2: the evasion network

An end-to-end TinyML pipeline for an on-board **evasion controller**: sense an
incoming object, recognise *what it is*, and command fin deflections to
**maximise miss distance** - fast enough to run inside a hard real-time flight
loop on a Teensy 4.1.

It ships as three cooperating parts:

1. a **trainer** (pure NumPy, no PyTorch/TensorFlow) that learns the policy from
   a physics simulator,
2. a **weight exporter** that bakes the trained network into a C header, and
3. **firmware** - a dependency-free C++ inference engine plus a Teensy sketch
   that drops into your existing servo loop.

---

## The one thing to understand first

**You do not train a neural network *on* a Teensy - and you don't want to.**
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
M7's FPU - so inference uses well under **1%** of even a 1 kHz control loop.
Compute is nowhere near your bottleneck; your sensor's update rate is.

---

## Results (measured, `artifacts/metrics.json`)

Closed-loop **median miss distance** over 120 held-out engagements per class -
flying straight (no evasion) vs. the trained network vs. the expert it learned
from. Bigger is better; a "hit" is a few metres.

| Inbound threat        | No evasion | **Trained net** | Expert (teacher) |
|-----------------------|-----------:|----------------:|-----------------:|
| **Guided** (tracks you) | **1.2 m** (dead hit) | **29.6 m** | 39.2 m |
| **Ballistic** (fast, unguided) | **7.1 m** (near hit) | **37.3 m** | 55.9 m |
| Debris (tumbling)     | 98.5 m | 133.8 m | 186.4 m |
| None (passing wide)   | 106.3 m | 146.4 m | 194.9 m |

The network converts a **1.2 m guaranteed intercept into a ~30 m miss**, from
*noisy, single-frame* sensor data, capturing ~70-75% of the expert's benefit.

**Threat classification:** 89.3% overall 4-way accuracy - and for the two
classes that can actually kill you it is far higher: **ballistic 95.4%**,
**guided 97.5%**. Most residual error is `none`↔`debris`, both non-collision.

**C++/Python parity:** the on-device engine reproduces the trained network to
**< 1e-5** (float32 rounding). Proven automatically by the host parity test.

---

## Quickstart

```bash
# Train, export weights to C, then build+run the parity test - all in one:
bash tools/run_all.sh
```

Individually:

```bash
python3 scripts/train.py       # -> artifacts/model.npz + metrics.json  (~2 min)
python3 scripts/export_c.py    # -> firmware/model_weights.h + parity_vectors.h
g++ -O2 -std=c++14 -I firmware firmware/host_parity_test.cpp -o /tmp/parity && /tmp/parity
```

Then flash the sketch for your board with Arduino IDE / PlatformIO:

- **Teensy 4.1**: `firmware/rocket_evasion.ino`
- **ESP32 / ESP32-S2 / ESP32-S3**: `firmware/rocket_esp32.ino` (Arduino core
  3.x; servos are driven by the LEDC peripheral, no Servo library needed).
  Note the S2 has no hardware FPU - inference is software-float but still fits
  a 100 Hz loop easily; the plain ESP32 and S3 have a single-precision FPU.

The exported weight headers are plain portable C - the SAME files work on both
boards, so the parity test covers both.

Requirements: Python 3 + NumPy, and any C++ compiler for the parity check.

---

## Wiring it into *your* rocket

The firmware is a harness with exactly **two** things for you to fill in, both
marked `INTEGRATION POINT` in `firmware/rocket_evasion.ino`:

1. **`read_threat_sensor()`** - return range, closing speed, and bearing /
   elevation off the nose from your radar / ToF / lidar / optical tracker. The
   controller finite-differences the line-of-sight rates itself, so raw range +
   angles are enough.

2. **`existing_control_loop()`** - return your current stabilization fin
   commands. The evasion command is blended *on top* (see `EVASION_AUTHORITY`);
   when no threat is present, your loop flies the vehicle unchanged.

Everything between them - feature assembly, inference, classification, and a
safety gate that holds fire on non-threats - is handled by
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
  motors.py            REAL F-class thrust curves (thrustcurve.org cert data)
  stab_sim.py          Wind/attitude flight sim + gain-scheduled PD expert
  simulator.py         3D engagement sim, threat models, sensor, expert evader
  data.py              Turns expert engagements into a supervised dataset
scripts/
  train_stab.py        Stabilization: harness -> BC -> DAgger -> RL fine-tune
  export_stab_c.py     stab_model.npz -> firmware/stab_model_weights.h
  train.py             Evasion: train + evaluate, save artifacts
  export_c.py          model.npz -> firmware/model_weights.h + parity_vectors.h
firmware/
  nn_inference.h       Dependency-free C++ forward pass (the deploy target)
  stabilization_controller.h  IMU features -> fin commands (trained stab net)
  stab_model_weights.h GENERATED: stabilization net weights
  host_stab_parity_test.cpp   Proves stab C++ == Python (run on host)
  evasion_controller.h Sensor stream -> fin commands + classification + gate
  model_weights.h      GENERATED: architecture, normalisation, weights
  parity_vectors.h     GENERATED: fixtures for the parity test
  host_parity_test.cpp Proves C++ == Python (run on host)
  rocket_evasion.ino   Teensy 4.1 flight sketch (your integration points)
  rocket_esp32.ino     ESP32 / ESP32-S2 / ESP32-S3 flight sketch (same brain)
artifacts/             Trained model.npz + metrics.json (committed)
docs/DESIGN.md         Architecture, feature contract, latency, safety, limits
tools/run_all.sh       One-shot train -> export -> verify
```

## Retraining for your hardware

The model is only as good as the simulator it learns from. To match your real
vehicle, edit the constants at the top of `rocketnn/simulator.py` (rocket max-g,
threat speeds, sensor noise, detection range) and re-run `tools/run_all.sh`.
Best of all is to replace the simulator's expert labels with **logged real
engagements** once you have them - the trainer is agnostic to where the
`(features -> action, class)` pairs come from.

See `docs/DESIGN.md` for the full feature contract, latency analysis, safety
notes, and known limitations before you fly anything.
