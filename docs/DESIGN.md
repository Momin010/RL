# Design notes

## 1. Problem framing

The on-board task is a reflex: given a stream of measurements about an
approaching object, decide **what it is** and command fin deflections that
**increase the miss distance**, blended on top of the vehicle's normal
stabilization. This is a control-policy problem, so we learn a policy by
**imitation**: a physics simulator generates engagements, a hand-written
*expert* evader produces the "right" fin commands, and a small network learns to
reproduce them from realistic (noisy) sensor features.

Why imitation rather than reinforcement learning (the repo is named `RL`):
imitation gives dense, stable supervision and trains in ~2 minutes with no
reward shaping or exploration instability. It is the pragmatic v1. The expert's
median miss distance is the network's performance ceiling; §7 covers how to push
past it with RL fine-tuning later.

## 2. Network architecture

A fully-connected MLP, deliberately tiny so it is fast and fully auditable:

```
input 10  ──▶  Dense 24 + ReLU  ──▶  Dense 16 + ReLU  ──▶  Dense 6 (linear)
```

- **766 parameters**, ~3 KB as float32.
- ReLU hidden activations (a single `max(0,x)` on the MCU — no `exp`/`tanh`
  tables in the hot path).
- One shared trunk with a combined output head:
  - outputs `[0], [1]` = pitch / yaw fin commands (regression, clamped to
    `[-1, 1]`),
  - outputs `[2..5]` = threat-class logits (argmax + softmax confidence).

Sharing the trunk means the features that identify *what* is inbound also inform
*how* to dodge it, which is both efficient and well-matched to the task.

## 3. Feature contract (the interface that must never drift)

The host trainer and the device firmware must agree exactly on the 10-element
input vector, its order, and its units. This is the contract:

| # | Feature          | Unit  | Source on the Teensy                         |
|---|------------------|-------|----------------------------------------------|
| 0 | range            | m     | tracker                                       |
| 1 | closing speed    | m/s   | tracker (range rate, + = approaching)         |
| 2 | bearing          | rad   | tracker (horizontal angle off nose)           |
| 3 | elevation        | rad   | tracker (vertical angle off nose)             |
| 4 | bearing rate     | rad/s | finite-differenced in `evasion_step`          |
| 5 | elevation rate   | rad/s | finite-differenced in `evasion_step`          |
| 6 | time-to-go       | s     | `range / max(closing, 1)`, computed for you   |
| 7 | own speed        | m/s   | IMU / pitot / baro                            |
| 8 | last fin pitch   | —     | carried by `EvasionState` (proprioception)    |
| 9 | last fin yaw     | —     | carried by `EvasionState`                     |

Defined once in `rocketnn/simulator.py::FEATURE_NAMES`, assembled identically in
`firmware/evasion_controller.h::evasion_step`. Inputs are standardised
(`(x - mean) / std`) using statistics baked into `model_weights.h`, so the
device and host normalise identically. **If you add/reorder a feature, retrain
and re-export — never hand-edit the header.**

## 4. Latency budget

One forward pass is 10→24→16→6 = **~736 multiply-accumulates**. On the Teensy
4.1 (600 MHz Cortex-M7, single-precision FPU) that is a few microseconds even
pessimistically (a hard upper bound of ~12 µs at 10 cycles/MAC; realistically
lower). The flight sketch measures the true value with the ARM DWT cycle counter
and prints it over serial, so you can confirm on your own board.

Consequences:
- At a **1 kHz** loop (1000 µs budget) inference is well under 1%.
- The real reaction-time driver is your **sensor update rate**, not compute. The
  "100 ms streaming start" you described is a sensor/pipeline latency target;
  the network does not meaningfully add to it.

Determinism: no dynamic allocation, no data-dependent branching beyond ReLU, no
blocking calls — the pass takes the same time every loop.

## 5. Simulator and expert (what the network learned from)

- **Kinematic 3D engagement.** The rocket holds roughly constant speed and
  steers via achievable lateral acceleration (~30 g), split into pitch/yaw. This
  is the right altitude for learning a *policy*; it is **not** a 6-DOF aero
  model.
- **Threat classes** have distinct motion signatures: BALLISTIC (fast, unguided,
  collision course), GUIDED (proportional-navigation pursuer, lower max-g so
  evasion can win), DEBRIS (tumbling), NONE (passing wide). Un-evaded
  ballistic/guided threats are set up to score a near-hit, which is what makes
  "miss distance" a meaningful score.
- **Sensor model** adds range/angle/closing noise and finite-differences the
  line-of-sight rates — the same computation the firmware does — so the network
  trains on the noise characteristics it will actually see.
- **Expert law:** zero-effort-miss avoidance — estimate the perpendicular offset
  at closest approach and accelerate to grow it, with urgency ramped by
  time-to-go, falling back to a fixed break direction on a perfectly symmetric
  collision.

## 6. Safety notes — read before flying

This is a research/education controller trained in simulation. Treat it
accordingly:

- **Sim-to-real gap is real.** The policy is only as faithful as the simulator
  constants in `rocketnn/simulator.py`. Tune them to your vehicle, and prefer
  retraining on **logged real engagements** once you have them.
- **It is not a safety system of record.** Keep independent hardware failsafes:
  the sketch centres the fins whenever disarmed (`PIN_ARM`), and the controller
  gate holds fire on non-threats. Do not remove these.
- **Bench-test the full path first** — feed recorded/synthetic tracks through
  `read_threat_sensor()` and watch the servo commands before any powered
  flight.
- **Actuator limits.** The command is a normalised deflection; make sure
  `SERVO_TRAVEL_US` and your linkage cannot drive the fins past their
  mechanical/aero-stall limits.
- **Comply with local rules** for powered rocketry and RF sensors.

## 7. Known limitations & next steps

- **Feed-forward, near-single-frame.** Temporal reasoning is limited to the two
  fed-back last-command inputs. A tiny GRU/TCN would improve guided-vs-ballistic
  discrimination and enable learned weave patterns; it also ports cleanly to C.
- **Imitation ceiling.** The net cannot beat its teacher. Fine-tuning the policy
  with RL (e.g. policy-gradient on miss distance) in the same simulator would
  let it exceed the expert — the simulator and closed-loop scorer needed for
  that already live in `rocketnn/simulator.py`.
- **float32, not int8.** float32 is the right call on the M7's FPU (accuracy +
  simplicity, still microseconds). int8 quantisation would shrink flash further
  if you ever move to a smaller MCU; the exporter is the place to add it.
- **Evasion only.** By design this maximises miss distance; it does not target
  or intercept anything.
