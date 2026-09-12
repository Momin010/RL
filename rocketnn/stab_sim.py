"""Flight simulator for active fin stabilization on an F-class rocket.

This models exactly the problem your Teensy + servos have to solve: the rocket
leaves the rail, wind pushes it sideways, and two orthogonal fin pairs (pitch
plane and yaw plane) must keep the tilt near zero. The longitudinal flight -
speed, altitude, dynamic pressure - comes from integrating a *real* thrust
curve (rocketnn/motors.py, thrustcurve.org certification data), because fin
authority scales with dynamic pressure and a controller trained on a fake
thrust profile learns the wrong gain schedule.

Physics (per lateral axis, pitch and yaw treated symmetrically):

    I * theta_ddot = M_static + M_damp + M_fin

  M_static : restoring/upsetting moment from angle of attack. The angle of
             attack is tilt plus the wind-induced component atan(w / v) - this
             is HOW wind knocks a rocket over, and why the problem gets easy as
             the rocket speeds up.
  M_damp   : aerodynamic pitch damping.
  M_fin    : lift from deflected fins; proportional to dynamic pressure, so
             authority is tiny just off the rail and large at max q.

The servo is modelled honestly: slew-rate limited, deflection limited, and one
control period of latency. The controller runs at CTRL_HZ (100 Hz = your
"100 ms streaming start" budget, 10 ms per decision) while physics integrates
at 1 kHz underneath. Sensors are noisy (gyro + attitude estimate), so the
policy must be robust to what a real IMU + Madgwick/Kalman filter gives you.
"""

from __future__ import annotations

import numpy as np

from .motors import Motor, MOTOR_NAMES

# ---- Airframe (edit to match your rocket) -----------------------------------
DRY_MASS = 0.55          # kg, airframe incl. electronics/servos, w/o motor
BODY_DIAM = 0.066        # m
REF_AREA = np.pi * (BODY_DIAM / 2) ** 2
CD_AXIAL = 0.55          # axial drag coefficient
INERTIA = 0.045          # kg m^2, lateral moment of inertia
CN_ALPHA = 8.0           # normal-force slope of whole airframe, per rad
STATIC_MARGIN = 0.10     # m, CP behind CG (positive = passively stable)
DAMP_COEF = 0.9          # aerodynamic pitch damping coefficient
FIN_AREA = 2 * 0.0028    # m^2, one fin pair (control fins)
FIN_CL_DELTA = 3.5       # fin lift slope per rad of deflection
FIN_ARM = 0.32           # m, control fins to CG
FIN_MAX_DEFL = np.deg2rad(15.0)   # mechanical deflection limit
SERVO_RATE = np.deg2rad(400.0)    # fin slew rate limit, rad/s
RAIL_LEN = 1.8           # m launch rail; attitude is locked until clear
RHO = 1.225              # kg/m^3
G = 9.81

CTRL_HZ = 100.0          # controller decision rate (10 ms period)
PHYS_HZ = 1000.0         # physics substep rate
CTRL_DT = 1.0 / CTRL_HZ

GYRO_NOISE = 0.03        # rad/s, 1-sigma
ATT_NOISE = 0.008        # rad,   1-sigma (post-filter attitude estimate)

N_FEATURES = 9
N_OUTPUTS = 2            # fin_pitch, fin_yaw in [-1, 1]

# Loss-of-control threshold: past this tilt an F-class bird is weathercocking
# into the wind and no fin will bring it back.
TILT_FAIL = np.deg2rad(30.0)


def sample_flight(rng, motor_name=None):
    """Randomised flight conditions: motor, steady wind, gusts, mass jitter."""
    return {
        "motor": motor_name or MOTOR_NAMES[int(rng.integers(len(MOTOR_NAMES)))],
        # Steady wind per axis (m/s) + Ornstein-Uhlenbeck gust parameters.
        "wind": rng.uniform(-6.0, 6.0, size=2),
        "gust_sigma": rng.uniform(0.5, 2.5),
        "gust_tau": rng.uniform(0.4, 1.5),
        "mass_scale": rng.uniform(0.9, 1.15),      # payload uncertainty
        "cg_jitter": rng.uniform(-0.03, 0.03),     # static-margin uncertainty
        "tilt0": rng.uniform(-1.0, 1.0, size=2) * np.deg2rad(4.0),  # rail angle
        "seed": int(rng.integers(2 ** 31)),
    }


def make_features(tilt_meas, rate_meas, v_air, q_dyn, fin_prev, burn_frac):
    """The exact 9-value feature vector the firmware assembles each loop."""
    return np.array([
        tilt_meas[0], rate_meas[0],
        tilt_meas[1], rate_meas[1],
        v_air / 50.0,                # airspeed, normalised
        q_dyn / 1500.0,              # dynamic pressure, normalised
        fin_prev[0], fin_prev[1],
        burn_frac,                   # 0..1 through the burn, 1 after burnout
    ])


def expert_action(tilt, rate, q_dyn):
    """Gain-scheduled PD teacher (uses TRUE state - a luxury the net won't get).

    Fin torque scales with q, so the loop gain is kept constant by scheduling
    the PD gains with 1/q. Near the rail (tiny q) commands saturate - that is
    physically correct: there is simply little authority at low speed.
    """
    q_eff = max(q_dyn, 40.0)
    kp, kd = 9.0, 2.2
    sched = 550.0 / q_eff
    cmd = -(kp * tilt + kd * rate) * sched
    return np.clip(cmd, -1.0, 1.0)


def fly(flight, policy, rng=None, record=False):
    """Closed-loop flight. ``policy(features) -> [fin_p, fin_y] in [-1,1]``.

    Returns a dict of metrics; with ``record=True`` also per-step features and
    the expert's action at every visited state (for DAgger relabelling).
    """
    rng = rng or np.random.default_rng(flight["seed"])
    motor = Motor(flight["motor"])
    dry = DRY_MASS * flight["mass_scale"]
    margin = STATIC_MARGIN + flight["cg_jitter"]

    dt = 1.0 / PHYS_HZ
    n_sub = int(PHYS_HZ / CTRL_HZ)

    # Longitudinal state
    v, alt, t = 0.0, 0.0, 0.0
    # Lateral state per axis: tilt, tilt rate
    tilt = flight["tilt0"].copy()
    trate = np.zeros(2)
    gust = np.zeros(2)
    fin = np.zeros(2)          # actual fin deflection (rad)
    fin_cmd_prev = np.zeros(2)  # last commanded, normalised (latency + feature)

    feats_log, expert_log, tilt_log, act_log = [], [], [], []
    max_tilt, sum_t2, n_ctrl = 0.0, 0.0, 0
    failed = False
    t_end = motor.burn_time + 3.0   # burn + 3 s coast: the window that matters

    while t < t_end:
        on_rail = alt < RAIL_LEN
        q_dyn = 0.5 * RHO * v * v

        # ---- controller tick (100 Hz) ----
        wind_p = flight["wind"] + gust
        if not on_rail:
            tilt_meas = tilt + rng.normal(0, ATT_NOISE, 2)
            rate_meas = trate + rng.normal(0, GYRO_NOISE, 2)
            burn_frac = min(t / motor.burn_time, 1.0)
            x = make_features(tilt_meas, rate_meas, v, q_dyn,
                              fin_cmd_prev, burn_frac)
            cmd = np.clip(np.asarray(policy(x), dtype=float), -1.0, 1.0)
            if record:
                # Expert relabels the TRUE state (DAgger-style supervision).
                exp = expert_action(tilt, trate, q_dyn)
                feats_log.append(x)
                expert_log.append(exp)
                act_log.append(cmd.copy())
        else:
            cmd = np.zeros(2)
        # One control period of latency: this tick's command is what the servo
        # chases during the NEXT period (realistic for a polled servo loop).
        target = fin_cmd_prev * FIN_MAX_DEFL
        fin_cmd_prev = cmd

        # ---- physics substeps (1 kHz) ----
        for _ in range(n_sub):
            thrust = motor.thrust(t)
            m = dry + motor.mass(t)
            drag = 0.5 * RHO * v * v * CD_AXIAL * REF_AREA
            v += (thrust - drag - m * G) / m * dt
            v = max(v, 0.0)
            alt += v * dt
            q = 0.5 * RHO * v * v

            # OU gust process
            tau, sig = flight["gust_tau"], flight["gust_sigma"]
            gust += (-gust / tau) * dt + sig * np.sqrt(2 * dt / tau) * rng.normal(size=2)

            if alt >= RAIL_LEN:
                # servo slew toward target deflection
                err = np.clip(target - fin, -SERVO_RATE * dt, SERVO_RATE * dt)
                fin = np.clip(fin + err, -FIN_MAX_DEFL, FIN_MAX_DEFL)

                v_eff = max(v, 3.0)
                alpha = tilt + np.arctan2(flight["wind"] + gust, v_eff)
                m_static = -q * REF_AREA * CN_ALPHA * alpha * margin
                m_damp = -DAMP_COEF * q * REF_AREA * FIN_ARM ** 2 / v_eff * trate
                m_fin = q * FIN_AREA * FIN_CL_DELTA * fin * FIN_ARM
                trate += (m_static + m_damp + m_fin) / INERTIA * dt
                tilt += trate * dt
            t += dt

        if not on_rail:
            a = float(np.max(np.abs(tilt)))
            max_tilt = max(max_tilt, a)
            sum_t2 += float(np.sum(tilt ** 2))
            n_ctrl += 1
            if record:
                tilt_log.append(tilt.copy())
            if a > TILT_FAIL:
                failed = True
                break
        if v <= 0.5 and t > 0.5:   # apogee / fell off the pad
            break

    rms = float(np.sqrt(sum_t2 / max(n_ctrl, 1) / 2))
    out = {
        "max_tilt_deg": float(np.rad2deg(max_tilt)),
        "rms_tilt_deg": float(np.rad2deg(rms)),
        "failed": failed,
        "apogee_m": alt,
        "motor": flight["motor"],
    }
    if record:
        out["feats"] = np.array(feats_log)
        out["expert"] = np.array(expert_log)
        out["actions"] = np.array(act_log)
        out["tilts"] = np.array(tilt_log)
    return out


def policy_zero(x):
    """Fins locked at neutral - the 'no active stabilization' baseline."""
    return np.zeros(2)


def policy_expert(x):
    """The PD teacher driven by the same noisy features the net sees."""
    tilt = np.array([x[0], x[2]])
    rate = np.array([x[1], x[3]])
    q = x[5] * 1500.0
    return expert_action(tilt, rate, q)


def evaluate(policy, n=120, seed=1000, motor_name=None):
    """Closed-loop evaluation over n randomised flights (fixed seed set)."""
    rng = np.random.default_rng(seed)
    res = [fly(sample_flight(rng, motor_name), policy) for _ in range(n)]
    return {
        "median_max_tilt_deg": float(np.median([r["max_tilt_deg"] for r in res])),
        "mean_rms_tilt_deg": float(np.mean([r["rms_tilt_deg"] for r in res])),
        "loss_of_control_pct": 100.0 * np.mean([r["failed"] for r in res]),
        "n": n,
    }
