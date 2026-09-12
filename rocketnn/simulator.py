"""3D engagement simulator, threat models, sensor model, and an expert evader.

The world is a kinematic 3D engagement. Our rocket flies at a roughly constant
speed and steers by commanding lateral acceleration (what fin deflections buy
you at speed) perpendicular to its velocity, split into a pitch component and a
yaw component. An incoming threat approaches on a collision or near-collision
course. The job of the on-board network is, from noisy sensor features, to (a)
recognise *what kind* of thing is inbound and (b) command fin deflections that
maximise the miss distance.

Nothing here runs on the Teensy. This module exists only to manufacture
training data and to score a trained policy in closed loop. It is deliberately
a *kinematic* model (point masses, achievable-lateral-accel steering) rather
than a full six-degree-of-freedom aero model - that is the right altitude for
learning an evasion *policy*, and it keeps the expert labels clean.

Threat classes
--------------
0 NONE      : passing wide / low closing speed. Not a real threat.
1 BALLISTIC : very fast, unguided, aimed on a collision course. Cannot correct.
2 GUIDED    : slower but actively steers (proportional navigation) to intercept.
3 DEBRIS    : tumbling, moderate speed, erratic small accelerations.

The classes have overlapping but distinguishable *motion signatures* (closing
speed, line-of-sight-rate behaviour, range profile), which is exactly what a
real kinematic track classifier keys on.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Physical constants and engagement configuration. SI units throughout
# (metres, seconds, radians). Tuned so that an un-evaded BALLISTIC/GUIDED threat
# scores a near-hit, which makes "miss distance" a meaningful score.
# ----------------------------------------------------------------------------
G = 9.80665
DT = 0.01                      # simulator step, 100 Hz
ROCKET_AMAX = 300.0            # rocket max lateral accel (~30 g)
THREAT_GUIDED_AMAX = 180.0     # guided threat max lateral accel (~18 g); < rocket so evasion can win
PRONAV_N = 3.5                 # proportional-navigation constant for guided threats
DETECT_RANGE = 500.0           # sensor acquisition range (m)
EVADE_TGO_START = 2.5          # begin evading when time-to-go drops below this (s)
EVADE_TGO_FULL = 0.30          # full fin authority by this time-to-go (s)
MAX_STEPS = 800                # hard cap on an engagement (8 s)

CLASS_NONE, CLASS_BALLISTIC, CLASS_GUIDED, CLASS_DEBRIS = 0, 1, 2, 3
CLASS_NAMES = ["none", "ballistic", "guided", "debris"]
N_CLASSES = 4

# Sensor noise (1-sigma). Rates are finite-differenced from noisy angles, so
# they end up considerably noisier than the raw angles - the network has to
# learn to cope, which is realistic.
NOISE_RANGE_FRAC = 0.02        # 2% of range
NOISE_ANGLE = np.radians(0.6)  # bearing/elevation
NOISE_CLOSING = 3.0            # m/s

# Feature vector layout (length 10). Keep in sync with firmware/DESIGN.md.
FEATURE_NAMES = [
    "range_m",          # 0  slant range to threat
    "closing_mps",      # 1  range rate (positive = approaching)
    "bearing_rad",      # 2  horizontal angle off the nose
    "elevation_rad",    # 3  vertical angle off the nose
    "bearing_rate",     # 4  d(bearing)/dt   (line-of-sight sweep, horizontal)
    "elevation_rate",   # 5  d(elevation)/dt (line-of-sight sweep, vertical)
    "tgo_s",            # 6  time-to-go estimate = range / closing
    "rocket_speed_mps", # 7  own forward speed
    "last_fin_pitch",   # 8  previous commanded pitch fin (proprioception)
    "last_fin_yaw",     # 9  previous commanded yaw fin
]
N_FEATURES = len(FEATURE_NAMES)
N_OUTPUTS = 2 + N_CLASSES      # [fin_pitch, fin_yaw, 4 class logits]


# ----------------------------------------------------------------------------
# Vector helpers
# ----------------------------------------------------------------------------
def _norm(v):
    return float(np.sqrt(np.dot(v, v)))


def _unit(v):
    n = _norm(v)
    return v / n if n > 1e-12 else v


def body_frame(vel):
    """Right-handed frame from a velocity vector: (forward, right, up)."""
    fwd = _unit(vel)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(fwd, world_up)) > 0.98:      # nearly vertical: pick another ref
        world_up = np.array([0.0, 1.0, 0.0])
    right = _unit(np.cross(fwd, world_up))
    up = np.cross(right, fwd)
    return fwd, right, up


# ----------------------------------------------------------------------------
# Sensor model
# ----------------------------------------------------------------------------
def true_geometry(rp, rv, tp, tv):
    """Noise-free relative geometry used by the expert and the scorer."""
    r = tp - rp
    R = _norm(r)
    los = r / max(R, 1e-9)
    fwd, right, up = body_frame(rv)
    fdot = float(np.dot(los, fwd))
    bearing = float(np.arctan2(np.dot(los, right), fdot))
    elevation = float(np.arctan2(np.dot(los, up), fdot))
    vrel = tv - rv
    closing = float(-np.dot(vrel, los))
    return dict(r=r, R=R, los=los, fwd=fwd, right=right, up=up,
                bearing=bearing, elevation=elevation, vrel=vrel, closing=closing)


def sense(rp, rv, tp, tv, prev, rng):
    """Return (feature_vector, state) with sensor noise applied.

    ``prev`` carries the previous noisy bearing/elevation and last fin commands
    so we can finite-difference the line-of-sight rates exactly the way the
    Teensy will from its incoming data stream. ``state`` is the updated carry.
    """
    g = true_geometry(rp, rv, tp, tv)
    R_n = g["R"] * (1.0 + NOISE_RANGE_FRAC * rng.standard_normal())
    bearing_n = g["bearing"] + NOISE_ANGLE * rng.standard_normal()
    elev_n = g["elevation"] + NOISE_ANGLE * rng.standard_normal()
    closing_n = g["closing"] + NOISE_CLOSING * rng.standard_normal()

    if prev.get("bearing") is None:
        brate = 0.0
        erate = 0.0
    else:
        brate = (bearing_n - prev["bearing"]) / DT
        erate = (elev_n - prev["elevation"]) / DT

    tgo = R_n / max(closing_n, 1.0)
    rocket_speed = _norm(rv)

    feat = np.array([
        R_n,
        closing_n,
        bearing_n,
        elev_n,
        brate,
        erate,
        tgo,
        rocket_speed,
        prev.get("last_fp", 0.0),
        prev.get("last_fy", 0.0),
    ], dtype=np.float64)

    state = dict(bearing=bearing_n, elevation=elev_n,
                 last_fp=prev.get("last_fp", 0.0), last_fy=prev.get("last_fy", 0.0))
    return feat, state


# ----------------------------------------------------------------------------
# Expert evader (the teacher the network imitates)
# ----------------------------------------------------------------------------
def expert_action(rp, rv, tp, tv, episode_dir):
    """Return (fin_pitch, fin_yaw) in [-1, 1] from the *clean* state.

    Strategy: zero-effort-miss avoidance. Compute the perpendicular offset the
    threat would have at closest approach if nobody maneuvered (the ZEM
    vector), and accelerate to grow that offset. Urgency (fin magnitude) ramps
    up as time-to-go shrinks. On a perfect collision course the ZEM direction is
    undefined, so we fall back to a fixed per-episode break direction.
    """
    g = true_geometry(rp, rv, tp, tv)
    R, closing = g["R"], g["closing"]

    # Not (yet) a closing threat within reach: coast.
    if closing <= 5.0 or R > DETECT_RANGE:
        return 0.0, 0.0

    tgo = R / max(closing, 1.0)
    urgency = (EVADE_TGO_START - tgo) / (EVADE_TGO_START - EVADE_TGO_FULL)
    urgency = float(np.clip(urgency, 0.0, 1.0))
    if urgency <= 0.0:
        return 0.0, 0.0

    vrel_hat = _unit(g["vrel"])
    r = g["r"]
    zem = r - np.dot(r, vrel_hat) * vrel_hat   # perpendicular miss vector
    zem_mag = _norm(zem)

    if zem_mag > 1.0:
        break_dir = -_unit(zem)                # accelerate away from the offset
    else:
        break_dir = episode_dir                # symmetric collision: pick a side

    d_right = float(np.dot(break_dir, g["right"]))
    d_up = float(np.dot(break_dir, g["up"]))
    m = np.hypot(d_right, d_up)
    if m > 1e-9:
        d_right, d_up = d_right / m, d_up / m

    fin_yaw = float(np.clip(urgency * d_right, -1.0, 1.0))
    fin_pitch = float(np.clip(urgency * d_up, -1.0, 1.0))
    return fin_pitch, fin_yaw


# ----------------------------------------------------------------------------
# Dynamics
# ----------------------------------------------------------------------------
def step_rocket(rp, rv, fin_pitch, fin_yaw):
    """Advance the rocket one step under a (clipped) lateral fin command."""
    fp = float(np.clip(fin_pitch, -1.0, 1.0))
    fy = float(np.clip(fin_yaw, -1.0, 1.0))
    mag = np.hypot(fp, fy)
    if mag > 1.0:                              # respect the total accel budget
        fp, fy = fp / mag, fy / mag
    speed = _norm(rv)
    _, right, up = body_frame(rv)
    a_lat = (fy * right + fp * up) * ROCKET_AMAX
    rv_new = rv + a_lat * DT
    rv_new = _unit(rv_new) * speed             # pure turn: hold speed constant
    rp_new = rp + rv_new * DT
    return rp_new, rv_new


def step_threat(tp, tv, rp, rv, kind, rng):
    """Advance the threat one step according to its class behaviour."""
    if kind == CLASS_GUIDED:
        r = rp - tp
        R = max(_norm(r), 1e-9)
        vrel = rv - tv
        omega = np.cross(r, vrel) / (R * R)    # line-of-sight angular velocity
        Vc = -float(np.dot(vrel, r / R))       # closing speed
        a_cmd = PRONAV_N * max(Vc, 0.0) * np.cross(omega, r / R)
        a_mag = _norm(a_cmd)
        if a_mag > THREAT_GUIDED_AMAX:
            a_cmd = a_cmd / a_mag * THREAT_GUIDED_AMAX
        speed = _norm(tv)
        tv_new = _unit(tv + a_cmd * DT) * speed
        tp_new = tp + tv_new * DT
        return tp_new, tv_new
    if kind == CLASS_DEBRIS:
        a = rng.standard_normal(3) * 25.0      # tumbling jitter
        tv_new = tv + a * DT
        return tp + tv_new * DT, tv_new
    # BALLISTIC and NONE: constant velocity.
    return tp + tv * DT, tv


# ----------------------------------------------------------------------------
# Scenario sampling
# ----------------------------------------------------------------------------
def _lead_intercept_dir(rp, rv, tp, speed, aim_point=None):
    """Direction for a threat at ``tp`` moving at ``speed`` to intercept a
    rocket at ``rp`` moving at ``rv`` (assuming the rocket flies straight).
    Falls back to aiming at ``aim_point`` (or the rocket now) if no solution."""
    target = rp if aim_point is None else aim_point
    d0 = target - tp
    a = np.dot(rv, rv) - speed * speed
    b = 2.0 * np.dot(d0, rv)
    c = np.dot(d0, d0)
    t = None
    if abs(a) < 1e-6:
        if abs(b) > 1e-9:
            t = -c / b
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            sq = np.sqrt(disc)
            roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
            roots = [r for r in roots if r > 1e-3]
            if roots:
                t = min(roots)
    if t is None or t <= 0:
        return _unit(target - tp)
    return _unit(d0 + rv * t)


def sample_scenario(rng, kind=None):
    """Sample an initial engagement. Returns a dict describing it."""
    if kind is None:
        kind = int(rng.integers(0, N_CLASSES))

    rocket_speed = float(rng.uniform(150.0, 250.0))
    rp = np.array([0.0, 0.0, float(rng.uniform(200.0, 800.0))])
    rv = np.array([rocket_speed, 0.0, 0.0])

    R0 = float(rng.uniform(280.0, 460.0))

    # Threat bearing/elevation off the nose at acquisition, by class.
    if kind == CLASS_NONE:
        az = np.radians(rng.uniform(16.0, 42.0) * rng.choice([-1, 1]))
        el = np.radians(rng.uniform(10.0, 35.0) * rng.choice([-1, 1]))
        speed = float(rng.uniform(80.0, 200.0))
    elif kind == CLASS_BALLISTIC:
        az = np.radians(rng.uniform(-8.0, 8.0))
        el = np.radians(rng.uniform(-8.0, 8.0))
        speed = float(rng.uniform(320.0, 560.0))
    elif kind == CLASS_GUIDED:
        az = np.radians(rng.uniform(-12.0, 12.0))
        el = np.radians(rng.uniform(-12.0, 12.0))
        speed = float(rng.uniform(180.0, 320.0))
    else:  # DEBRIS
        az = np.radians(rng.uniform(-20.0, 20.0))
        el = np.radians(rng.uniform(-20.0, 20.0))
        speed = float(rng.uniform(90.0, 190.0))

    fwd, right, up = body_frame(rv)
    los = np.cos(el) * (np.cos(az) * fwd + np.sin(az) * right) + np.sin(el) * up
    tp = rp + R0 * los

    if kind in (CLASS_BALLISTIC, CLASS_GUIDED):
        aim = rp                                   # collision course
        if kind == CLASS_BALLISTIC:                # small unguided aim error
            aim = rp + rng.standard_normal(3) * 6.0
        tdir = _lead_intercept_dir(rp, rv, tp, speed, aim_point=aim)
    else:                                          # deliberate wide/near miss
        offset = right * rng.uniform(40.0, 120.0) * rng.choice([-1, 1]) \
            + up * rng.uniform(30.0, 100.0) * rng.choice([-1, 1])
        tdir = _lead_intercept_dir(rp, rv, tp, speed, aim_point=rp + offset)
    tv = tdir * speed

    # Fixed break direction for symmetric-collision fallback.
    ang = rng.uniform(0, 2 * np.pi)
    episode_dir = np.cos(ang) * right + np.sin(ang) * up

    return dict(kind=kind, rp=rp, rv=rv, tp=tp, tv=tv, episode_dir=episode_dir)


# ----------------------------------------------------------------------------
# Engagement runner
# ----------------------------------------------------------------------------
def run_engagement(scn, controller, rng, record=False):
    """Simulate one engagement.

    ``controller`` maps (feature_vector, carry) -> (fin_pitch, fin_yaw). Use
    ``controller="expert"`` to fly the expert directly, ``controller="none"``
    to fly straight, or pass a callable (e.g. the trained network).

    Returns a dict with the miss distance and, if ``record``, the per-step
    (features, expert_action, class) tuples for supervised learning.
    """
    rp, rv = scn["rp"].copy(), scn["rv"].copy()
    tp, tv = scn["tp"].copy(), scn["tv"].copy()
    kind = scn["kind"]
    edir = scn["episode_dir"]

    carry = dict(bearing=None, elevation=None, last_fp=0.0, last_fy=0.0)
    feats, acts = [], []
    min_range = _norm(tp - rp)
    prev_range = min_range
    receding = 0

    for _ in range(MAX_STEPS):
        feat, carry = sense(rp, rv, tp, tv, carry, rng)

        exp_fp, exp_fy = expert_action(rp, rv, tp, tv, edir)
        if controller == "expert":
            fp, fy = exp_fp, exp_fy
        elif controller == "none":
            fp, fy = 0.0, 0.0
        else:
            fp, fy = controller(feat, carry)

        if record:
            feats.append(feat.copy())
            acts.append((exp_fp, exp_fy, kind))

        carry["last_fp"], carry["last_fy"] = fp, fy
        rp, rv = step_rocket(rp, rv, fp, fy)
        tp, tv = step_threat(tp, tv, rp, rv, kind, rng)

        R = _norm(tp - rp)
        min_range = min(min_range, R)
        receding = receding + 1 if R > prev_range else 0
        prev_range = R
        if receding >= 3 and R > 40.0:         # clearly past closest approach
            break

    out = dict(miss=min_range, kind=kind)
    if record:
        out["feats"] = np.array(feats)
        out["acts"] = acts
    return out
