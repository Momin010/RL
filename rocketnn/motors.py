"""Real F-class motor data, harvested from thrustcurve.org.

Every curve below is certified test-stand data (NAR/TRA/CAR certification
firings) downloaded from the thrustcurve.org public API
(https://www.thrustcurve.org/api/v1/download.json) on 2026-07-22. The six
motors were chosen to span the F class: from the long, gentle 3.45 s burn of
the Estes F15 to the violent 0.8 s kick of the Cesaroni F70 — so a controller
trained across them has seen both ends of the dynamic-pressure envelope your
airframe can encounter on an F motor.

Each entry:
  curve       : list of (time_s, thrust_N) samples, linearly interpolated
  total_mass  : loaded motor mass, kg
  prop_mass   : propellant mass, kg (burned off proportionally to impulse)
  diameter_mm : motor diameter (for reference only)
"""

from __future__ import annotations

import numpy as np

MOTORS = {
    "Estes_F15": {
        "curve": [(0, 0), (0.148, 7.638), (0.228, 12.253), (0.294, 16.391), (0.353, 20.21), (0.382, 22.756), (0.419, 25.26), (0.477, 23.074), (0.52, 20.845), (0.593, 19.093), (0.688, 17.5), (0.855, 16.225), (1.037, 15.427), (1.205, 14.948), (1.423, 14.627), (1.452, 15.741), (1.503, 14.785), (1.736, 14.623), (1.955, 14.303), (2.21, 14.141), (2.494, 13.819), (2.763, 13.338), (3.12, 13.334), (3.382, 13.013), (3.404, 9.352), (3.418, 4.895), (3.45, 0)],
        "total_mass": 0.102, "prop_mass": 0.060, "diameter_mm": 29,
    },
    "AeroTech_F12J": {
        "curve": [(0.037, 20.894), (0.054, 22.152), (0.101, 22.152), (0.148, 22.571), (0.165, 23.409), (0.2, 22.421), (0.281, 22.142), (0.369, 22.132), (0.474, 22.271), (0.526, 23.54), (0.549, 21.982), (0.637, 22.122), (0.724, 21.842), (0.8, 21.413), (0.823, 22.251), (0.846, 20.714), (0.881, 21.553), (0.945, 21.123), (1.021, 20.704), (1.114, 20.554), (1.213, 19.296), (1.382, 18.298), (1.481, 18.019), (1.737, 15.343), (1.79, 17.3), (1.883, 13.936), (2.051, 11.26), (2.22, 7.468), (2.447, 3.671), (2.709, 1.135), (2.93, 0)],
        "total_mass": 0.0686, "prop_mass": 0.0303, "diameter_mm": 24,
    },
    "AeroTech_F24W": {
        "curve": [(0.033, 16.442), (0.112, 40.646), (0.125, 41.45), (0.18, 40.927), (0.245, 40.626), (0.281, 41.017), (0.355, 40.024), (0.438, 39.713), (0.543, 38.227), (0.603, 37.032), (0.658, 33.779), (0.685, 34.663), (0.726, 29.934), (0.772, 30.216), (0.951, 26.953), (1.071, 25.166), (1.107, 23.088), (1.185, 21.311), (1.383, 17.144), (1.649, 10.91), (1.828, 5.869), (1.938, 2.903), (1.988, 2.306), (2.048, 1.412), (2.13, 0)],
        "total_mass": 0.0629, "prop_mass": 0.019, "diameter_mm": 24,
    },
    "Quest_F41W": {
        "curve": [(0, 0), (0.018, 0.679), (0.031, 8.821), (0.048, 25.558), (0.07, 30.308), (0.099, 33.249), (0.145, 34.832), (0.231, 36.641), (0.409, 41.843), (0.647, 45.236), (0.884, 47.498), (0.947, 47.724), (0.995, 51.569), (1.032, 51.343), (1.049, 49.081), (1.059, 44.558), (1.077, 35.963), (1.1, 26.689), (1.167, 9.273), (1.2, 0)],
        "total_mass": 0.0569, "prop_mass": 0.0271, "diameter_mm": 24,
    },
    "AeroTech_F44W": {
        "curve": [(0, 0), (0.02, 2.676), (0.026, 6.02), (0.034, 11.204), (0.063, 24.917), (0.135, 59.031), (0.15, 61.372), (0.2, 63.212), (0.299, 65.218), (0.4, 65.051), (0.5, 63.379), (0.6, 58.529), (0.668, 55.017), (0.686, 53.68), (0.687, 54.014), (0.7, 49.165), (0.719, 43.813), (0.774, 22.074), (0.795, 13.211), (0.812, 7.86), (0.825, 4.515), (0.854, 2.007), (0.899, 0.167), (0.998, 0)],
        "total_mass": 0.048, "prop_mass": 0.0197, "diameter_mm": 24,
    },
    "Cesaroni_53F70": {
        "curve": [(0.001, 8.303), (0.013, 85.68), (0.023, 96.149), (0.052, 78.821), (0.1, 83.634), (0.379, 77.858), (0.641, 62.575), (0.665, 55.716), (0.706, 23.947), (0.744, 9.146), (0.816, 0)],
        "total_mass": 0.073, "prop_mass": 0.0225, "diameter_mm": 24,
    },
}

MOTOR_NAMES = sorted(MOTORS.keys())


class Motor:
    """Interpolated thrust and mass-vs-time model of a real motor."""

    def __init__(self, name):
        m = MOTORS[name]
        self.name = name
        pts = np.array(m["curve"], dtype=float)
        self.t = pts[:, 0]
        self.f = pts[:, 1]
        self.burn_time = float(self.t[-1])
        self.total_mass = float(m["total_mass"])
        self.prop_mass = float(m["prop_mass"])
        # Cumulative impulse, so propellant mass burns off in proportion to
        # impulse delivered (a standard and accurate assumption).
        imp = np.concatenate([[0.0], np.cumsum(
            0.5 * (self.f[1:] + self.f[:-1]) * np.diff(self.t))])
        self.total_impulse = float(imp[-1])
        self._imp = imp

    def thrust(self, t):
        if t <= self.t[0] or t >= self.burn_time:
            return 0.0
        return float(np.interp(t, self.t, self.f))

    def mass(self, t):
        """Motor mass at time t (propellant burns off with impulse)."""
        if t >= self.burn_time:
            return self.total_mass - self.prop_mass
        frac = float(np.interp(t, self.t, self._imp)) / self.total_impulse
        return self.total_mass - frac * self.prop_mass
