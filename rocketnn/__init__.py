"""rocketnn: a tiny, dependency-light neural-network pipeline for an
on-board rocket evasion controller targeting the Teensy 4.1.

Modules
-------
nn         : a from-scratch multilayer perceptron (forward + backprop + Adam)
simulator  : a 3D engagement simulator, threat models, and an expert evader
data       : turns simulated engagements into a supervised training set
"""

__all__ = ["nn", "simulator", "data"]
