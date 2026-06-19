"""OCT catheter drive-characterization toolkit.

Bench-test tooling for characterizing the *drive mechanics* of an intravascular
OCT (Optical Coherence Tomography) imaging catheter: proximal rotary drive that
spins the fiber-optic core and the linear pullback stage that translates it.

This is engineering / V&V characterization of mechanical drive performance only.
It makes NO clinical, diagnostic, or medical-efficacy claims.

Public API
----------
- generator: synthetic ground-truth drive signals (rotation + pullback) with
  injectable NURD, jitter, and pullback speed error.
- nurd: Non-Uniform Rotational Distortion metrics.
- rotation: rotational stability (wow/flutter, period-to-period, RPM stats).
- pullback: pullback speed accuracy / uniformity / linearity.
- angular: cumulative angular position error per revolution.
- spec: spec loading + PASS/FAIL characterization report generation.
- backlash: torque-cable wind-up / backlash deadband model (stretch).
- aline: A-line trigger-interval timing histogram (stretch).
"""

from __future__ import annotations

from . import angular, backlash, generator, nurd, pullback, rotation, spec
from ._signal import DriveSignal
from .generator import (
    NurdHarmonic,
    PullbackParams,
    RotationParams,
    generate_drive_signal,
)

__all__ = [
    "DriveSignal",
    "NurdHarmonic",
    "RotationParams",
    "PullbackParams",
    "generate_drive_signal",
    "generator",
    "nurd",
    "rotation",
    "pullback",
    "angular",
    "spec",
    "backlash",
    "aline",
]

__version__ = "0.1.0"
