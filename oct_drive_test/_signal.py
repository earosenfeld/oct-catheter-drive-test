"""Shared drive-signal container passed between the generator and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DriveSignal:
    """Time-stamped kinematic record of an OCT catheter drive.

    All arrays share the same length and the same time base ``t``.

    Attributes
    ----------
    t:
        Sample timestamps [s], strictly increasing, nominally uniform at ``fs``.
    theta:
        Measured angular position of the rotating fiber core [rad], unwrapped
        (monotonically increasing for forward rotation; cumulative, not modulo 2pi).
    omega:
        Measured angular velocity [rad/s]. The instantaneous spin rate.
    position:
        Measured linear pullback-stage position [mm].
    fs:
        Sampling frequency [Hz].
    rpm_nominal:
        Commanded nominal rotation rate [revolutions per minute], for reference.
    pullback_speed_nominal:
        Commanded nominal pullback speed [mm/s], for reference.
    theta_cmd:
        Commanded (ideal) angular position [rad], if known. Used for angular
        position-error metrics. ``None`` when not modeled.
    meta:
        Free-form provenance / ground-truth dictionary recording what was injected.
    """

    t: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    position: np.ndarray
    fs: float
    rpm_nominal: float
    pullback_speed_nominal: float
    theta_cmd: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.theta = np.asarray(self.theta, dtype=float)
        self.omega = np.asarray(self.omega, dtype=float)
        self.position = np.asarray(self.position, dtype=float)
        if self.theta_cmd is not None:
            self.theta_cmd = np.asarray(self.theta_cmd, dtype=float)
        n = self.t.shape[0]
        for name in ("theta", "omega", "position"):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(
                    f"{name} length {arr.shape[0]} != t length {n}"
                )
        if self.theta_cmd is not None and self.theta_cmd.shape[0] != n:
            raise ValueError("theta_cmd length must match t length")
        if n >= 2 and not np.all(np.diff(self.t) > 0):
            raise ValueError("t must be strictly increasing")

    @property
    def n_samples(self) -> int:
        return int(self.t.shape[0])

    @property
    def duration(self) -> float:
        """Total record duration [s]."""
        if self.n_samples < 2:
            return 0.0
        return float(self.t[-1] - self.t[0])

    @property
    def f_rot_nominal(self) -> float:
        """Nominal rotation frequency [Hz] = rpm / 60."""
        return self.rpm_nominal / 60.0

    @property
    def n_revolutions(self) -> float:
        """Total revolutions traversed, from the unwrapped angle."""
        if self.n_samples < 2:
            return 0.0
        return float((self.theta[-1] - self.theta[0]) / (2.0 * np.pi))
