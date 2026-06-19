"""Angular position-error metrics.

The OCT image maps each A-line to an azimuthal angle. If the *measured* angle
drifts from the *commanded* angle, azimuthal registration errors accumulate. We
quantify the cumulative commanded-vs-measured angle error, reported per
revolution so a drift trend is visible.

For a drive whose mean rate matches command but has zero-mean NURD, the angle
error oscillates but does not accumulate; a mean-rate error accumulates linearly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._signal import DriveSignal


@dataclass
class AngularError:
    """Angular position-error result bundle (angles in radians and degrees)."""

    rotation_number: np.ndarray  # 0-based revolution index
    error_rad: np.ndarray  # (theta_measured - theta_cmd) at each rev boundary
    error_deg: np.ndarray
    max_abs_error_rad: float
    max_abs_error_deg: float
    rms_error_rad: float
    rms_error_deg: float
    drift_rad_per_rev: float  # linear trend of error vs revolution [rad/rev]


def angular_position_error(signal: DriveSignal) -> AngularError:
    """Cumulative commanded-vs-measured angular error, sampled per revolution.

    Requires ``signal.theta_cmd``. The error is evaluated at each measured
    full-revolution boundary so revolution index aligns with the physical
    rotation count.
    """
    if signal.theta_cmd is None:
        raise ValueError(
            "angular_position_error requires signal.theta_cmd (commanded angle)"
        )

    theta = signal.theta
    theta_cmd = signal.theta_cmd
    theta0 = theta[0]
    total = theta[-1] - theta0
    n_full = int(np.floor(total / (2.0 * np.pi)))

    if n_full < 1:
        empty = np.array([], dtype=float)
        return AngularError(
            rotation_number=np.array([], dtype=int),
            error_rad=empty,
            error_deg=empty,
            max_abs_error_rad=float("nan"),
            max_abs_error_deg=float("nan"),
            rms_error_rad=float("nan"),
            rms_error_deg=float("nan"),
            drift_rad_per_rev=float("nan"),
        )

    targets = theta0 + 2.0 * np.pi * np.arange(1, n_full + 1)
    t_cross = np.interp(targets, theta, signal.t)
    # Measured angle at the crossing is exactly `targets`; commanded angle is
    # interpolated at the same wall-clock time. The error is measured-minus-cmd.
    cmd_at_cross = np.interp(t_cross, signal.t, theta_cmd)
    error = targets - cmd_at_cross

    rev_no = np.arange(1, n_full + 1)
    max_abs = float(np.max(np.abs(error)))
    rms = float(np.sqrt(np.mean(error**2)))

    if n_full >= 2:
        slope = float(np.polyfit(rev_no, error, 1)[0])
    else:
        slope = float("nan")

    return AngularError(
        rotation_number=rev_no - 1,
        error_rad=error,
        error_deg=np.degrees(error),
        max_abs_error_rad=max_abs,
        max_abs_error_deg=float(np.degrees(max_abs)),
        rms_error_rad=rms,
        rms_error_deg=float(np.degrees(rms)),
        drift_rad_per_rev=slope,
    )
