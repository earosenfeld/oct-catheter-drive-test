"""Pullback speed accuracy and uniformity metrics.

The linear pullback stage translates the rotating optics along the vessel; its
speed sets the *longitudinal* image scaling. Speed error stretches/compresses
the reconstructed length; speed non-uniformity warps it locally.

Metrics
-------
- mean / median measured speed vs commanded [mm/s] and percent error,
- speed coefficient of variation CV = std(v)/mean(v),
- maximum instantaneous deviation from the mean [mm/s and %],
- linearity residual R^2 of a straight-line fit to position(t): how close the
  motion is to constant-velocity (R^2 -> 1 is ideal).

Speed is estimated from the measured position by finite difference. To avoid
amplifying sample noise, an optional smoothing window can be applied; by default
a light centered difference is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._signal import DriveSignal


@dataclass
class PullbackMetrics:
    """Pullback accuracy/uniformity result bundle."""

    commanded_speed_mm_s: float
    mean_speed_mm_s: float
    median_speed_mm_s: float
    speed_error_pct: float  # (mean - commanded)/commanded * 100
    speed_cv: float  # std/mean of instantaneous speed
    speed_cv_pct: float
    max_abs_deviation_mm_s: float  # max |v - mean|
    max_abs_deviation_pct: float  # as % of mean
    linearity_r2: float  # R^2 of position vs time straight-line fit


def estimate_speed(signal: DriveSignal, smooth: int = 1) -> np.ndarray:
    """Instantaneous pullback speed [mm/s] from measured position via gradient.

    Parameters
    ----------
    smooth:
        Moving-average window (samples) applied to position before differencing
        to suppress quantization/noise. ``1`` disables smoothing.
    """
    pos = signal.position
    if smooth and smooth > 1:
        kernel = np.ones(smooth) / smooth
        pos = np.convolve(pos, kernel, mode="same")
    return np.gradient(pos, signal.t)


def pullback_metrics(signal: DriveSignal, smooth: int = 1) -> PullbackMetrics:
    """Compute pullback accuracy/uniformity metrics.

    If the commanded speed is 0 (spin-only acquisition), error/CV fields are NaN
    but linearity is still reported (trivially flat).
    """
    v_cmd = signal.pullback_speed_nominal
    v = estimate_speed(signal, smooth=smooth)

    # Trim edges where np.gradient is one-sided / convolution is biased.
    edge = max(smooth, 2)
    if v.size > 2 * edge:
        v_core = v[edge:-edge]
    else:
        v_core = v

    mean_v = float(np.mean(v_core))
    median_v = float(np.median(v_core))
    std_v = float(np.std(v_core))

    if v_cmd != 0.0:
        speed_error_pct = 100.0 * (mean_v - v_cmd) / v_cmd
    else:
        speed_error_pct = float("nan")

    if mean_v != 0.0:
        cv = std_v / mean_v
        max_dev = float(np.max(np.abs(v_core - mean_v)))
        max_dev_pct = 100.0 * max_dev / abs(mean_v)
    else:
        cv = float("nan")
        max_dev = float(np.max(np.abs(v_core - mean_v))) if v_core.size else float("nan")
        max_dev_pct = float("nan")

    r2 = _linear_r2(signal.t, signal.position)

    return PullbackMetrics(
        commanded_speed_mm_s=v_cmd,
        mean_speed_mm_s=mean_v,
        median_speed_mm_s=median_v,
        speed_error_pct=speed_error_pct,
        speed_cv=cv,
        speed_cv_pct=100.0 * cv if not np.isnan(cv) else float("nan"),
        max_abs_deviation_mm_s=max_dev,
        max_abs_deviation_pct=max_dev_pct,
        linearity_r2=r2,
    )


def _linear_r2(t: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of determination of a least-squares straight-line fit y~t."""
    if t.size < 2:
        return float("nan")
    # Degenerate (no motion): a flat line fits perfectly.
    if np.allclose(y, y[0]):
        return 1.0
    coeffs = np.polyfit(t, y, 1)
    y_hat = np.polyval(coeffs, t)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot
