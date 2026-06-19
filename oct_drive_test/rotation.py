"""Rotational stability metrics for the OCT proximal drive.

Beyond the intra-revolution NURD index, the *revolution-to-revolution* timing
stability matters: if successive revolutions take different times, longitudinal
and azimuthal registration drift. Borrowing audio-engineering terms:

- **wow**   : slow rotation-rate variation (low frequency, below ~a few Hz of
              modulation of the rate) -- e.g. gradual drag build-up.
- **flutter**: fast rotation-rate variation (higher-frequency rate modulation)
              -- e.g. per-revolution stick-slip.

Period-to-period metrics
------------------------
The rotation period of revolution ``i`` is the time between consecutive
2*pi angle crossings. From the sequence of periods ``T_i`` we report:

- mean / std of T_i,
- std as a percentage of the nominal period,
- peak-to-peak as a percentage of the nominal period,
- RPM mean/std vs nominal (RPM_i = 60 / T_i).

Wow/flutter split
-----------------
The fractional period deviation series ``(T_i - T_nominal)/T_nominal`` is
sampled at one point per revolution (rate = f_rot). Its variation is split by a
modulation-frequency cutoff (default 2 Hz of *rate* modulation) into a low-band
(wow) RMS and high-band (flutter) RMS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._signal import DriveSignal


@dataclass
class RotationStability:
    """Rotational-stability result bundle."""

    periods_s: np.ndarray  # per-revolution period T_i [s]
    period_mean_s: float
    period_std_s: float
    period_std_pct: float  # std(T_i) / T_nominal * 100
    period_p2p_pct: float  # (max-min)(T_i) / T_nominal * 100
    rpm_mean: float
    rpm_std: float
    rpm_nominal: float
    rpm_error_pct: float  # (rpm_mean - rpm_nominal)/rpm_nominal * 100
    wow_rms_pct: float  # low-band fractional period deviation RMS [%]
    flutter_rms_pct: float  # high-band fractional period deviation RMS [%]
    wow_flutter_split_hz: float


def _crossing_times(t: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Linear-interpolated times at which the unwrapped angle hits each 2*pi.

    Returns the wall-clock times of successive full-revolution boundaries.
    """
    theta0 = theta[0]
    total = theta[-1] - theta0
    n_full = int(np.floor(total / (2.0 * np.pi)))
    if n_full < 1:
        return np.array([], dtype=float)
    targets = theta0 + 2.0 * np.pi * np.arange(1, n_full + 1)
    # theta is monotincreasing for forward rotation -> interp is well-defined.
    return np.interp(targets, theta, t)


def rotation_periods(signal: DriveSignal) -> np.ndarray:
    """Per-revolution rotation periods T_i [s] from angle crossings."""
    times = _crossing_times(signal.t, signal.theta)
    if times.size < 2:
        return np.array([], dtype=float)
    return np.diff(times)


def rotation_stability(
    signal: DriveSignal, wow_flutter_split_hz: float = 2.0
) -> RotationStability:
    """Compute rotational-stability metrics from the angle record."""
    periods = rotation_periods(signal)
    t_nom = 60.0 / signal.rpm_nominal  # nominal period [s]

    if periods.size == 0:
        nan = float("nan")
        return RotationStability(
            periods_s=periods,
            period_mean_s=nan,
            period_std_s=nan,
            period_std_pct=nan,
            period_p2p_pct=nan,
            rpm_mean=nan,
            rpm_std=nan,
            rpm_nominal=signal.rpm_nominal,
            rpm_error_pct=nan,
            wow_rms_pct=nan,
            flutter_rms_pct=nan,
            wow_flutter_split_hz=wow_flutter_split_hz,
        )

    period_mean = float(np.mean(periods))
    period_std = float(np.std(periods))
    period_p2p = float(np.max(periods) - np.min(periods))

    rpm_i = 60.0 / periods
    rpm_mean = float(np.mean(rpm_i))
    rpm_std = float(np.std(rpm_i))
    rpm_error_pct = 100.0 * (rpm_mean - signal.rpm_nominal) / signal.rpm_nominal

    # Fractional period-deviation series, one sample per revolution.
    frac_dev = (periods - t_nom) / t_nom
    f_sample = 1.0 / period_mean  # ~ f_rot, revolutions per second
    wow_rms, flutter_rms = _wow_flutter(
        frac_dev, f_sample, wow_flutter_split_hz
    )

    return RotationStability(
        periods_s=periods,
        period_mean_s=period_mean,
        period_std_s=period_std,
        period_std_pct=100.0 * period_std / t_nom,
        period_p2p_pct=100.0 * period_p2p / t_nom,
        rpm_mean=rpm_mean,
        rpm_std=rpm_std,
        rpm_nominal=signal.rpm_nominal,
        rpm_error_pct=rpm_error_pct,
        wow_rms_pct=100.0 * wow_rms,
        flutter_rms_pct=100.0 * flutter_rms,
        wow_flutter_split_hz=wow_flutter_split_hz,
    )


def _wow_flutter(
    frac_dev: np.ndarray, f_sample: float, split_hz: float
) -> tuple[float, float]:
    """Split a per-revolution fractional-deviation series into wow/flutter RMS.

    The series is treated as a signal sampled at ``f_sample`` (rev/s). Its
    spectral energy below ``split_hz`` is wow; above is flutter. Implemented via
    rfft so it needs no SciPy and degrades gracefully for short series.
    """
    n = frac_dev.size
    if n < 2:
        return float("nan"), float("nan")
    x = frac_dev - np.mean(frac_dev)  # remove DC (constant bias is not w/f)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / f_sample)
    # Parseval-consistent power per bin (one-sided).
    power = (np.abs(spec) ** 2)
    low = freqs <= split_hz
    high = freqs > split_hz
    total = np.sum(power)
    if total <= 0:
        return 0.0, 0.0
    var = np.var(x)
    wow_var = var * np.sum(power[low]) / total
    flutter_var = var * np.sum(power[high]) / total
    return float(np.sqrt(wow_var)), float(np.sqrt(flutter_var))
