"""A-line trigger-interval timing analysis (stretch).

Each OCT frame is built from N A-lines acquired at a fixed trigger rate. Jitter
in the A-line trigger interval distorts the azimuthal sampling grid. Given a
sequence of A-line trigger timestamps (or a constant rate + jitter), this module
characterizes the interval distribution and bins it into a histogram.

Also derives lines-per-revolution given the rotation frequency, and the
fractional interval jitter (std/mean of the trigger interval).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ALineTiming:
    """A-line trigger-interval timing result."""

    n_triggers: int
    interval_mean_s: float
    interval_std_s: float
    interval_jitter_pct: float  # std/mean * 100
    rate_hz: float  # 1 / mean interval
    lines_per_rev: float  # rate / f_rot (NaN if f_rot not given)
    hist_counts: np.ndarray
    hist_bin_edges: np.ndarray


def synth_trigger_times(
    n: int,
    rate_hz: float,
    jitter_pct: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate synthetic A-line trigger timestamps at ``rate_hz`` with jitter.

    Each interval is ``(1/rate)*(1 + N(0, jitter_pct))`` so the *interval* jitter
    is a known fraction -- used by the validation test to recover it.

    Returns exactly ``n`` timestamps (``n - 1`` intervals), starting at t = 0.
    """
    if rate_hz <= 0:
        raise ValueError("rate_hz must be > 0")
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    nominal = 1.0 / rate_hz
    intervals = nominal * (1.0 + jitter_pct * rng.standard_normal(n - 1))
    intervals = np.clip(intervals, 1e-12, None)  # intervals must be positive
    times = np.concatenate(([0.0], np.cumsum(intervals)))
    return times


def aline_timing(
    trigger_times: np.ndarray,
    f_rot: float | None = None,
    bins: int = 30,
) -> ALineTiming:
    """Characterize A-line trigger-interval timing from a timestamp series.

    Parameters
    ----------
    trigger_times:
        Monotonic A-line trigger timestamps [s].
    f_rot:
        Rotation frequency [Hz] to derive lines-per-revolution. Optional.
    bins:
        Number of histogram bins over the interval distribution.
    """
    trigger_times = np.asarray(trigger_times, dtype=float)
    if trigger_times.size < 2:
        raise ValueError("need at least 2 trigger timestamps")
    intervals = np.diff(trigger_times)
    mean = float(np.mean(intervals))
    std = float(np.std(intervals))
    jitter_pct = 100.0 * std / mean if mean > 0 else float("nan")
    rate = 1.0 / mean if mean > 0 else float("nan")
    if f_rot is not None and f_rot > 0:
        lines_per_rev = rate / f_rot
    else:
        lines_per_rev = float("nan")
    counts, edges = np.histogram(intervals, bins=bins)
    return ALineTiming(
        n_triggers=int(trigger_times.size),
        interval_mean_s=mean,
        interval_std_s=std,
        interval_jitter_pct=jitter_pct,
        rate_hz=rate,
        lines_per_rev=lines_per_rev,
        hist_counts=counts,
        hist_bin_edges=edges,
    )
