"""Validation of stretch modules: torque-cable backlash + A-line timing."""

from __future__ import annotations

import numpy as np
import pytest

from oct_drive_test import aline, backlash


# --------------------------- backlash --------------------------------------- #
def test_backlash_zero_lash_is_identity():
    x = np.linspace(0, 10, 200)
    y = backlash.apply_backlash(x, lash=0.0)
    assert np.allclose(x, y)


def test_backlash_lag_bounded_by_half_lash():
    """Under monotonic motion the output lags input by exactly lash/2."""
    x = np.linspace(0, 20, 500)
    lash = 0.4
    y = backlash.apply_backlash(x, lash=lash)
    # After initial engagement, lag settles to lash/2.
    settled = (y - x)[50:]
    assert np.allclose(settled, -lash / 2.0, atol=1e-9)


def test_backlash_deadband_on_reversal():
    """On reversal the output holds for a full lash width before re-engaging."""
    # Triangle wave: up then down.
    up = np.linspace(0, 5, 250)
    down = np.linspace(5, 0, 250)
    x = np.concatenate([up, down])
    lash = 0.5
    y = backlash.apply_backlash(x, lash=lash)
    # At the peak the output equals peak - lash/2 = 4.75.
    peak_out = y[249]
    assert peak_out == pytest.approx(5.0 - lash / 2.0, abs=1e-6)
    # Just after reversal the output should not yet move (deadband).
    # Find first index after peak where input dropped by < lash.
    assert y[260] == pytest.approx(peak_out, abs=1e-9)


def test_estimate_lash_recovers_injected(make_signal):
    """estimate_lash recovers the injected deadband width from paired records."""
    x = np.concatenate([np.linspace(0, 8, 400), np.linspace(8, 0, 400)])
    lash = 0.6
    y = backlash.apply_backlash(x, lash=lash)
    est = backlash.estimate_lash(x, y)
    assert est.lash_estimate == pytest.approx(lash, rel=0.05)
    assert est.n_reversals == 1


# --------------------------- A-line timing ---------------------------------- #
def test_aline_timing_recovers_rate():
    """No jitter -> measured rate equals nominal."""
    times = aline.synth_trigger_times(n=1000, rate_hz=50_000.0, jitter_pct=0.0)
    res = aline.aline_timing(times)
    assert res.rate_hz == pytest.approx(50_000.0, rel=1e-6)
    assert res.interval_jitter_pct == pytest.approx(0.0, abs=1e-6)


def test_aline_timing_recovers_jitter():
    """Injected interval jitter is recovered within sampling error."""
    jit = 0.05  # 5% RMS
    times = aline.synth_trigger_times(
        n=20000, rate_hz=50_000.0, jitter_pct=jit, seed=0
    )
    res = aline.aline_timing(times)
    assert res.interval_jitter_pct == pytest.approx(5.0, rel=0.1)


def test_aline_lines_per_rev():
    """rate / f_rot gives lines-per-revolution."""
    times = aline.synth_trigger_times(n=5000, rate_hz=50_000.0, jitter_pct=0.0)
    res = aline.aline_timing(times, f_rot=100.0)  # 100 rev/s
    assert res.lines_per_rev == pytest.approx(500.0, rel=1e-3)


def test_aline_histogram_shape():
    times = aline.synth_trigger_times(
        n=5000, rate_hz=50_000.0, jitter_pct=0.05, seed=1
    )
    res = aline.aline_timing(times, bins=20)
    assert res.hist_counts.sum() == 4999  # n-1 intervals
    assert res.hist_bin_edges.size == 21


def test_aline_requires_two_timestamps():
    with pytest.raises(ValueError):
        aline.aline_timing(np.array([0.0]))
