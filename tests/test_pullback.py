"""Ground-truth validation of pullback accuracy/uniformity metrics."""

from __future__ import annotations

import numpy as np
import pytest

from oct_drive_test import pullback


def test_mean_speed_recovers_commanded(make_signal):
    """No error -> measured mean speed equals commanded."""
    sig = make_signal(pullback_mm_s=20.0, duration_s=0.5)
    pb = pullback.pullback_metrics(sig)
    assert pb.mean_speed_mm_s == pytest.approx(20.0, rel=1e-3)
    assert abs(pb.speed_error_pct) < 0.5


def test_speed_error_recovered(make_signal):
    """Injected +8% speed error -> measured error ~ +8%."""
    sig = make_signal(pullback_mm_s=20.0, speed_error_pct=0.08, duration_s=0.5)
    pb = pullback.pullback_metrics(sig)
    assert pb.speed_error_pct == pytest.approx(8.0, abs=0.2)
    assert pb.mean_speed_mm_s == pytest.approx(20.0 * 1.08, rel=1e-3)


def test_speed_error_sign(make_signal):
    """A negative (under-speed) error reads negative."""
    sig = make_signal(pullback_mm_s=20.0, speed_error_pct=-0.05, duration_s=0.5)
    pb = pullback.pullback_metrics(sig)
    assert pb.speed_error_pct == pytest.approx(-5.0, abs=0.2)


def test_error_monotonic(make_signal):
    errs = [-0.05, 0.0, 0.05, 0.10]
    measured = [
        pullback.pullback_metrics(
            make_signal(pullback_mm_s=20.0, speed_error_pct=e, duration_s=0.5)
        ).speed_error_pct
        for e in errs
    ]
    assert all(b > a for a, b in zip(measured, measured[1:]))


def test_clean_linearity_r2_near_one(make_signal):
    """Constant-speed pullback -> position vs time R^2 ~ 1."""
    sig = make_signal(pullback_mm_s=20.0, duration_s=0.5)
    pb = pullback.pullback_metrics(sig)
    assert pb.linearity_r2 > 0.9999


def test_ripple_raises_cv(make_signal):
    """Speed ripple increases the speed CV but not the mean (much)."""
    clean = pullback.pullback_metrics(
        make_signal(pullback_mm_s=20.0, duration_s=1.0)
    )
    rippled = pullback.pullback_metrics(
        make_signal(
            pullback_mm_s=20.0,
            duration_s=1.0,
            ripple_pct=0.1,
            ripple_hz=5.0,
        )
    )
    assert rippled.speed_cv > clean.speed_cv
    # Mean essentially unchanged (ripple is zero-mean).
    assert rippled.mean_speed_mm_s == pytest.approx(20.0, rel=1e-2)


def test_max_deviation_tracks_ripple_amplitude(make_signal):
    """A 10% ripple -> max instantaneous deviation ~ 10% of mean."""
    sig = make_signal(
        pullback_mm_s=20.0, duration_s=1.0, ripple_pct=0.1, ripple_hz=5.0
    )
    pb = pullback.pullback_metrics(sig)
    assert pb.max_abs_deviation_pct == pytest.approx(10.0, abs=2.0)


def test_spin_only_has_nan_speed_error(make_signal):
    """No pullback commanded -> speed error is NaN, linearity trivially 1."""
    sig = make_signal(pullback_mm_s=0.0)
    pb = pullback.pullback_metrics(sig)
    assert np.isnan(pb.speed_error_pct)
    assert pb.linearity_r2 == pytest.approx(1.0)
