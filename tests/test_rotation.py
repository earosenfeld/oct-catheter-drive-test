"""Ground-truth validation of rotational-stability metrics."""

from __future__ import annotations

import numpy as np
import pytest

from oct_drive_test import rotation


def test_clean_periods_match_nominal(clean_signal):
    """Clean 6000 rpm -> period 10 ms, ~zero std."""
    stab = rotation.rotation_stability(clean_signal)
    assert stab.period_mean_s == pytest.approx(0.010, rel=1e-4)
    assert stab.period_std_pct < 1e-3
    assert stab.rpm_mean == pytest.approx(6000.0, rel=1e-4)
    assert abs(stab.rpm_error_pct) < 1e-3


def test_rpm_mean_recovers_nominal_for_various_rpm(make_signal):
    for rpm in (3000.0, 6000.0, 9000.0):
        sig = make_signal(rpm=rpm, duration_s=0.5)
        stab = rotation.rotation_stability(sig)
        assert stab.rpm_mean == pytest.approx(rpm, rel=1e-3)


def test_jitter_increases_period_std(make_signal):
    """Injected jitter raises the period-to-period std monotonically."""
    s0 = rotation.rotation_stability(make_signal(jitter_pct=0.0)).period_std_pct
    s1 = rotation.rotation_stability(
        make_signal(jitter_pct=0.01, seed=1)
    ).period_std_pct
    s2 = rotation.rotation_stability(
        make_signal(jitter_pct=0.03, seed=1)
    ).period_std_pct
    assert s0 < s1 < s2
    assert s0 < 1e-2  # clean is essentially zero


def test_nurd_does_not_bias_mean_rpm(make_signal):
    """Zero-mean NURD modulation must not shift the mean RPM."""
    sig = make_signal(nurd=[(1, 0.2)], jitter_pct=0.0, duration_s=1.0)
    stab = rotation.rotation_stability(sig)
    assert stab.rpm_mean == pytest.approx(6000.0, rel=1e-3)


def test_wow_vs_flutter_band_separation(make_signal):
    """Slow rate modulation lands in wow; fast in flutter.

    A 1x NURD at f_rot=100 Hz modulates the *rate* at 100 Hz, which is far above
    a 2 Hz wow/flutter split -> should appear as flutter, not wow.
    """
    sig = make_signal(nurd=[(1, 0.1)], jitter_pct=0.0, duration_s=1.0)
    stab = rotation.rotation_stability(sig, wow_flutter_split_hz=2.0)
    # Per-rev sampling of a 1x modulation aliases but its energy is high-band
    # relative to the very low split -> flutter dominates wow.
    assert stab.flutter_rms_pct >= stab.wow_rms_pct


def test_period_p2p_nonnegative(make_signal):
    sig = make_signal(jitter_pct=0.02, seed=3)
    stab = rotation.rotation_stability(sig)
    assert stab.period_p2p_pct >= stab.period_std_pct >= 0.0


def test_empty_when_too_short():
    """A sub-revolution record yields no periods and NaN stats (no crash)."""
    from oct_drive_test.generator import RotationParams, generate_drive_signal

    # 6000 rpm = 100 rev/s; 1 ms is only 0.1 rev.
    sig = generate_drive_signal(
        0.001, 200_000.0, rotation=RotationParams(rpm=6000.0)
    )
    stab = rotation.rotation_stability(sig)
    assert stab.periods_s.size == 0
    assert np.isnan(stab.rpm_mean)
