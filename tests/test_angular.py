"""Ground-truth validation of angular position-error metrics."""

from __future__ import annotations

import numpy as np
import pytest

from oct_drive_test import angular


def test_clean_angular_error_near_zero(clean_signal):
    """Clean drive: measured angle tracks command -> error ~ 0."""
    err = angular.angular_position_error(clean_signal)
    assert err.max_abs_error_rad < 1e-6
    assert abs(err.drift_rad_per_rev) < 1e-6


def test_zero_mean_nurd_does_not_accumulate(make_signal):
    """Zero-mean NURD oscillates the angle error but does not drift."""
    sig = make_signal(nurd=[(1, 0.2)], jitter_pct=0.0, duration_s=1.0)
    err = angular.angular_position_error(sig)
    # Error oscillates (nonzero max) ...
    assert err.max_abs_error_rad > 0
    # ... but the per-rev drift slope is ~0 (no net accumulation).
    assert abs(err.drift_rad_per_rev) < 0.05


def test_mean_rate_error_accumulates_linearly(make_signal):
    """A constant rate offset accumulates angle error linearly per revolution.

    Build a signal whose actual rate is slightly higher than the commanded rate
    by spinning at rpm but comparing against a command at a different nominal.
    Here we simulate via NURD-free signal where theta_cmd uses omega0; the rate
    matches, so instead we inject a tiny DC via a known trick: scale rpm.
    """
    # Drive truly spins at 6060 rpm but command (theta_cmd) is at 6060 too, so
    # they match. To get linear drift we compare a 1% faster drive to a command
    # built at the nominal: emulate by constructing manually.
    from oct_drive_test.generator import RotationParams, generate_drive_signal

    sig = generate_drive_signal(
        1.0, 50_000.0, rotation=RotationParams(rpm=6000.0), seed=0
    )
    # Inject a 1% faster *actual* rotation while keeping theta_cmd at nominal.
    sig.omega = sig.omega * 1.01
    # Recompute measured theta as integral of the faster omega.
    dt = np.diff(sig.t)
    inc = 0.5 * (sig.omega[1:] + sig.omega[:-1]) * dt
    sig.theta = np.concatenate(([0.0], np.cumsum(inc)))
    err = angular.angular_position_error(sig)
    # ~1% of 2*pi per revolution of accumulating lead.
    assert err.drift_rad_per_rev == pytest.approx(0.01 * 2 * np.pi, rel=0.1)
    # Error grows monotonically with revolution number.
    assert np.all(np.diff(err.error_rad) > 0)


def test_error_units_consistent(make_signal):
    sig = make_signal(nurd=[(1, 0.1)], duration_s=0.5)
    err = angular.angular_position_error(sig)
    assert err.error_deg == pytest.approx(np.degrees(err.error_rad))
    assert err.max_abs_error_deg == pytest.approx(np.degrees(err.max_abs_error_rad))


def test_requires_theta_cmd(make_signal):
    sig = make_signal()
    sig.theta_cmd = None
    with pytest.raises(ValueError, match="theta_cmd"):
        angular.angular_position_error(sig)
