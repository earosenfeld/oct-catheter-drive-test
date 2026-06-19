"""Validate the synthetic generator itself (the ground-truth instrument).

If the generator is not internally self-consistent, no downstream metric
validation is trustworthy. We check that theta is the integral of omega, that
the mean rate matches nominal, and that injected NURD shows up as the intended
fractional velocity modulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from oct_drive_test.generator import (
    NurdHarmonic,
    PullbackParams,
    RotationParams,
    generate_drive_signal,
)


def test_theta_is_integral_of_omega(make_signal):
    """d(theta)/dt must equal omega (generator self-consistency)."""
    sig = make_signal(nurd=[(1, 0.1), (3, 0.05)], jitter_pct=0.0)
    dtheta = np.gradient(sig.theta, sig.t)
    # Interior samples (gradient edges are one-sided) should match omega tightly.
    core = slice(5, -5)
    assert np.allclose(dtheta[core], sig.omega[core], rtol=1e-3, atol=1e-3 * sig.meta["omega0_rad_s"])


def test_clean_mean_rate_matches_nominal(clean_signal):
    """Mean omega of a clean signal equals omega0 = 2*pi*f_rot."""
    omega0 = clean_signal.meta["omega0_rad_s"]
    assert np.mean(clean_signal.omega) == pytest.approx(omega0, rel=1e-9)
    # std should be essentially zero for a clean signal.
    assert np.std(clean_signal.omega) / omega0 < 1e-9


def test_injected_nurd_modulation_amplitude(make_signal):
    """A single injected 1x harmonic of amplitude a -> omega swings by ~+/- a."""
    a = 0.2
    sig = make_signal(nurd=[(1, a)], jitter_pct=0.0)
    omega0 = sig.meta["omega0_rad_s"]
    frac = sig.omega / omega0 - 1.0  # fractional modulation
    # Peak fractional modulation recovers the injected amplitude.
    assert np.max(frac) == pytest.approx(a, abs=0.01)
    assert np.min(frac) == pytest.approx(-a, abs=0.01)


def test_revolution_count_matches_duration(clean_signal):
    """A clean 6000 rpm (100 rev/s) drive over 0.5 s -> ~50 revolutions."""
    assert clean_signal.n_revolutions == pytest.approx(50.0, abs=0.5)


def test_pullback_position_integrates_speed(make_signal):
    """With commanded speed v and no error, final position ~ v * duration."""
    v = 20.0
    dur = 0.5
    sig = make_signal(pullback_mm_s=v, duration_s=dur)
    assert sig.position[-1] == pytest.approx(v * dur, rel=1e-3)


def test_pullback_speed_error_biases_position(make_signal):
    """A +10% speed error -> position end is ~10% higher than commanded."""
    v, dur, err = 20.0, 0.5, 0.10
    sig = make_signal(pullback_mm_s=v, speed_error_pct=err, duration_s=dur)
    assert sig.position[-1] == pytest.approx(v * dur * (1 + err), rel=1e-3)


def test_nyquist_guard_raises_for_undersampled_harmonic():
    """Generator must refuse a sampling rate too low for the highest harmonic."""
    rot = RotationParams(
        rpm=6000.0, nurd_harmonics=[NurdHarmonic(k=10, amplitude=0.1)]
    )
    # f_rot=100 Hz, 10x = 1000 Hz; need fs >= 4000. 2000 should raise.
    with pytest.raises(ValueError, match="too low to resolve"):
        generate_drive_signal(0.1, 2000.0, rotation=rot)


def test_invalid_harmonic_order_rejected():
    with pytest.raises(ValueError):
        NurdHarmonic(k=0, amplitude=0.1)


def test_meta_records_ground_truth(make_signal):
    """The meta dict must faithfully record what was injected."""
    sig = make_signal(nurd=[(2, 0.15)], jitter_pct=0.03, pullback_mm_s=10.0,
                      speed_error_pct=0.05)
    meta = sig.meta
    assert meta["injected_nurd_harmonics"][0] == {"k": 2, "amplitude": 0.15, "phase": 0.0}
    assert meta["injected_jitter_pct"] == 0.03
    assert meta["injected_pullback_speed_error_pct"] == 0.05
    assert meta["pullback_speed_mm_s"] == 10.0


def test_reproducible_with_seed():
    """Same seed -> identical jitter realization."""
    rot = RotationParams(rpm=6000.0, jitter_pct=0.02)
    a = generate_drive_signal(0.2, 50_000.0, rotation=rot, seed=42)
    b = generate_drive_signal(0.2, 50_000.0, rotation=rot, seed=42)
    assert np.array_equal(a.omega, b.omega)
