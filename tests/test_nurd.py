"""Ground-truth validation of the NURD metrics.

Strategy: inject a NURD modulation with KNOWN parameters, then assert the
metrics recover it. For a single sinusoidal velocity modulation of fractional
amplitude ``a`` (omega = omega0*(1 + a*sin(...))):

- std(omega)/mean(omega) = a / sqrt(2)              (RMS of a unit sine)
- spectral single-bin ratio |OMEGA(k f)| / |OMEGA(0)| = a / 2

These two analytic relations anchor the tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from oct_drive_test import nurd


def test_zero_nurd_index_is_near_zero(clean_signal):
    """No injected NURD -> index ~ 0 (machine-precision floor)."""
    idx = nurd.nurd_index_pct(clean_signal.omega)
    assert idx < 1e-6


def test_index_recovers_a_over_sqrt2(make_signal):
    """Single 1x harmonic amplitude a -> index == a/sqrt(2) (full record)."""
    for a in (0.05, 0.10, 0.20):
        sig = make_signal(nurd=[(1, a)], jitter_pct=0.0, duration_s=1.0)
        idx = nurd.nurd_index(sig.omega)  # dimensionless
        assert idx == pytest.approx(a / np.sqrt(2.0), rel=0.02)


def test_index_monotonic_in_amplitude(make_signal):
    """Increasing injected NURD -> monotonically increasing measured index."""
    amps = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
    measured = [
        nurd.nurd_index_pct(make_signal(nurd=[(1, a)] if a else None).omega)
        for a in amps
    ]
    # Strictly increasing.
    assert all(b > a for a, b in zip(measured, measured[1:]))
    # And the zero case is ~0.
    assert measured[0] < 1e-6


def test_spectral_ratio_recovers_a_over_2(make_signal):
    """Spectral single-bin ratio of a 1x harmonic -> a/2."""
    a = 0.2
    sig = make_signal(nurd=[(1, a)], jitter_pct=0.0, duration_s=1.0)
    spec = nurd.spectral_nurd(sig, k_max=10)
    assert spec.dominant_k == 1
    assert spec.dominant_ratio == pytest.approx(a / 2.0, rel=0.05)


def test_dominant_harmonic_identified(make_signal):
    """When the 3x harmonic is largest, the analyzer reports k=3 as dominant."""
    sig = make_signal(
        nurd=[(1, 0.04), (3, 0.20), (5, 0.05)], jitter_pct=0.0, duration_s=1.0
    )
    spec = nurd.spectral_nurd(sig, k_max=10)
    assert spec.dominant_k == 3
    # The 3x ratio should clearly exceed the 1x and 5x ratios.
    r = dict(zip(spec.harmonic_orders.tolist(), spec.harmonic_ratios.tolist()))
    assert r[3] > r[1] and r[3] > r[5]


def test_spectral_ratios_scale_with_amplitudes(make_signal):
    """Two harmonics with amplitudes 2:1 -> ratios ~2:1."""
    sig = make_signal(
        nurd=[(1, 0.20), (2, 0.10)], jitter_pct=0.0, duration_s=1.0
    )
    spec = nurd.spectral_nurd(sig, k_max=10)
    r = dict(zip(spec.harmonic_orders.tolist(), spec.harmonic_ratios.tolist()))
    assert r[1] / r[2] == pytest.approx(2.0, rel=0.1)


def test_per_rotation_map_length_and_values(make_signal):
    """Per-rotation NURD map: ~one entry per revolution, each ~ a/sqrt(2)."""
    a = 0.1
    sig = make_signal(nurd=[(1, a)], jitter_pct=0.0, duration_s=0.5)
    per_rot = nurd.nurd_index_per_rotation(sig)
    n_rev = sig.n_revolutions
    # Number of analyzed rotations is within 1 of the true revolution count.
    assert abs(per_rot["nurd_index"].size - round(n_rev)) <= 1
    # Each per-rotation index is close to the analytic a/sqrt(2).
    median_idx = np.median(per_rot["nurd_index"])
    assert median_idx == pytest.approx(a / np.sqrt(2.0), rel=0.1)


def test_per_rotation_map_zero_for_clean(clean_signal):
    per_rot = nurd.nurd_index_per_rotation(clean_signal)
    assert per_rot["nurd_index"].size > 0
    assert np.max(per_rot["nurd_index_pct"]) < 1e-5


def test_nurd_summary_keys(make_signal):
    sig = make_signal(nurd=[(1, 0.1)])
    s = nurd.nurd_summary(sig)
    for key in (
        "nurd_index_pct_overall",
        "nurd_index_pct_per_rotation_mean",
        "dominant_harmonic_k",
        "harmonic_ratios",
    ):
        assert key in s


def test_jitter_raises_index_above_pure_nurd(make_signal):
    """Adding random jitter increases the index beyond the deterministic part."""
    base = nurd.nurd_index_pct(make_signal(nurd=[(1, 0.05)], jitter_pct=0.0).omega)
    noisy = nurd.nurd_index_pct(
        make_signal(nurd=[(1, 0.05)], jitter_pct=0.03).omega
    )
    assert noisy > base
