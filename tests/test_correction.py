"""Ground-truth validation of NURD correction by angular resampling.

Strategy mirrors the rest of the suite: inject a KNOWN NURD modulation, confirm
the measured NURD index is high, then assert that ``correct_nurd`` collapses the
residual NURD index of the corrected (uniform-angle) stream by a large factor --
while leaving a clean (NURD-free) signal essentially untouched (no injected
distortion) and producing a monotonic, uniformly-spaced angular grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from oct_drive_test import correction, nurd


# ---------------------------------------------------------------------------
# Core: known NURD -> high index -> large residual drop after correction.
# ---------------------------------------------------------------------------
def test_residual_nurd_matches_velocity_index_on_input(make_signal):
    """residual_nurd(theta) == std(omega)/mean(omega): the angle-increment form
    of the NURD index equals the velocity form on the same (uniform-time) stream."""
    sig = make_signal(nurd=[(1, 0.18), (3, 0.06)], jitter_pct=0.0, duration_s=0.5)
    via_theta = correction.residual_nurd(sig.theta)
    via_omega = nurd.nurd_index(sig.omega)
    assert via_theta == pytest.approx(via_omega, rel=1e-3)


def test_correction_collapses_injected_nurd(make_signal):
    """Inject strong NURD -> high index -> residual drops by >10x after correct."""
    sig = make_signal(nurd=[(1, 0.18), (3, 0.06)], jitter_pct=0.0, duration_s=0.5)

    before = nurd.nurd_index(sig.omega)
    assert before > 0.05  # a genuinely large, in-the-fail-zone NURD (~13%)

    corr = correction.correct_drive_signal(sig, n_per_rev=512)

    # The result's own before/after bookkeeping agrees with the metric module.
    assert corr.nurd_index_before == pytest.approx(before, rel=1e-3)

    # Residual collapses far past the 10x bar (resampling makes angle uniform).
    assert corr.nurd_index_after < before / 10.0
    assert corr.improvement_factor > 10.0


def test_correction_factor_scales_with_grid(make_signal):
    """Correction works across several injected amplitudes and grid densities."""
    for a in (0.08, 0.15, 0.25):
        sig = make_signal(nurd=[(1, a)], jitter_pct=0.0, duration_s=0.4)
        corr = correction.correct_drive_signal(sig, n_per_rev=256)
        assert corr.nurd_index_before == pytest.approx(
            a / np.sqrt(2.0), rel=0.03
        )
        # >5x bar comfortably cleared (in practice many orders of magnitude).
        assert corr.improvement_factor > 5.0


def test_correction_with_higher_harmonic(make_signal):
    """A 3x-dominant NURD is also flattened by the angular resampling."""
    sig = make_signal(nurd=[(1, 0.04), (3, 0.20)], jitter_pct=0.0, duration_s=0.5)
    corr = correction.correct_drive_signal(sig, n_per_rev=512)
    assert corr.nurd_index_before > 0.10
    assert corr.improvement_factor > 10.0


# ---------------------------------------------------------------------------
# Clean signal: correction is near-identity, injects no distortion.
# ---------------------------------------------------------------------------
def test_clean_signal_correction_is_near_identity(clean_signal):
    """On a NURD-free drive the residual stays ~0 and no distortion is added."""
    corr = correction.correct_drive_signal(clean_signal, n_per_rev=512)
    # Input is already uniform-angle: index ~ machine floor.
    assert corr.nurd_index_before < 1e-6
    # Correction does NOT inject distortion: residual also ~ machine floor.
    assert corr.nurd_index_after < 1e-6


def test_clean_signal_grid_tracks_input_times(clean_signal):
    """For a constant-rate drive, corrected times advance ~uniformly in time too
    (uniform angle <=> uniform time when omega is constant)."""
    corr = correction.correct_drive_signal(clean_signal, n_per_rev=360)
    dt = np.diff(corr.t_grid)
    # Constant omega -> equal angle steps land at equal time steps.
    assert np.std(dt) / np.mean(dt) < 1e-4


def test_clean_resampled_samples_preserved(clean_signal):
    """A constant per-A-line payload survives resampling unchanged (no ringing)."""
    n = clean_signal.n_samples
    payload = np.full(n, 7.5)
    corr = correction.correct_drive_signal(
        clean_signal, samples=payload, n_per_rev=360
    )
    assert corr.samples is not None
    assert np.allclose(corr.samples, 7.5)


# ---------------------------------------------------------------------------
# Corrected angular grid: monotonic and uniformly spaced.
# ---------------------------------------------------------------------------
def test_corrected_grid_is_monotonic_and_uniform(make_signal):
    """Corrected angle grid is strictly increasing and equi-spaced; times too."""
    sig = make_signal(nurd=[(1, 0.18), (3, 0.06)], jitter_pct=0.0, duration_s=0.5)
    corr = correction.correct_drive_signal(sig, n_per_rev=512)

    grid = corr.theta_grid
    assert np.all(np.diff(grid) > 0)  # strictly increasing angle
    assert np.all(np.diff(corr.t_grid) > 0)  # strictly increasing time

    steps = np.diff(grid)
    # Uniform within tight tolerance (constructed from a fixed dtheta).
    assert np.allclose(steps, corr.dtheta, rtol=1e-9, atol=1e-9)
    assert corr.dtheta == pytest.approx(2.0 * np.pi / 512.0, rel=1e-12)

    # std/mean of the spacing is the residual NURD index -> ~0.
    assert np.std(steps) / np.mean(steps) < 1e-6


def test_n_per_rev_gives_integer_points_per_revolution(make_signal):
    """n_per_rev guarantees exactly that many grid points span each 2*pi."""
    n_per_rev = 256
    sig = make_signal(nurd=[(1, 0.12)], jitter_pct=0.0, duration_s=0.3)
    corr = correction.correct_drive_signal(sig, n_per_rev=n_per_rev)
    # One revolution of angle should contain exactly n_per_rev steps.
    steps_per_rev = (2.0 * np.pi) / corr.dtheta
    assert steps_per_rev == pytest.approx(n_per_rev, rel=1e-9)


def test_n_out_controls_output_length(make_signal):
    """Without n_per_rev, n_out sets the output A-line count over the full span."""
    sig = make_signal(nurd=[(1, 0.10)], jitter_pct=0.0, duration_s=0.3)
    corr = correction.correct_nurd(sig.t, sig.theta, n_out=1000)
    assert corr.n_out == 1000
    assert np.all(np.diff(corr.theta_grid) > 0)
    # Still flattens the NURD.
    assert corr.improvement_factor > 5.0


# ---------------------------------------------------------------------------
# Resampling payload + provenance.
# ---------------------------------------------------------------------------
def test_resampled_samples_shape_and_index_mapping(make_signal):
    """2-D samples (A-line x depth) resample along the A-line axis; indices map."""
    sig = make_signal(nurd=[(1, 0.15)], jitter_pct=0.0, duration_s=0.2)
    n = sig.n_samples
    depth = 8
    payload = np.random.default_rng(0).standard_normal((n, depth))
    corr = correction.correct_nurd(
        sig.t, sig.theta, samples=payload, n_per_rev=128
    )
    assert corr.samples.shape == (corr.n_out, depth)
    # Fractional A-line indices stay within the original sample range, monotone.
    assert corr.aline_index.min() >= 0.0
    assert corr.aline_index.max() <= n - 1
    assert np.all(np.diff(corr.aline_index) > 0)


def test_correct_nurd_input_validation():
    """Guard rails: shape mismatch and non-monotone angle are rejected."""
    t = np.linspace(0, 1, 100)
    theta = np.linspace(0, 10, 100)
    with pytest.raises(ValueError):
        correction.correct_nurd(t, theta[:-1])  # length mismatch
    with pytest.raises(ValueError):
        correction.correct_nurd(t, theta, samples=np.zeros(99))  # bad payload len
    bad_theta = theta.copy()
    bad_theta[50] = bad_theta[49]  # non-strictly-increasing
    with pytest.raises(ValueError):
        correction.correct_nurd(t, bad_theta)
