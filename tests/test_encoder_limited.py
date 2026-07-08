"""NURD correction evaluated under encoder limits instead of ground truth."""

import numpy as np
import pytest

from oct_drive_test.correction import correct_nurd
from oct_drive_test.encoder import simulate_encoder_readout, true_residual_nurd
from oct_drive_test.generator import NurdHarmonic, RotationParams, generate_drive_signal


@pytest.fixture()
def distorted_signal():
    rotation = RotationParams(
        rpm=6000.0,
        nurd_harmonics=[NurdHarmonic(k=1, amplitude=0.20), NurdHarmonic(k=2, amplitude=0.10)],
        jitter_pct=0.005,
    )
    return generate_drive_signal(duration_s=0.05, fs=200_000.0, rotation=rotation, seed=42)


def test_encoder_readout_is_strictly_increasing(distorted_signal):
    sig = distorted_signal
    theta_meas = simulate_encoder_readout(sig.t, sig.theta, counts_per_rev=2048, seed=1)
    assert np.all(np.diff(theta_meas) > 0)
    # Estimate stays within a few counts of the truth (edge interpolation
    # plus timing jitter; ends extrapolate flat).
    step = 2 * np.pi / 2048
    interior = slice(100, -100)
    assert np.max(np.abs(theta_meas[interior] - sig.theta[interior])) < 5 * step


def test_encoder_limited_correction_leaves_real_residual(distorted_signal):
    sig = distorted_signal
    theta_meas = simulate_encoder_readout(sig.t, sig.theta, counts_per_rev=2048, seed=1)

    corr = correct_nurd(sig.t, theta_meas, n_per_rev=512)
    residual = true_residual_nurd(sig.t, sig.theta, corr)
    before = corr.nurd_index_before

    # The residual is genuinely non-zero (no ground-truth tautology) ...
    assert residual > 1e-4
    # ... but the correction still removes the overwhelming bulk of the NURD.
    assert residual < 0.25 * before


def test_residual_grows_with_edge_timing_jitter(distorted_signal):
    """At OCT spin rates the correction limit is set by edge-timing jitter."""
    sig = distorted_signal
    residuals = []
    for jitter in (0.0, 1e-6, 4e-6):
        theta_meas = simulate_encoder_readout(
            sig.t, sig.theta, counts_per_rev=2048,
            edge_jitter_s_rms=jitter, seed=1)
        corr = correct_nurd(sig.t, theta_meas, n_per_rev=512)
        residuals.append(true_residual_nurd(sig.t, sig.theta, corr))
    assert residuals[0] < residuals[1] < residuals[2]


def test_ground_truth_correction_is_lower_bound(distorted_signal):
    sig = distorted_signal
    corr_truth = correct_nurd(sig.t, sig.theta, n_per_rev=512)
    truth_residual = true_residual_nurd(sig.t, sig.theta, corr_truth)

    theta_meas = simulate_encoder_readout(sig.t, sig.theta, counts_per_rev=2048, seed=1)
    corr_enc = correct_nurd(sig.t, theta_meas, n_per_rev=512)
    enc_residual = true_residual_nurd(sig.t, sig.theta, corr_enc)

    assert truth_residual < enc_residual
