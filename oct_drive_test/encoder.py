"""Encoder-limited angle measurement.

A real drive never hands the correction algorithm the true fiber angle -- it
hands it an encoder readout: quantized to the disc's line count, corrupted by
sub-count electrical jitter, and decoded monotonically. This module models
that readout so the NURD correction can be evaluated under realistic encoder
limits instead of against the generator's ground truth.
"""
from __future__ import annotations

import numpy as np

__all__ = ["simulate_encoder_readout", "true_residual_nurd"]


def simulate_encoder_readout(
    t: np.ndarray,
    theta: np.ndarray,
    counts_per_rev: int = 2048,
    edge_jitter_s_rms: float = 0.5e-6,
    seed: int | None = None,
) -> np.ndarray:
    """Encoder-limited estimate of a strictly increasing angle series.

    Models the standard edge-timestamp decode: the encoder emits an edge each
    time the shaft crosses a count boundary, the acquisition timestamps that
    edge (with electrical jitter), and the angle-vs-time estimate is the
    interpolation through those (timestamp, count) pairs. Resolution is set by
    ``counts_per_rev``; accuracy by the edge-timing jitter.

    Parameters
    ----------
    t:
        Sample times [s], strictly increasing.
    theta:
        True cumulative angle [rad] at those times, strictly increasing.
    counts_per_rev:
        Encoder resolution (quadrature-decoded counts per revolution).
    edge_jitter_s_rms:
        RMS timing jitter on each decoded edge [s].
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    Strictly increasing angle estimate [rad] at the sample times ``t``.
    """
    t = np.asarray(t, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if counts_per_rev < 8:
        raise ValueError("counts_per_rev must be >= 8")
    rng = np.random.default_rng(seed)

    step = 2.0 * np.pi / counts_per_rev
    # Count-boundary angles spanned by the record.
    k_first = np.ceil(theta[0] / step)
    k_last = np.floor(theta[-1] / step)
    boundaries = np.arange(k_first, k_last + 1) * step
    # True crossing time of each boundary (invert the monotone theta(t)),
    # plus electrical timestamp jitter.
    t_edges = np.interp(boundaries, theta, t)
    t_edges = t_edges + rng.normal(0.0, edge_jitter_s_rms, t_edges.shape)
    t_edges = np.maximum.accumulate(t_edges)  # decoder cannot reorder edges

    # Angle estimate at the acquisition sample times.
    theta_est = np.interp(t, t_edges, boundaries)
    # Strict monotonicity for downstream theta(t) inversion.
    theta_est = np.maximum.accumulate(theta_est)
    theta_est += np.arange(t.size) * (step * 1e-9)
    return theta_est


def true_residual_nurd(
    times: np.ndarray,
    theta_true: np.ndarray,
    correction,
) -> float:
    """Residual NURD actually left in a corrected stream.

    ``NurdCorrection.nurd_index_after`` is evaluated on the correction's own
    angular grid, which is uniform by construction -- it measures how uniform
    the *estimated* grid is. This helper instead evaluates the **true** angle
    at the corrected A-line times: if the encoder estimate was imperfect, the
    true angles are not quite equi-spaced and the residual is non-zero. That
    number is the physically meaningful correction limit.
    """
    from .correction import residual_nurd

    theta_at_grid = np.interp(correction.t_grid, times, theta_true)
    return residual_nurd(theta_at_grid)
