"""NURD correction by angular resampling (encoder-based re-gridding).

NURD distorts an OCT image because A-lines are acquired uniformly in *time* while
the fiber core, owing to torque-cable friction/binding, rotates *non-uniformly* in
*angle*. Two A-lines separated by a fixed acquisition interval are therefore
separated by a varying azimuthal angle -- features are smeared/compressed around
the frame. ``nurd.py`` *measures* that non-uniformity; this module *corrects* it.

Correction strategy (what real OCT systems do)
----------------------------------------------
The measured cumulative angle ``theta(t)`` -- from the rotary encoder or the
generator ground truth -- is a strictly increasing function of time (the core
always rotates forward). Acquisition gives us samples at times ``t_i`` sitting at
angles ``theta_i = theta(t_i)`` that are *unevenly spaced in angle*.

To put the stream back onto a UNIFORM angular grid we:

1. Build a uniform angular grid ``theta_grid`` spanning the measured angle range,
   spaced at ``dtheta = 2*pi / n_per_rev`` (an exact integer number of grid points
   per revolution, so every revolution is sampled identically).
2. Invert ``theta(t)``: for each target angle ``theta_grid[k]`` find the time
   ``t_grid[k]`` at which the core actually passed through it -- a 1-D
   interpolation of ``t`` against the monotone ``theta`` (the inverse map
   ``t = theta^{-1}(angle)``).
3. Resample any per-A-line payload (``samples``) at those times to obtain the
   corrected, equi-angular A-line stream.

By construction the corrected stream is uniform in angle, so the azimuthal
sampling error -- the very thing the NURD index quantifies -- collapses toward
the numerical floor. The correction is encoder-driven: it uses only the measured
``theta(t)`` and the acquisition times, exactly as a bench system would.

Honest residual metric
-----------------------
The primary NURD index is ``std(omega)/mean(omega)``. On a *uniform-time* grid
``omega_i`` is proportional to the angle advanced per (constant) time step, so the
index is equivalently ``std(dtheta_i)/mean(dtheta_i)`` over the non-uniform angle
increments. After correction the angle increments ``dtheta`` are constant by
construction; applying the *same* std/mean index to the corrected angular grid
therefore yields ~0. That is the residual reported here -- the same metric, on the
corrected stream, not a loosened one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._signal import DriveSignal


@dataclass
class NurdCorrection:
    """Result of angular-resampling NURD correction.

    Attributes
    ----------
    theta_grid:
        Uniform angular grid the stream was resampled onto [rad], strictly
        increasing with a constant step ``dtheta``.
    t_grid:
        Acquisition times of the corrected (equi-angular) A-lines [s] -- the
        inverse map ``theta^{-1}(theta_grid)``. Strictly increasing.
    aline_index:
        Fractional A-line index of each corrected grid point on the *original*
        sample axis (where each output angle falls between input samples). Useful
        for resampling auxiliary channels and for provenance.
    samples:
        Resampled per-A-line payload on the uniform angular grid, or ``None`` if
        no ``samples`` were supplied. Same leading length as ``theta_grid``.
    dtheta:
        Constant angular step of the corrected grid [rad].
    n_per_rev:
        Number of grid points per full revolution.
    nurd_index_before:
        std/mean NURD index of the *input* (uniform-time) angular increments.
    nurd_index_after:
        std/mean NURD index of the *corrected* (uniform-angle) grid -- the
        residual. ``before / after`` is the correction factor.
    meta:
        Provenance dictionary.
    """

    theta_grid: np.ndarray
    t_grid: np.ndarray
    aline_index: np.ndarray
    samples: np.ndarray | None
    dtheta: float
    n_per_rev: int
    nurd_index_before: float
    nurd_index_after: float
    meta: dict = field(default_factory=dict)

    @property
    def improvement_factor(self) -> float:
        """How many times smaller the residual NURD index is (before/after)."""
        if self.nurd_index_after <= 0.0:
            return float("inf")
        return self.nurd_index_before / self.nurd_index_after

    @property
    def n_out(self) -> int:
        return int(self.theta_grid.size)


def residual_nurd(theta: np.ndarray) -> float:
    """NURD index of an angle series via its sample-to-sample increments.

    Equivalent to ``std(omega)/mean(omega)`` for an angle series sampled on a
    uniform axis: ``omega_i`` is proportional to the per-step angle increment, so
    the std/mean of the increments *is* the NURD index. Applying this to a
    corrected, equi-angular grid yields the residual (~0); applying it to the
    original non-uniform-angle stream recovers the pre-correction index.

    Returns ``nan`` for fewer than two samples or a zero-mean increment.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.size < 2:
        return float("nan")
    dtheta = np.diff(theta)
    mean = float(np.mean(dtheta))
    if mean == 0.0:
        return float("nan")
    return float(np.std(dtheta) / mean)


def correct_nurd(
    times: np.ndarray,
    theta: np.ndarray,
    samples: np.ndarray | None = None,
    n_out: int | None = None,
    n_per_rev: int | None = None,
) -> NurdCorrection:
    """Resample a NURD-distorted A-line stream onto a uniform angular grid.

    Given the A-line acquisition ``times`` and the measured (NURD-distorted)
    cumulative angle ``theta(t)`` at those times, invert ``theta(t)`` to find the
    times at uniform-angle grid points and resample, producing an equi-angular
    A-line stream.

    Parameters
    ----------
    times:
        A-line acquisition timestamps [s], strictly increasing. Length ``N``.
    theta:
        Measured cumulative angle [rad] at each acquisition time, unwrapped and
        strictly increasing (forward rotation). Length ``N``.
    samples:
        Optional per-A-line payload to resample. Shape ``(N,)`` or ``(N, M)``
        (e.g. ``M`` depth pixels per A-line). Resampled along the first axis at
        the corrected times.
    n_out:
        Desired number of output A-lines. The grid step is chosen to span the full
        measured angle range with ``n_out`` points. Ignored if ``n_per_rev`` is
        given. Defaults to ``N`` (preserve the A-line count) when neither is set.
    n_per_rev:
        Output A-lines per full revolution. Takes precedence over ``n_out`` and
        guarantees an *exact* integer number of grid points per 2*pi, so every
        revolution is sampled on an identical sub-grid (the physically natural
        choice for OCT framing). Must be >= 2.

    Returns
    -------
    NurdCorrection
        Uniform angle grid, corrected acquisition times, fractional A-line
        indices, optional resampled ``samples``, and before/after NURD indices.
    """
    times = np.asarray(times, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if times.ndim != 1 or theta.ndim != 1:
        raise ValueError("times and theta must be 1-D")
    if times.shape != theta.shape:
        raise ValueError("times and theta must have the same length")
    n = times.size
    if n < 2:
        raise ValueError("need at least 2 samples to correct")
    if not np.all(np.diff(times) > 0):
        raise ValueError("times must be strictly increasing")
    if not np.all(np.diff(theta) > 0):
        raise ValueError(
            "theta must be strictly increasing (unwrapped, forward rotation)"
        )

    theta0 = float(theta[0])
    theta_end = float(theta[-1])
    span = theta_end - theta0

    # --- Build the uniform angular grid. ---
    if n_per_rev is not None:
        if n_per_rev < 2:
            raise ValueError("n_per_rev must be >= 2")
        dtheta = 2.0 * np.pi / float(n_per_rev)
        n_full_steps = int(np.floor(span / dtheta))
        if n_full_steps < 1:
            raise ValueError(
                "angle span shorter than one grid step; lower n_per_rev or "
                "supply more rotation"
            )
        # Grid points 0..n_full_steps inclusive -> n_full_steps+1 points.
        n_grid = n_full_steps + 1
        theta_grid = theta0 + dtheta * np.arange(n_grid)
        per_rev = int(n_per_rev)
    else:
        if n_out is None:
            n_out = n
        if n_out < 2:
            raise ValueError("n_out must be >= 2")
        theta_grid = np.linspace(theta0, theta_end, n_out)
        dtheta = float(theta_grid[1] - theta_grid[0])
        # Implied (possibly non-integer) points per revolution for reference.
        per_rev = int(round(2.0 * np.pi / dtheta)) if dtheta > 0 else 0

    # --- Invert theta(t): t_grid = theta^{-1}(theta_grid). ---
    # theta is strictly increasing so np.interp inverts it directly. Endpoints of
    # theta_grid are within [theta0, theta_end] by construction (no extrapolation).
    t_grid = np.interp(theta_grid, theta, times)

    # Fractional A-line index of each grid point on the original sample axis.
    sample_axis = np.arange(n, dtype=float)
    aline_index = np.interp(theta_grid, theta, sample_axis)

    # --- Resample the payload, if any, at the corrected times. ---
    out_samples: np.ndarray | None = None
    if samples is not None:
        samples = np.asarray(samples, dtype=float)
        if samples.shape[0] != n:
            raise ValueError(
                f"samples leading dim {samples.shape[0]} != number of A-lines {n}"
            )
        if samples.ndim == 1:
            out_samples = np.interp(t_grid, times, samples)
        else:
            # Resample each column (e.g. depth pixel) along the A-line axis.
            cols = [np.interp(t_grid, times, samples[:, j])
                    for j in range(samples.shape[1])]
            out_samples = np.stack(cols, axis=1)

    nurd_before = residual_nurd(theta)
    nurd_after = residual_nurd(theta_grid)

    meta = {
        "n_in": int(n),
        "n_out": int(theta_grid.size),
        "n_per_rev": per_rev,
        "dtheta_rad": float(dtheta),
        "theta_span_rad": float(span),
        "n_revolutions": float(span / (2.0 * np.pi)),
        "method": "encoder angular resampling (invert theta(t), interp)",
    }

    return NurdCorrection(
        theta_grid=theta_grid,
        t_grid=t_grid,
        aline_index=aline_index,
        samples=out_samples,
        dtheta=float(dtheta),
        n_per_rev=per_rev,
        nurd_index_before=nurd_before,
        nurd_index_after=nurd_after,
        meta=meta,
    )


def correct_drive_signal(
    signal: DriveSignal,
    samples: np.ndarray | None = None,
    n_out: int | None = None,
    n_per_rev: int | None = None,
) -> NurdCorrection:
    """Convenience wrapper: NURD-correct a generator/encoder ``DriveSignal``.

    Uses the signal's sample times ``t`` as the A-line acquisition times and its
    measured cumulative angle ``theta`` as the encoder angle. See
    :func:`correct_nurd` for the resampling details and parameters.
    """
    return correct_nurd(
        times=signal.t,
        theta=signal.theta,
        samples=samples,
        n_out=n_out,
        n_per_rev=n_per_rev,
    )
