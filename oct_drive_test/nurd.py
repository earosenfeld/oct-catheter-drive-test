"""NURD (Non-Uniform Rotational Distortion) metrics.

NURD is the dominant image-quality killer in intravascular OCT: torque-cable
friction/binding makes the distal optics speed up and slow down within each
revolution, smearing/compressing image features azimuthally. These functions
quantify it from the measured angular-velocity record ``omega(t)``.

Definitions implemented
-----------------------
1. Primary NURD index (per rotation and overall)::

       NURD_index = std(omega) / mean(omega)            [dimensionless, x100 -> %]

   Computed over a window; ``nurd_index_per_rotation`` slices the record into
   individual revolutions (by unwrapped angle) and reports the index of each.

2. Spectral NURD::

       Take FFT of omega(t); express harmonic magnitude at k*f_rot relative to
       the DC (mean) component:  H_k = |OMEGA(k*f_rot)| / |OMEGA(0)|.

   The dominant NURD harmonic is the k>=1 with the largest H_k. For a pure
   sinusoidal modulation of fractional amplitude a_k, H_k recovers a_k/2 (the
   one-sided spectral amplitude of a sine is half its peak-to-... amplitude),
   and the overall std/mean index equals a_k/sqrt(2). Both relations are used
   by the validation tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._signal import DriveSignal


@dataclass
class SpectralNurd:
    """Result of the spectral NURD analysis."""

    harmonic_orders: np.ndarray  # k values (>=1)
    harmonic_ratios: np.ndarray  # |OMEGA(k f_rot)| / |OMEGA(0)|
    dominant_k: int  # harmonic order with largest ratio
    dominant_ratio: float  # that largest ratio
    freqs: np.ndarray  # full one-sided frequency axis [Hz]
    magnitude: np.ndarray  # full one-sided magnitude spectrum of omega
    f_rot: float  # rotation frequency used [Hz]


def nurd_index(omega: np.ndarray) -> float:
    """Primary NURD index = std(omega) / mean(omega) (dimensionless).

    Multiply by 100 for a percentage. Uses the population std (ddof=0).
    """
    omega = np.asarray(omega, dtype=float)
    if omega.size == 0:
        return float("nan")
    mean = float(np.mean(omega))
    if mean == 0.0:
        return float("nan")
    return float(np.std(omega) / mean)


def nurd_index_pct(omega: np.ndarray) -> float:
    """Primary NURD index expressed as a percentage."""
    return 100.0 * nurd_index(omega)


def _rotation_boundaries(theta: np.ndarray) -> np.ndarray:
    """Sample indices where the unwrapped angle crosses each 2*pi multiple.

    Returns the indices marking the start of each completed revolution, i.e. the
    first sample whose cumulative angle has advanced past the next 2*pi boundary.
    """
    theta = np.asarray(theta, dtype=float)
    theta0 = theta[0]
    # Revolution number reached at each sample.
    rev = np.floor((theta - theta0) / (2.0 * np.pi)).astype(int)
    # Boundary where rev increments.
    changes = np.nonzero(np.diff(rev) > 0)[0] + 1
    return changes


def nurd_index_per_rotation(signal: DriveSignal) -> dict:
    """Per-rotation NURD map: NURD index for each completed revolution.

    Returns
    -------
    dict with keys:
        rotation_number : np.ndarray of int (0-based)
        nurd_index      : np.ndarray, std/mean of omega within that revolution
        nurd_index_pct  : np.ndarray, same x100
        start_index     : np.ndarray, sample index where each revolution starts
    """
    bounds = _rotation_boundaries(signal.theta)
    if bounds.size < 1:
        return {
            "rotation_number": np.array([], dtype=int),
            "nurd_index": np.array([], dtype=float),
            "nurd_index_pct": np.array([], dtype=float),
            "start_index": np.array([], dtype=int),
        }

    edges = np.concatenate(([0], bounds, [signal.n_samples]))
    idx = []
    vals = []
    starts = []
    rot_no = 0
    for a, b in zip(edges[:-1], edges[1:]):
        seg = signal.omega[a:b]
        if seg.size >= 2:
            vals.append(nurd_index(seg))
            idx.append(rot_no)
            starts.append(int(a))
            rot_no += 1
    vals_arr = np.asarray(vals, dtype=float)
    return {
        "rotation_number": np.asarray(idx, dtype=int),
        "nurd_index": vals_arr,
        "nurd_index_pct": 100.0 * vals_arr,
        "start_index": np.asarray(starts, dtype=int),
    }


def spectral_nurd(
    signal: DriveSignal,
    f_rot: float | None = None,
    k_max: int = 10,
) -> SpectralNurd:
    """Spectral NURD: harmonic content of omega(t) at multiples of f_rot.

    Parameters
    ----------
    f_rot:
        Rotation frequency [Hz]. Defaults to the signal's nominal value.
    k_max:
        Highest harmonic order to report (1..k_max).

    The harmonic magnitude at ``k*f_rot`` is taken from the nearest FFT bin and
    normalized by the DC magnitude to yield a dimensionless ratio. A Hann window
    reduces spectral leakage; the DC and harmonic magnitudes are both measured on
    the windowed spectrum so the ratio is consistent.
    """
    if f_rot is None:
        f_rot = signal.f_rot_nominal
    omega = signal.omega
    n = omega.size
    fs = signal.fs

    # Hann window to control leakage when the record is not an integer #rev.
    win = np.hanning(n)
    win_sum = np.sum(win)
    spec = np.fft.rfft(omega * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(spec)

    # DC magnitude == sum(omega*win); normalize to mean amplitude.
    dc = mag[0]
    if dc == 0:
        dc = np.finfo(float).eps

    orders = np.arange(1, k_max + 1)
    ratios = np.zeros(orders.size, dtype=float)
    nyq = fs / 2.0
    for i, k in enumerate(orders):
        target = k * f_rot
        if target >= nyq:
            ratios[i] = np.nan
            continue
        bin_idx = int(np.argmin(np.abs(freqs - target)))
        # One-sided amplitude: for a real sine of fractional amp a_k, the
        # windowed single-bin magnitude / DC magnitude recovers a_k / 2.
        ratios[i] = mag[bin_idx] / dc

    valid = ~np.isnan(ratios)
    if np.any(valid):
        dom_local = int(np.nanargmax(ratios))
        dominant_k = int(orders[dom_local])
        dominant_ratio = float(ratios[dom_local])
    else:
        dominant_k = 0
        dominant_ratio = float("nan")

    # Convert magnitude to mean-referenced amplitude spectrum for plotting.
    amp_spectrum = mag / (win_sum / 2.0)
    amp_spectrum[0] = mag[0] / win_sum  # DC -> mean

    return SpectralNurd(
        harmonic_orders=orders,
        harmonic_ratios=ratios,
        dominant_k=dominant_k,
        dominant_ratio=dominant_ratio,
        freqs=freqs,
        magnitude=amp_spectrum,
        f_rot=f_rot,
    )


def nurd_summary(signal: DriveSignal, k_max: int = 10) -> dict:
    """Convenience: overall index + per-rotation stats + dominant harmonic."""
    per_rot = nurd_index_per_rotation(signal)
    spec = spectral_nurd(signal, k_max=k_max)
    pr = per_rot["nurd_index_pct"]
    return {
        "nurd_index_pct_overall": nurd_index_pct(signal.omega),
        "nurd_index_pct_per_rotation_mean": (
            float(np.mean(pr)) if pr.size else float("nan")
        ),
        "nurd_index_pct_per_rotation_max": (
            float(np.max(pr)) if pr.size else float("nan")
        ),
        "n_rotations": int(pr.size),
        "dominant_harmonic_k": spec.dominant_k,
        "dominant_harmonic_ratio": spec.dominant_ratio,
        "harmonic_orders": spec.harmonic_orders.tolist(),
        "harmonic_ratios": spec.harmonic_ratios.tolist(),
    }
