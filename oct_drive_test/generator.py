"""Synthetic OCT drive-signal generator (ground-truth instrument).

Builds a :class:`~oct_drive_test._signal.DriveSignal` with analytically known
ground truth so that the NURD / stability / pullback metrics can be *validated*:
inject a known distortion, then assert the metric recovers it.

Rotation model
--------------
The instantaneous angular velocity of the fiber core is modeled as a nominal
rate modulated by a sum of NURD harmonics plus random period-to-period jitter::

    omega(t) = omega0 * (1 + sum_k a_k * sin(2*pi*k*f_rot*t + phi_k)) * (1 + j(t))

where

- ``omega0 = 2*pi*f_rot`` is the nominal angular rate [rad/s],
- ``f_rot = rpm/60`` is the rotation frequency [Hz],
- ``a_k`` is the fractional amplitude of the ``k``-th NURD harmonic (k=1..~10),
- ``phi_k`` is its phase [rad],
- ``j(t)`` is slowly-varying random rotational jitter (fractional).

NURD physically arises from torque-cable friction/binding that periodically
speeds up and slows down the distal optics within each revolution; the 1x term
dominates a simple bind, higher harmonics appear with multi-lobe friction.

The measured (unwrapped) angle is the exact time-integral of ``omega(t)`` so the
generator is self-consistent: theta is the cumulative integral of omega, and the
commanded angle ``theta_cmd`` advances at the constant nominal rate.

Pullback model
--------------
The linear pullback position advances at a commanded speed with optional
constant speed error (calibration bias), sinusoidal ripple, and random noise::

    v(t) = v_cmd * (1 + speed_error_frac) * (1 + ripple) + noise
    p(t) = integral of v(t)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NurdHarmonic:
    """One periodic NURD component at integer multiple ``k`` of rotation freq.

    Parameters
    ----------
    k:
        Harmonic order relative to the rotation frequency (1 = once-per-rev).
    amplitude:
        Fractional velocity-modulation amplitude ``a_k`` (e.g. 0.10 = +/-10%).
    phase:
        Phase offset [rad].
    """

    k: int
    amplitude: float
    phase: float = 0.0

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("NURD harmonic order k must be >= 1")
        if self.amplitude < 0:
            raise ValueError("NURD amplitude must be >= 0")


@dataclass
class RotationParams:
    """Parameters of the proximal rotary drive.

    Parameters
    ----------
    rpm:
        Nominal rotation rate [rev/min]. A typical OCT drive spins ~100 rev/s
        (6000 rpm); the default here is 6000 rpm = 100 Hz.
    nurd_harmonics:
        Periodic angular-velocity modulations injected as NURD.
    jitter_pct:
        RMS of the random period-to-period rotational jitter, as a fraction of
        the nominal rate (e.g. 0.01 = 1% RMS). Modeled as low-pass band-limited
        noise so that it perturbs the rate from revolution to revolution.
    jitter_corner_hz:
        Corner frequency [Hz] of the jitter's low-pass character. Defaults to the
        rotation frequency so jitter is a per-revolution effect, not white noise.
    """

    rpm: float = 6000.0
    nurd_harmonics: list[NurdHarmonic] = field(default_factory=list)
    jitter_pct: float = 0.0
    jitter_corner_hz: float | None = None

    @property
    def f_rot(self) -> float:
        return self.rpm / 60.0

    @property
    def omega0(self) -> float:
        return 2.0 * np.pi * self.f_rot


@dataclass
class PullbackParams:
    """Parameters of the linear pullback stage.

    Parameters
    ----------
    speed_mm_s:
        Commanded pullback speed [mm/s]. Set to 0 for a stationary (spin-only)
        acquisition.
    speed_error_pct:
        Constant fractional speed error / calibration bias (e.g. 0.05 = stage
        actually moves 5% faster than commanded).
    ripple_pct:
        Fractional amplitude of sinusoidal speed ripple.
    ripple_hz:
        Frequency [Hz] of the speed ripple (e.g. lead-screw / belt periodicity).
    noise_pct:
        RMS of white speed noise, as a fraction of commanded speed.
    """

    speed_mm_s: float = 0.0
    speed_error_pct: float = 0.0
    ripple_pct: float = 0.0
    ripple_hz: float = 0.0
    noise_pct: float = 0.0


def _bandlimited_noise(
    n: int, fs: float, corner_hz: float, rng: np.random.Generator
) -> np.ndarray:
    """Unit-RMS, zero-mean noise low-passed to ``corner_hz`` via FFT brick wall.

    Returns an array normalized to unit standard deviation so the caller can
    scale it directly by a desired fractional RMS.
    """
    white = rng.standard_normal(n)
    if corner_hz is None or corner_hz <= 0 or corner_hz >= fs / 2.0:
        out = white
    else:
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spec = np.fft.rfft(white)
        spec[freqs > corner_hz] = 0.0
        out = np.fft.irfft(spec, n=n)
    out = out - np.mean(out)
    std = np.std(out)
    if std > 0:
        out = out / std
    return out


def generate_drive_signal(
    duration_s: float,
    fs: float,
    rotation: RotationParams | None = None,
    pullback: PullbackParams | None = None,
    seed: int | None = None,
):
    """Generate a synthetic OCT drive signal with known ground truth.

    Parameters
    ----------
    duration_s:
        Record length [s].
    fs:
        Sampling frequency [Hz]. Must be high enough to resolve the highest NURD
        harmonic (>= a few x ``k_max * f_rot``); a guard assertion enforces this.
    rotation, pullback:
        Drive parameters. Defaults give a clean 6000 rpm spin with no pullback.
    seed:
        RNG seed for reproducible jitter/noise.

    Returns
    -------
    DriveSignal
        With ``meta`` recording every injected ground-truth quantity.
    """
    from ._signal import DriveSignal

    if rotation is None:
        rotation = RotationParams()
    if pullback is None:
        pullback = PullbackParams()
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if fs <= 0:
        raise ValueError("fs must be > 0")

    rng = np.random.default_rng(seed)

    n = int(round(duration_s * fs))
    if n < 4:
        raise ValueError("duration too short for the sampling rate")
    t = np.arange(n, dtype=float) / fs

    f_rot = rotation.f_rot
    omega0 = rotation.omega0

    # --- Nyquist guard: highest harmonic must be well within Nyquist. ---
    k_max = max((h.k for h in rotation.nurd_harmonics), default=1)
    highest_freq = k_max * f_rot
    if highest_freq > 0 and fs < 4.0 * highest_freq:
        raise ValueError(
            f"fs={fs} Hz too low to resolve harmonic {k_max}x at f_rot={f_rot} "
            f"Hz (need fs >= {4.0 * highest_freq:.1f} Hz)"
        )

    # --- Deterministic NURD modulation: fractional velocity perturbation. ---
    nurd_mod = np.zeros(n, dtype=float)
    for h in rotation.nurd_harmonics:
        nurd_mod += h.amplitude * np.sin(2.0 * np.pi * h.k * f_rot * t + h.phase)

    # --- Random rotational jitter (band-limited, unit-RMS * jitter_pct). ---
    if rotation.jitter_pct > 0:
        corner = rotation.jitter_corner_hz
        if corner is None:
            corner = f_rot  # per-revolution jitter by default
        jitter = rotation.jitter_pct * _bandlimited_noise(n, fs, corner, rng)
    else:
        jitter = np.zeros(n, dtype=float)

    # Instantaneous angular velocity [rad/s].
    omega = omega0 * (1.0 + nurd_mod) * (1.0 + jitter)

    # Measured angle = exact cumulative integral of omega (trapezoid), so the
    # generator is internally self-consistent (theta' == omega).
    theta = _cumulative_integral(omega, t)
    theta_cmd = omega0 * t  # ideal constant-rate command

    # --- Pullback channel ---
    v_cmd = pullback.speed_mm_s
    if v_cmd != 0.0:
        v = v_cmd * (1.0 + pullback.speed_error_pct)
        if pullback.ripple_pct > 0 and pullback.ripple_hz > 0:
            v = v * (
                1.0
                + pullback.ripple_pct
                * np.sin(2.0 * np.pi * pullback.ripple_hz * t)
            )
        else:
            v = np.full(n, v, dtype=float)
        if pullback.noise_pct > 0:
            v = v + (pullback.noise_pct * abs(v_cmd)) * rng.standard_normal(n)
        position = _cumulative_integral(v, t)
    else:
        position = np.zeros(n, dtype=float)

    meta = {
        "rpm_nominal": rotation.rpm,
        "f_rot_nominal_hz": f_rot,
        "omega0_rad_s": omega0,
        "injected_nurd_harmonics": [
            {"k": h.k, "amplitude": h.amplitude, "phase": h.phase}
            for h in rotation.nurd_harmonics
        ],
        "injected_jitter_pct": rotation.jitter_pct,
        "pullback_speed_mm_s": v_cmd,
        "injected_pullback_speed_error_pct": pullback.speed_error_pct,
        "injected_pullback_ripple_pct": pullback.ripple_pct,
        "injected_pullback_noise_pct": pullback.noise_pct,
        "fs_hz": fs,
        "duration_s": duration_s,
        "seed": seed,
    }

    return DriveSignal(
        t=t,
        theta=theta,
        omega=omega,
        position=position,
        fs=fs,
        rpm_nominal=rotation.rpm,
        pullback_speed_nominal=v_cmd,
        theta_cmd=theta_cmd,
        meta=meta,
    )


def _cumulative_integral(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of ``y`` over ``t``, starting at 0."""
    out = np.zeros_like(y)
    if y.shape[0] < 2:
        return out
    dt = np.diff(t)
    increments = 0.5 * (y[1:] + y[:-1]) * dt
    out[1:] = np.cumsum(increments)
    return out
