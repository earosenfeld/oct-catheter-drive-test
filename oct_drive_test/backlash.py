"""Torque-cable wind-up / backlash deadband model (stretch).

A long flexible torque cable transmits rotation from the proximal motor to the
distal optics. On a direction reversal the cable must "wind up" through a lash
deadband before the distal end begins to move -- the distal angle lags the
proximal command by up to the deadband width. For continuous one-direction OCT
spin this is mostly a non-issue, but during oscillatory/servo motion or at
start/stop it produces hysteresis that this model reproduces.

Model
-----
Classic backlash (play) hysteresis: the distal (output) angle ``theta_out``
follows the proximal (input) angle ``theta_in`` only when the input pushes
against an edge of a deadband of total width ``lash``::

    if theta_in > theta_out + lash/2:  theta_out = theta_in - lash/2   (engaged +)
    if theta_in < theta_out - lash/2:  theta_out = theta_in + lash/2   (engaged -)
    otherwise:                         theta_out unchanged             (in deadband)

On a reversal the output holds still until the input has traversed the full
``lash`` width, then re-engages -- the signature backlash "dead zone".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def apply_backlash(theta_in: np.ndarray, lash: float) -> np.ndarray:
    """Apply a backlash (lash deadband) nonlinearity to an input angle series.

    Parameters
    ----------
    theta_in:
        Proximal/input angle [rad] (any units; ``lash`` must match).
    lash:
        Total deadband width [rad]. ``0`` is an ideal rigid coupling.

    Returns
    -------
    np.ndarray
        Output (distal) angle, same shape as ``theta_in``.
    """
    theta_in = np.asarray(theta_in, dtype=float)
    out = np.empty_like(theta_in)
    if theta_in.size == 0:
        return out
    half = lash / 2.0
    # Initialize output centered in the deadband around the first input.
    y = theta_in[0]
    out[0] = y
    for i in range(1, theta_in.size):
        x = theta_in[i]
        if x > y + half:
            y = x - half
        elif x < y - half:
            y = x + half
        # else: within deadband, y holds
        out[i] = y
    return out


@dataclass
class BacklashEstimate:
    """Result of estimating lash from paired input/output angle records."""

    lash_estimate: float  # estimated deadband width [rad]
    n_reversals: int  # number of direction reversals observed in input
    max_lag: float  # max |theta_in - theta_out| [rad]


def estimate_lash(
    theta_in: np.ndarray, theta_out: np.ndarray
) -> BacklashEstimate:
    """Estimate the lash width from a paired input/output angle record.

    At steady engagement the input/output offset sits at +/- lash/2; the total
    deadband is the peak-to-peak of (input - output). Reversal count is the
    number of input-direction sign changes.
    """
    theta_in = np.asarray(theta_in, dtype=float)
    theta_out = np.asarray(theta_out, dtype=float)
    offset = theta_in - theta_out
    lash_est = float(np.max(offset) - np.min(offset))
    d = np.diff(theta_in)
    signs = np.sign(d)
    signs = signs[signs != 0]
    if signs.size >= 2:
        n_rev = int(np.sum(np.diff(signs) != 0))
    else:
        n_rev = 0
    return BacklashEstimate(
        lash_estimate=lash_est,
        n_reversals=n_rev,
        max_lag=float(np.max(np.abs(offset))),
    )
