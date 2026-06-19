"""Shared pytest fixtures and path setup for the OCT drive-test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the package importable when running tests without an editable install.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oct_drive_test.generator import (  # noqa: E402
    NurdHarmonic,
    PullbackParams,
    RotationParams,
    generate_drive_signal,
)


@pytest.fixture
def clean_signal():
    """A near-ideal 6000 rpm drive: no NURD, no jitter, no pullback."""
    return generate_drive_signal(
        duration_s=0.5,
        fs=50_000.0,
        rotation=RotationParams(rpm=6000.0),
        seed=0,
    )


@pytest.fixture
def make_signal():
    """Factory to build signals with arbitrary injected parameters."""

    def _make(
        rpm=6000.0,
        nurd=None,
        jitter_pct=0.0,
        pullback_mm_s=0.0,
        speed_error_pct=0.0,
        duration_s=0.5,
        fs=50_000.0,
        seed=0,
        **pb_kwargs,
    ):
        harmonics = []
        if nurd:
            for k, amp in nurd:
                harmonics.append(NurdHarmonic(k=k, amplitude=amp))
        rot = RotationParams(
            rpm=rpm, nurd_harmonics=harmonics, jitter_pct=jitter_pct
        )
        pb = PullbackParams(
            speed_mm_s=pullback_mm_s,
            speed_error_pct=speed_error_pct,
            **pb_kwargs,
        )
        return generate_drive_signal(
            duration_s=duration_s,
            fs=fs,
            rotation=rot,
            pullback=pb,
            seed=seed,
        )

    return _make
