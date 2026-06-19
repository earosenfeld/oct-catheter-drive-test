"""Minimal CLI: generate a synthetic drive signal and emit a spec report.

Usage::

    oct-drive-report --spec spec.yaml --out report.md [--nurd 0.0] [--seed 0]

This is primarily a convenience wrapper around the library for demos and CI; the
real entry points are the library functions in :mod:`oct_drive_test`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .generator import (
    NurdHarmonic,
    PullbackParams,
    RotationParams,
    generate_drive_signal,
)
from .spec import DriveSpec, characterize, render_markdown, write_report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate a synthetic OCT drive signal and stamp a "
        "characterization report against a spec (engineering V&V only)."
    )
    p.add_argument("--spec", required=True, help="Path to spec.yaml")
    p.add_argument("--out", default=None, help="Report output path (.md/.html)")
    p.add_argument(
        "--nurd",
        type=float,
        default=0.0,
        help="Injected 1x NURD fractional amplitude (e.g. 0.1 = 10%%)",
    )
    p.add_argument(
        "--jitter", type=float, default=0.0, help="Injected jitter fraction RMS"
    )
    p.add_argument(
        "--pullback-mm-s", type=float, default=0.0, help="Pullback speed mm/s"
    )
    p.add_argument("--duration", type=float, default=1.0, help="Seconds")
    p.add_argument("--fs", type=float, default=50_000.0, help="Sample rate Hz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    spec = DriveSpec.from_yaml(args.spec)
    rot = RotationParams(
        rpm=spec.rpm_nominal,
        nurd_harmonics=(
            [NurdHarmonic(k=1, amplitude=args.nurd)] if args.nurd > 0 else []
        ),
        jitter_pct=args.jitter,
    )
    pb = PullbackParams(speed_mm_s=args.pullback_mm_s)
    sig = generate_drive_signal(
        duration_s=args.duration,
        fs=args.fs,
        rotation=rot,
        pullback=pb,
        seed=args.seed,
    )
    result = characterize(sig, spec)
    print(render_markdown(result))
    if args.out:
        out = write_report(result, Path(args.out))
        print(f"\n[written: {out}]")
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
