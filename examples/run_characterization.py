#!/usr/bin/env python3
"""Example: generate synthetic OCT drive signals and emit characterization reports.

Runs two scenarios against the example spec:

1. A clean, in-spec drive  -> expected PASS.
2. A degraded drive with excess NURD + pullback error -> expected FAIL.

Both reports are printed and written to ``examples/output/``. This doubles as the
CI smoke test. It makes NO clinical claims -- bench drive characterization only.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from a fresh clone (no install) as `python examples/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oct_drive_test import nurd, rotation
from oct_drive_test.generator import (
    NurdHarmonic,
    PullbackParams,
    RotationParams,
    generate_drive_signal,
)
from oct_drive_test.spec import (
    DriveSpec,
    characterize,
    render_markdown,
    write_report,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "output"


def build_clean_signal(spec: DriveSpec):
    """A well-behaved drive: tiny NURD, tiny jitter, accurate 20 mm/s pullback."""
    rot = RotationParams(
        rpm=spec.rpm_nominal,
        nurd_harmonics=[NurdHarmonic(k=1, amplitude=0.01)],  # ~0.7% index
        jitter_pct=0.002,
    )
    pb = PullbackParams(speed_mm_s=20.0, speed_error_pct=0.005)  # 0.5% error
    return generate_drive_signal(
        duration_s=1.0, fs=50_000.0, rotation=rot, pullback=pb, seed=1
    )


def build_degraded_signal(spec: DriveSpec):
    """A binding torque cable: strong 1x + 3x NURD and 8% pullback overspeed."""
    rot = RotationParams(
        rpm=spec.rpm_nominal,
        nurd_harmonics=[
            NurdHarmonic(k=1, amplitude=0.18),  # dominant once-per-rev bind
            NurdHarmonic(k=3, amplitude=0.06),
        ],
        jitter_pct=0.01,
    )
    pb = PullbackParams(speed_mm_s=20.0, speed_error_pct=0.08, ripple_pct=0.05,
                        ripple_hz=5.0, noise_pct=0.01)
    return generate_drive_signal(
        duration_s=1.0, fs=50_000.0, rotation=rot, pullback=pb, seed=2
    )


def main() -> int:
    OUT.mkdir(exist_ok=True)
    spec = DriveSpec.from_yaml(REPO / "spec.yaml")

    print("=" * 70)
    print(f"Spec: {spec.name}")
    print(
        f"  rpm_nominal={spec.rpm_nominal}  nurd_max={spec.nurd_index_max_pct}% "
        f"pullback_acc={spec.pullback_accuracy_pct}% jitter_max={spec.jitter_max_pct}%"
    )
    print("=" * 70)

    for label, sig in [
        ("clean", build_clean_signal(spec)),
        ("degraded", build_degraded_signal(spec)),
    ]:
        result = characterize(sig, spec)
        ns = nurd.nurd_summary(sig)
        stab = rotation.rotation_stability(sig)

        print(f"\n--- Scenario: {label}  =>  {result.status} ---")
        print(
            f"  NURD index: {ns['nurd_index_pct_overall']:.2f}%  "
            f"(dominant harmonic {ns['dominant_harmonic_k']}x)"
        )
        print(
            f"  RPM mean {stab.rpm_mean:.1f} (nominal {stab.rpm_nominal:.0f}), "
            f"period std {stab.period_std_pct:.3f}%"
        )
        for c in result.checks:
            print(
                f"    [{c.status}] {c.name}: {c.measured:.3g}{c.units} "
                f"(limit {c.limit:.3g}{c.units})"
            )

        md_path = write_report(result, OUT / f"report_{label}.md")
        html_path = write_report(result, OUT / f"report_{label}.html")
        print(f"  wrote {md_path.name}, {html_path.name}")

    # Also dump the clean markdown report inline for visibility.
    print("\n" + "=" * 70)
    print("Full markdown report (clean scenario):")
    print("=" * 70)
    clean_result = characterize(build_clean_signal(spec), spec)
    print(render_markdown(clean_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
