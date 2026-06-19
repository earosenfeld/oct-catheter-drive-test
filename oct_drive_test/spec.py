"""Specification loading and PASS/FAIL characterization reporting (V&V).

A drive-characterization *spec* defines the acceptance limits for the mechanical
drive. Running the metrics against a captured (or synthetic) drive signal and
comparing to the spec yields a stamped PASS/FAIL report -- the engineering V&V
deliverable for a build/lot.

Spec fields (spec.yaml)
-----------------------
- rpm_nominal          : commanded rotation rate [rev/min]
- rpm_tolerance_pct    : allowed |mean RPM error| from nominal [%]
- nurd_index_max_pct   : allowed overall NURD index [%]
- pullback_accuracy_pct: allowed |pullback speed error| [%] (skipped if no pullback)
- jitter_max_pct       : allowed rotational period std (period-to-period) [%]

This module ONLY characterizes drive mechanics. It does NOT assess image
diagnostic quality or any clinical/medical-efficacy criterion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ._signal import DriveSignal
from .nurd import nurd_index_pct, nurd_summary
from .pullback import pullback_metrics
from .rotation import rotation_stability


@dataclass
class DriveSpec:
    """Acceptance limits for an OCT drive characterization."""

    rpm_nominal: float
    rpm_tolerance_pct: float
    nurd_index_max_pct: float
    pullback_accuracy_pct: float
    jitter_max_pct: float
    name: str = "OCT drive spec"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DriveSpec":
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "DriveSpec":
        required = [
            "rpm_nominal",
            "rpm_tolerance_pct",
            "nurd_index_max_pct",
            "pullback_accuracy_pct",
            "jitter_max_pct",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"spec missing required fields: {missing}")
        return cls(
            rpm_nominal=float(data["rpm_nominal"]),
            rpm_tolerance_pct=float(data["rpm_tolerance_pct"]),
            nurd_index_max_pct=float(data["nurd_index_max_pct"]),
            pullback_accuracy_pct=float(data["pullback_accuracy_pct"]),
            jitter_max_pct=float(data["jitter_max_pct"]),
            name=str(data.get("name", "OCT drive spec")),
        )

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))


@dataclass
class LimitCheck:
    """One measured-vs-limit acceptance line item."""

    name: str
    measured: float
    limit: float
    units: str
    comparison: str  # human-readable, e.g. "<= limit" or "|.| <= limit"
    passed: bool

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class CharacterizationResult:
    """Full characterization result: per-limit checks + overall verdict."""

    spec_name: str
    checks: list[LimitCheck] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def characterize(signal: DriveSignal, spec: DriveSpec) -> CharacterizationResult:
    """Run all metrics on ``signal`` and check each against ``spec``.

    Returns a :class:`CharacterizationResult` with a PASS/FAIL line per limit
    and an overall verdict (FAIL if any single limit fails).
    """
    checks: list[LimitCheck] = []

    # --- NURD index limit ---
    nurd_pct = nurd_index_pct(signal.omega)
    checks.append(
        LimitCheck(
            name="NURD index",
            measured=nurd_pct,
            limit=spec.nurd_index_max_pct,
            units="%",
            comparison="<= limit",
            passed=bool(nurd_pct <= spec.nurd_index_max_pct),
        )
    )

    # --- RPM mean error / tolerance ---
    stab = rotation_stability(signal)
    rpm_err = abs(stab.rpm_error_pct)
    checks.append(
        LimitCheck(
            name="RPM mean error",
            measured=rpm_err,
            limit=spec.rpm_tolerance_pct,
            units="% (abs)",
            comparison="|RPM error| <= limit",
            passed=bool(rpm_err <= spec.rpm_tolerance_pct),
        )
    )

    # --- Rotational jitter (period-to-period std) ---
    jitter_pct = stab.period_std_pct
    checks.append(
        LimitCheck(
            name="Rotational jitter (period std)",
            measured=jitter_pct,
            limit=spec.jitter_max_pct,
            units="%",
            comparison="<= limit",
            passed=bool(jitter_pct <= spec.jitter_max_pct),
        )
    )

    # --- Pullback accuracy (only when a pullback was commanded) ---
    pb = pullback_metrics(signal)
    if signal.pullback_speed_nominal != 0.0:
        pb_err = abs(pb.speed_error_pct)
        checks.append(
            LimitCheck(
                name="Pullback speed error",
                measured=pb_err,
                limit=spec.pullback_accuracy_pct,
                units="% (abs)",
                comparison="|speed error| <= limit",
                passed=bool(pb_err <= spec.pullback_accuracy_pct),
            )
        )

    details = {
        "nurd": nurd_summary(signal),
        "rotation": {
            "rpm_mean": stab.rpm_mean,
            "rpm_std": stab.rpm_std,
            "rpm_error_pct": stab.rpm_error_pct,
            "period_std_pct": stab.period_std_pct,
            "period_p2p_pct": stab.period_p2p_pct,
            "wow_rms_pct": stab.wow_rms_pct,
            "flutter_rms_pct": stab.flutter_rms_pct,
        },
        "pullback": {
            "commanded_speed_mm_s": pb.commanded_speed_mm_s,
            "mean_speed_mm_s": pb.mean_speed_mm_s,
            "speed_error_pct": pb.speed_error_pct,
            "speed_cv_pct": pb.speed_cv_pct,
            "max_abs_deviation_pct": pb.max_abs_deviation_pct,
            "linearity_r2": pb.linearity_r2,
        },
        "signal_meta": signal.meta,
    }

    return CharacterizationResult(
        spec_name=spec.name, checks=checks, details=details
    )


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    if isinstance(x, float) and (np.isnan(x)):
        return "n/a"
    return f"{x:.4g}"


def render_markdown(result: CharacterizationResult) -> str:
    """Render a characterization result as a Markdown V&V report."""
    lines: list[str] = []
    lines.append(f"# OCT Drive Characterization Report")
    lines.append("")
    lines.append(f"**Spec:** {result.spec_name}  ")
    lines.append(f"**Overall result:** **{result.status}**")
    lines.append("")
    lines.append(
        "_Engineering bench characterization of drive mechanics only. "
        "No clinical, diagnostic, or medical-efficacy claim is made._"
    )
    lines.append("")
    lines.append("## Acceptance limits")
    lines.append("")
    lines.append("| Parameter | Measured | Limit | Check | Result |")
    lines.append("|---|---|---|---|---|")
    for c in result.checks:
        lines.append(
            f"| {c.name} | {_fmt(c.measured)} {c.units} "
            f"| {_fmt(c.limit)} {c.units} | {c.comparison} | **{c.status}** |"
        )
    lines.append("")

    # Supporting detail
    nurd = result.details.get("nurd", {})
    rot = result.details.get("rotation", {})
    pb = result.details.get("pullback", {})
    lines.append("## Supporting measurements")
    lines.append("")
    lines.append("### NURD")
    lines.append(
        f"- Overall NURD index: {_fmt(nurd.get('nurd_index_pct_overall', float('nan')))} %"
    )
    lines.append(
        f"- Per-rotation NURD (mean / max): "
        f"{_fmt(nurd.get('nurd_index_pct_per_rotation_mean', float('nan')))} % / "
        f"{_fmt(nurd.get('nurd_index_pct_per_rotation_max', float('nan')))} %"
    )
    lines.append(
        f"- Dominant NURD harmonic: {nurd.get('dominant_harmonic_k', 'n/a')}x "
        f"(ratio {_fmt(nurd.get('dominant_harmonic_ratio', float('nan')))})"
    )
    lines.append(f"- Revolutions analyzed: {nurd.get('n_rotations', 'n/a')}")
    lines.append("")
    lines.append("### Rotational stability")
    lines.append(f"- RPM mean: {_fmt(rot.get('rpm_mean', float('nan')))}")
    lines.append(f"- RPM std: {_fmt(rot.get('rpm_std', float('nan')))}")
    lines.append(
        f"- Period std / peak-to-peak: {_fmt(rot.get('period_std_pct', float('nan')))} % / "
        f"{_fmt(rot.get('period_p2p_pct', float('nan')))} %"
    )
    lines.append(
        f"- Wow / flutter RMS: {_fmt(rot.get('wow_rms_pct', float('nan')))} % / "
        f"{_fmt(rot.get('flutter_rms_pct', float('nan')))} %"
    )
    lines.append("")
    lines.append("### Pullback")
    if pb.get("commanded_speed_mm_s", 0.0):
        lines.append(
            f"- Commanded / measured speed: {_fmt(pb.get('commanded_speed_mm_s', float('nan')))} / "
            f"{_fmt(pb.get('mean_speed_mm_s', float('nan')))} mm/s"
        )
        lines.append(
            f"- Speed error: {_fmt(pb.get('speed_error_pct', float('nan')))} %"
        )
        lines.append(
            f"- Speed CV: {_fmt(pb.get('speed_cv_pct', float('nan')))} %; "
            f"max deviation: {_fmt(pb.get('max_abs_deviation_pct', float('nan')))} %"
        )
        lines.append(
            f"- Linearity R^2: {_fmt(pb.get('linearity_r2', float('nan')))}"
        )
    else:
        lines.append("- No pullback commanded (spin-only acquisition).")
    lines.append("")
    return "\n".join(lines)


def render_html(result: CharacterizationResult) -> str:
    """Render a minimal standalone HTML version of the report."""
    color = "#1a7f37" if result.passed else "#cf222e"
    rows = []
    for c in result.checks:
        rc = "#1a7f37" if c.passed else "#cf222e"
        rows.append(
            f"<tr><td>{c.name}</td><td>{_fmt(c.measured)} {c.units}</td>"
            f"<td>{_fmt(c.limit)} {c.units}</td><td>{c.comparison}</td>"
            f"<td style='color:{rc};font-weight:bold'>{c.status}</td></tr>"
        )
    table = "\n".join(rows)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>OCT Drive Characterization</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color:#1f2328; }}
table {{ border-collapse: collapse; margin-top: 1rem; }}
th, td {{ border: 1px solid #d0d7de; padding: 6px 12px; text-align:left; }}
th {{ background:#f6f8fa; }}
.verdict {{ color:{color}; font-weight:bold; font-size:1.2rem; }}
.disclaimer {{ color:#656d76; font-style:italic; margin-top:.5rem; }}
</style></head>
<body>
<h1>OCT Drive Characterization Report</h1>
<p><b>Spec:</b> {result.spec_name}</p>
<p class="verdict">Overall result: {result.status}</p>
<p class="disclaimer">Engineering bench characterization of drive mechanics only.
No clinical, diagnostic, or medical-efficacy claim is made.</p>
<table>
<tr><th>Parameter</th><th>Measured</th><th>Limit</th><th>Check</th><th>Result</th></tr>
{table}
</table>
</body></html>
"""


def write_report(
    result: CharacterizationResult,
    path: str | Path,
    fmt: str | None = None,
) -> Path:
    """Write the report to ``path``; format inferred from suffix if ``fmt`` None.

    Supported formats: ``md`` (markdown) and ``html``.
    """
    path = Path(path)
    if fmt is None:
        fmt = "html" if path.suffix.lower() in (".html", ".htm") else "md"
    if fmt == "html":
        path.write_text(render_html(result))
    else:
        path.write_text(render_markdown(result))
    return path
