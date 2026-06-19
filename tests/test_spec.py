"""Validation of spec loading and PASS/FAIL characterization reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from oct_drive_test.spec import (
    DriveSpec,
    characterize,
    render_html,
    render_markdown,
    write_report,
)

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def spec():
    return DriveSpec.from_yaml(REPO / "spec.yaml")


def test_spec_loads_from_yaml(spec):
    assert spec.rpm_nominal == 6000.0
    assert spec.nurd_index_max_pct == 5.0
    assert spec.pullback_accuracy_pct == 3.0
    assert spec.jitter_max_pct == 2.0


def test_spec_missing_field_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("rpm_nominal: 6000\n")  # missing the rest
    with pytest.raises(ValueError, match="missing required"):
        DriveSpec.from_yaml(bad)


def test_clean_signal_passes(spec, make_signal):
    """A clean, in-spec drive must PASS every limit."""
    sig = make_signal(
        nurd=[(1, 0.01)],  # ~0.7% index, well under 5%
        jitter_pct=0.002,
        pullback_mm_s=20.0,
        speed_error_pct=0.005,  # 0.5% << 3%
        duration_s=1.0,
    )
    result = characterize(sig, spec)
    assert result.passed, [
        (c.name, c.measured, c.limit) for c in result.checks if not c.passed
    ]
    assert result.status == "PASS"


def test_excess_nurd_fails_nurd_limit(spec, make_signal):
    """Injected excess NURD must FAIL specifically the NURD limit."""
    sig = make_signal(
        nurd=[(1, 0.20)],  # index ~14% >> 5% limit
        jitter_pct=0.0,
        duration_s=1.0,
    )
    result = characterize(sig, spec)
    assert not result.passed
    nurd_check = next(c for c in result.checks if c.name == "NURD index")
    assert not nurd_check.passed
    assert nurd_check.measured > spec.nurd_index_max_pct


def test_excess_jitter_fails_jitter_limit(spec, make_signal):
    """Injected excess jitter must FAIL the jitter (period std) limit."""
    sig = make_signal(jitter_pct=0.08, duration_s=1.0, seed=5)
    result = characterize(sig, spec)
    jitter_check = next(
        c for c in result.checks if "jitter" in c.name.lower()
    )
    assert not jitter_check.passed
    assert not result.passed


def test_excess_pullback_error_fails_pullback_limit(spec, make_signal):
    """Injected excess pullback error must FAIL the pullback limit."""
    sig = make_signal(
        pullback_mm_s=20.0, speed_error_pct=0.10, duration_s=1.0  # 10% >> 3%
    )
    result = characterize(sig, spec)
    pb_check = next(c for c in result.checks if "Pullback" in c.name)
    assert not pb_check.passed
    assert not result.passed


def test_no_pullback_check_when_spin_only(spec, make_signal):
    """Spin-only acquisition omits the pullback check entirely."""
    sig = make_signal(nurd=[(1, 0.01)], jitter_pct=0.002, duration_s=1.0)
    result = characterize(sig, spec)
    assert not any("Pullback" in c.name for c in result.checks)


def test_markdown_report_contains_verdict_and_disclaimer(spec, make_signal):
    sig = make_signal(nurd=[(1, 0.2)], duration_s=1.0)
    md = render_markdown(characterize(sig, spec))
    assert "FAIL" in md
    assert "NURD" in md
    # Must carry the non-clinical disclaimer.
    assert "clinical" in md.lower()


def test_html_report_renders(spec, make_signal):
    sig = make_signal(nurd=[(1, 0.01)], jitter_pct=0.002, duration_s=1.0)
    html = render_html(characterize(sig, spec))
    assert "<table" in html
    assert "PASS" in html
    assert "clinical" in html.lower()


def test_write_report_infers_format(spec, make_signal, tmp_path):
    sig = make_signal(nurd=[(1, 0.01)], jitter_pct=0.002, duration_s=1.0)
    result = characterize(sig, spec)
    md = write_report(result, tmp_path / "r.md")
    html = write_report(result, tmp_path / "r.html")
    assert md.read_text().startswith("# OCT Drive Characterization")
    assert "<!doctype html>" in html.read_text().lower()


def test_spec_roundtrip_yaml(spec, tmp_path):
    out = tmp_path / "spec_out.yaml"
    spec.to_yaml(out)
    reloaded = DriveSpec.from_yaml(out)
    assert reloaded.nurd_index_max_pct == spec.nurd_index_max_pct
    assert reloaded.rpm_nominal == spec.rpm_nominal
