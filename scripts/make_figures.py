#!/usr/bin/env python3
"""Generate portfolio visualizations for the OCT catheter drive characterization toolkit.

Calls the *real* public API (no re-implementation of any metric) to render four
figures into ``assets/``:

- angular_velocity.png   -- omega(t) wobble, clean vs NURD-degraded
- nurd_spectrum.png      -- spectral NURD harmonic ratios at k*f_rot
- pullback_linearity.png -- position vs time with linear fit + speed error
- nurd_per_rotation.png  -- per-revolution NURD index map vs spec limit

Engineering / V&V drive-mechanics characterization only. NO clinical or
diagnostic claims. Fixed seeds -> reproducible output. Headless (Agg backend).

Run:  .venv/bin/python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# House style (verbatim).
# ----------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#334155", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e2e8f0", "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.frameon": False, "lines.linewidth": 2.0,
})
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]

# ----------------------------------------------------------------------------
# Real API.
# ----------------------------------------------------------------------------
from oct_drive_test import nurd, pullback
from oct_drive_test.generator import (
    NurdHarmonic,
    PullbackParams,
    RotationParams,
    generate_drive_signal,
)
from oct_drive_test.spec import DriveSpec

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "assets"

BLUE, RED, GREEN, AMBER = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]
RPM = 6000.0          # 100 rev/s nominal proximal rotation
F_ROT = RPM / 60.0    # 100 Hz


# ----------------------------------------------------------------------------
# Scenario builders -- mirror examples/run_characterization.py conventions.
# ----------------------------------------------------------------------------
def build_clean(duration_s: float, fs: float):
    """Well-behaved drive: tiny 1x NURD, tiny jitter, accurate pullback."""
    rot = RotationParams(
        rpm=RPM,
        nurd_harmonics=[NurdHarmonic(k=1, amplitude=0.01)],
        jitter_pct=0.002,
    )
    pb = PullbackParams(speed_mm_s=20.0, speed_error_pct=0.005)
    return generate_drive_signal(
        duration_s=duration_s, fs=fs, rotation=rot, pullback=pb, seed=1
    )


def build_degraded(duration_s: float, fs: float):
    """Binding torque cable: strong 1x + 3x NURD, jitter, pullback overspeed."""
    rot = RotationParams(
        rpm=RPM,
        nurd_harmonics=[
            NurdHarmonic(k=1, amplitude=0.18),  # dominant once-per-rev bind
            NurdHarmonic(k=3, amplitude=0.06),
        ],
        jitter_pct=0.01,
    )
    pb = PullbackParams(
        speed_mm_s=20.0, speed_error_pct=0.08,
        ripple_pct=0.05, ripple_hz=5.0, noise_pct=0.01,
    )
    return generate_drive_signal(
        duration_s=duration_s, fs=fs, rotation=rot, pullback=pb, seed=2
    )


# ----------------------------------------------------------------------------
# Figure 1: angular velocity omega(t), clean vs degraded.
# ----------------------------------------------------------------------------
def fig_angular_velocity(path: Path) -> None:
    # Short record + moderate fs: a few revolutions, wobble clearly resolved.
    n_rev = 4
    duration = n_rev / F_ROT  # 0.04 s -> exactly 4 revolutions
    fs = 20_000.0
    clean = build_clean(duration, fs)
    degr = build_degraded(duration, fs)

    # Convert rad/s -> instantaneous RPM for an engineer-friendly y-axis.
    clean_rpm = clean.omega * 60.0 / (2.0 * np.pi)
    degr_rpm = degr.omega * 60.0 / (2.0 * np.pi)
    t_ms_c = clean.t * 1e3
    t_ms_d = degr.t * 1e3

    nurd_clean = nurd.nurd_index_pct(clean.omega)
    nurd_degr = nurd.nurd_index_pct(degr.omega)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(t_ms_d, degr_rpm, color=RED, label="degraded (NURD)", zorder=3)
    ax.plot(t_ms_c, clean_rpm, color=BLUE, label="clean", zorder=4)
    ax.axhline(RPM, color="#334155", lw=1.0, ls="--", zorder=2)
    ax.text(
        t_ms_d[-1], RPM, "  nominal 6000 rpm",
        va="center", ha="left", fontsize=9, color="#334155",
    )

    # Mark revolution boundaries (one per 1/f_rot).
    for r in range(1, n_rev):
        ax.axvline(r / F_ROT * 1e3, color="#cbd5e1", lw=0.8, ls=":", zorder=1)

    ax.set_xlabel("time [ms]")
    ax.set_ylabel("instantaneous rotation rate [rpm]")
    ax.set_title("Angular velocity within each revolution: clean vs NURD-degraded")
    ax.set_xlim(t_ms_d[0], t_ms_d[-1])
    ax.legend(loc="upper right", ncol=2)

    txt = (
        f"NURD index (std/mean of ω)\n"
        f"clean:    {nurd_clean:.2f} %\n"
        f"degraded: {nurd_degr:.2f} %"
    )
    ax.text(
        0.015, 0.04, txt, transform=ax.transAxes, va="bottom", ha="left",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cbd5e1", lw=0.8),
    )

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}  (clean {nurd_clean:.2f}% / degraded {nurd_degr:.2f}%)")


# ----------------------------------------------------------------------------
# Figure 2: spectral NURD harmonic ratios for the degraded signal.
# ----------------------------------------------------------------------------
def fig_nurd_spectrum(path: Path) -> None:
    # Long record + high fs -> clean spectral bins (matches example fidelity).
    degr = build_degraded(1.0, 50_000.0)
    spec = nurd.spectral_nurd(degr, k_max=10)

    k = spec.harmonic_orders
    ratios_pct = spec.harmonic_ratios * 100.0  # |OMEGA(k f_rot)| / |OMEGA(0)| in %
    dom_k = spec.dominant_k

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    colors = [RED if kk == dom_k else BLUE for kk in k]
    markerline, stemlines, baseline = ax.stem(k, ratios_pct, basefmt=" ")
    plt.setp(stemlines, linewidth=2.0)
    plt.setp(markerline, markersize=7)
    # Per-harmonic coloring (stem draws single-color; recolor manually).
    ax.cla()
    ax.grid(True)
    for kk, rr, cc in zip(k, ratios_pct, colors):
        ax.plot([kk, kk], [0, rr], color=cc, lw=2.4, zorder=2,
                solid_capstyle="round")
        ax.plot(kk, rr, "o", color=cc, ms=7, zorder=3)
    ax.axhline(0, color="#334155", lw=0.8)

    # Annotate the dominant harmonic.
    dom_idx = int(np.where(k == dom_k)[0][0])
    dom_val = ratios_pct[dom_idx]
    ax.annotate(
        f"dominant NURD harmonic\n{dom_k}× = {dom_val:.1f}%",
        xy=(dom_k, dom_val), xytext=(dom_k + 1.4, dom_val * 0.86),
        fontsize=10, color=RED, ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.4),
    )

    ax.set_xlabel("harmonic order  k  (multiple of f_rot = 100 Hz)")
    ax.set_ylabel("harmonic ratio  |Ω(k·f_rot)| / |Ω(0)|  [%]")
    ax.set_title("Spectral NURD: angular-velocity harmonic content (degraded drive)")
    ax.set_xticks(k)
    ax.set_xlim(0.4, 10.6)
    ax.set_ylim(0, max(ratios_pct) * 1.18)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}  (dominant {dom_k}x = {dom_val:.1f}%)")


# ----------------------------------------------------------------------------
# Figure 3: pullback position vs time with linear fit.
# ----------------------------------------------------------------------------
def fig_pullback_linearity(path: Path) -> None:
    degr = build_degraded(1.0, 50_000.0)
    pm = pullback.pullback_metrics(degr)

    t = degr.t
    pos = degr.position
    coeffs = np.polyfit(t, pos, 1)         # [slope mm/s, intercept mm]
    fit = np.polyval(coeffs, t)
    fit_speed = coeffs[0]
    commanded = degr.pullback_speed_nominal

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    # Decimate the raw trace for a light marker scatter; keep the line full-res.
    step = max(1, t.size // 400)
    ax.plot(t, pos, color=GREEN, lw=2.0, label="measured position", zorder=3)
    ax.plot(
        t, fit, color="#334155", lw=1.4, ls="--",
        label=f"linear fit ({fit_speed:.2f} mm/s)", zorder=4,
    )
    ax.scatter(
        t[::step], pos[::step], s=10, color=GREEN, alpha=0.35,
        zorder=2, edgecolors="none",
    )

    ax.set_xlabel("time [s]")
    ax.set_ylabel("pullback position [mm]")
    ax.set_title("Pullback linearity: measured position vs constant-velocity fit")
    ax.legend(loc="upper left")
    ax.set_xlim(t[0], t[-1])

    txt = (
        f"commanded:  {commanded:.2f} mm/s\n"
        f"measured:   {pm.mean_speed_mm_s:.2f} mm/s\n"
        f"speed error: {pm.speed_error_pct:+.2f} %\n"
        f"linearity R²: {pm.linearity_r2:.5f}"
    )
    ax.text(
        0.985, 0.04, txt, transform=ax.transAxes, va="bottom", ha="right",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cbd5e1", lw=0.8),
    )

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(
        f"  wrote {path.name}  (err {pm.speed_error_pct:+.2f}%, "
        f"R2 {pm.linearity_r2:.5f})"
    )


# ----------------------------------------------------------------------------
# Figure 4: per-rotation NURD index map, clean vs degraded, with spec limit.
# ----------------------------------------------------------------------------
def fig_nurd_per_rotation(path: Path) -> None:
    # ~0.5 s -> ~50 revolutions to populate the map.
    duration = 0.5
    fs = 50_000.0
    clean = build_clean(duration, fs)
    degr = build_degraded(duration, fs)

    pr_clean = nurd.nurd_index_per_rotation(clean)
    pr_degr = nurd.nurd_index_per_rotation(degr)

    spec = DriveSpec.from_yaml(REPO / "spec.yaml")
    limit = spec.nurd_index_max_pct

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(
        pr_degr["rotation_number"], pr_degr["nurd_index_pct"],
        color=RED, marker="o", ms=3.5, label="degraded (NURD)", zorder=3,
    )
    ax.plot(
        pr_clean["rotation_number"], pr_clean["nurd_index_pct"],
        color=BLUE, marker="o", ms=3.5, label="clean", zorder=4,
    )
    ax.axhline(
        limit, color=AMBER, lw=1.8, ls="--",
        label=f"spec limit ({limit:.0f}%)", zorder=2,
    )
    ax.fill_between(
        [-1, max(pr_degr["rotation_number"].max(),
                 pr_clean["rotation_number"].max()) + 1],
        limit, ax.get_ylim()[1], color=AMBER, alpha=0.06, zorder=0,
    )

    ax.set_xlabel("revolution number")
    ax.set_ylabel("NURD index  std(ω)/mean(ω)  [%]")
    ax.set_title("Per-revolution NURD index vs acceptance limit")
    ax.set_xlim(-0.5, max(pr_degr["rotation_number"].max(),
                          pr_clean["rotation_number"].max()) + 0.5)
    ax.legend(loc="center right", ncol=1)

    mean_clean = float(np.mean(pr_clean["nurd_index_pct"]))
    mean_degr = float(np.mean(pr_degr["nurd_index_pct"]))
    print(
        f"  wrote {path.name}  (mean clean {mean_clean:.2f}% / "
        f"degraded {mean_degr:.2f}%, limit {limit:.0f}%)"
    )

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"Rendering figures -> {ASSETS}")
    fig_angular_velocity(ASSETS / "angular_velocity.png")
    fig_nurd_spectrum(ASSETS / "nurd_spectrum.png")
    fig_pullback_linearity(ASSETS / "pullback_linearity.png")
    fig_nurd_per_rotation(ASSETS / "nurd_per_rotation.png")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
