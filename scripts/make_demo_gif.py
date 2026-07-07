"""Generate assets/nurd_correction_demo.gif.

Animated side-by-side polar B-scan of a synthetic vessel phantom imaged by a
NURD-distorted drive: the left panel renders A-lines at the *assumed* uniform
angles (what an uncorrected scanner displays), the right panel renders the
encoder-based correction from :func:`oct_drive_test.correction.correct_nurd`.
The phantom outline is overlaid on both so the geometric distortion — and its
removal — is visible at a glance.

Run from the repo root:

    python scripts/make_demo_gif.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from oct_drive_test.correction import correct_nurd
from oct_drive_test.generator import NurdHarmonic, RotationParams, generate_drive_signal

OUT = Path(__file__).resolve().parents[1] / "assets" / "nurd_correction_demo.gif"

# ---------------------------------------------------------------------------
# Vessel phantom: wall radius vs angle, plus an echogenic wedge feature.
# ---------------------------------------------------------------------------

N_DEPTH = 110
R_MAX = 1.0


def wall_radius(angle: np.ndarray) -> np.ndarray:
    """Lumen wall radius [0..1] as a function of true angle [rad]."""
    return 0.62 + 0.06 * np.cos(3.0 * angle) + 0.03 * np.sin(angle)


def phantom_alines(angles: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Synthesize one A-line (depth profile) per true angle."""
    r = np.linspace(0.0, R_MAX, N_DEPTH)
    rw = wall_radius(np.mod(angles, 2.0 * np.pi))
    # Bright vessel wall band.
    img = np.exp(-((r[None, :] - rw[:, None]) ** 2) / (2 * 0.035**2))
    # Echogenic wedge (plaque) between 40 and 75 degrees.
    a = np.mod(angles, 2.0 * np.pi)
    wedge = (a > np.deg2rad(40)) & (a < np.deg2rad(75))
    plaque = np.exp(-((r[None, :] - 0.45) ** 2) / (2 * 0.09**2)) * 0.85
    img[wedge] += plaque[0]
    # Mild speckle.
    img += 0.08 * rng.standard_normal(img.shape)
    return np.clip(img, 0.0, 1.3)


def main() -> None:
    rng = np.random.default_rng(11)

    # Heavy but realistic NURD: 25% once-per-rev + 12% twice-per-rev velocity
    # modulation, plus 1% rotational jitter.
    rotation = RotationParams(
        rpm=6000.0,
        nurd_harmonics=[NurdHarmonic(k=1, amplitude=0.25), NurdHarmonic(k=2, amplitude=0.12, phase=1.1)],
        jitter_pct=0.01,
    )
    fs = 200_000.0
    sig = generate_drive_signal(duration_s=0.02, fs=fs, rotation=rotation, seed=3)

    # Take exactly one full revolution of A-lines out of the record.
    start = np.searchsorted(sig.theta, sig.theta[0] + 2.0 * np.pi * 0.2)
    stop = np.searchsorted(sig.theta, sig.theta[start] + 2.0 * np.pi)
    t = sig.t[start:stop]
    theta = sig.theta[start:stop]

    # Physical acquisition: each A-line samples the phantom at the TRUE angle.
    alines = phantom_alines(theta, rng)

    # Encoder-based correction onto an exact uniform angular grid.
    n = t.size
    corr = correct_nurd(t, theta, samples=alines, n_per_rev=n)
    corrected = corr.samples if corr.samples is not None else alines

    n_show = 720  # rendered angular columns
    idx = np.linspace(0, n - 1, n_show).astype(int)
    r = np.linspace(0.0, R_MAX, N_DEPTH)
    # Both panels start the sweep at the true angle of the first A-line, so the
    # phantom overlay lines up and the only difference is the angular grid.
    phi0 = float(theta[0])
    ang_grid = phi0 + np.linspace(0.0, 2.0 * np.pi, n_show)

    distorted_img = alines[idx]
    corrected_img = corrected[np.linspace(0, corrected.shape[0] - 1, n_show).astype(int)]

    fig, axes = plt.subplots(
        1, 2, figsize=(8.0, 4.3), subplot_kw={"projection": "polar"}, facecolor="#0d1117"
    )
    titles = [
        f"Uncorrected — NURD index {corr.nurd_index_before:.3f}",
        f"Encoder-corrected — NURD index {corr.nurd_index_after:.3f}",
    ]
    meshes, sweeps = [], []
    outline_angle = np.linspace(0.0, 2.0 * np.pi, 400)
    for ax, title in zip(axes, titles):
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, color="#e6edf3", fontsize=10, pad=12)
        mesh = ax.pcolormesh(
            ang_grid,
            r,
            np.zeros((N_DEPTH, n_show)),
            cmap="inferno",
            vmin=0.0,
            vmax=1.2,
            shading="nearest",
        )
        ax.plot(
            outline_angle,
            wall_radius(outline_angle),
            color="#2ea6ff",
            lw=1.5,
            ls="--",
            alpha=1.0,
            label="true wall",
        )
        (sweep,) = ax.plot([0, 0], [0, R_MAX], color="white", lw=1.2, alpha=0.9)
        meshes.append(mesh)
        sweeps.append(sweep)
    axes[0].legend(
        loc="lower center", bbox_to_anchor=(1.1, -0.18), frameon=False,
        labelcolor="#e6edf3", fontsize=8,
    )
    fig.suptitle(
        "NURD correction: same A-lines, angular grid rebuilt from the encoder",
        color="#e6edf3", fontsize=11,
    )

    n_frames = 40

    def update(frame: int):
        k = int((frame + 1) / n_frames * n_show)
        for mesh, sweep, img in zip(meshes, sweeps, (distorted_img, corrected_img)):
            shown = np.zeros((N_DEPTH, n_show))
            shown[:, :k] = img[:k].T
            mesh.set_array(shown.ravel())
            a = ang_grid[min(k, n_show - 1)]
            sweep.set_data([a, a], [0, R_MAX])
        return meshes + sweeps

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    OUT.parent.mkdir(exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=12), dpi=80)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
