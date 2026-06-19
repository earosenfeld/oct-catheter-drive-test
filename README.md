# OCT Catheter Drive Test

Bench **characterization toolkit for the drive mechanics of intravascular OCT
(Optical Coherence Tomography) catheters** — rotational stability, **NURD
(Non-Uniform Rotational Distortion)**, and pullback accuracy — with
ground-truth-validated metrics and an automated pass/fail characterization report.

> Engineering verification & validation of *drive mechanics* only. This is a
> simulation / signal-analysis bench and makes **no clinical or diagnostic claims**.

## Why this exists

An OCT catheter images by spinning a fiber-optic core (typically ~100 rev/s) at
the distal tip while pulling it back along the vessel. Image quality is governed
by how *uniformly* that core rotates and how *accurately* it is pulled back:

- **NURD** — torque-cable friction/binding makes the distal optics speed up and
  slow down within a single revolution, smearing features azimuthally. It is the
  #1 image-quality killer and the headline metric here.
- **Rotational jitter / wow & flutter** — revolution-to-revolution speed instability.
- **Pullback accuracy & uniformity** — sets the longitudinal (pullback-axis) image scale.

## How it works

A synthetic (or measured) drive signal — the rotation and pullback encoder
channels — flows through the metric analyzers, each result is compared against
`spec.yaml`, and a PASS/FAIL characterization report is stamped.

```mermaid
flowchart LR
    A["Drive signal<br/>rotation + pullback<br/>encoders"] --> B["Metrics<br/>• NURD index<br/>• spectral NURD<br/>• rotational stability<br/>• pullback accuracy<br/>• angular error"]
    B --> C["spec.yaml<br/>acceptance limits"]
    C --> D["PASS / FAIL<br/>characterization report"]
```

## Metrics implemented

| Metric | Definition |
|---|---|
| NURD index | `std(ω) / mean(ω)` over a revolution (dimensionless, ×100 → %) |
| Spectral NURD | Hann-windowed FFT of `ω(t)`; harmonic content at `k·f_rot` relative to DC, with the dominant harmonic |
| Per-rotation NURD map | NURD index for each completed revolution |
| Rotational stability | period-to-period std & peak-to-peak; wow (slow) vs flutter (fast) split; RPM mean/std |
| Pullback accuracy | measured vs commanded mm/s, % error, speed CV, max deviation, position-vs-time linearity R² |
| Angular position error | cumulative commanded-vs-measured θ per revolution |

A **synthetic drive-signal generator** injects *known* NURD harmonics, jitter and
pullback error, so every metric is validated against ground truth in the test
suite (inject a known distortion → assert the analyzer recovers it).

## Visualizations

Generated from the real API by `scripts/make_figures.py` (fixed seeds, headless),
contrasting a clean in-spec drive against a degraded one (binding torque cable:
strong 1× + 3× NURD, jitter, 8% pullback overspeed). Engineering characterization
of drive mechanics only — no clinical or diagnostic interpretation.

![Angular velocity wobble](assets/angular_velocity.png)

*Instantaneous rotation rate across four revolutions. The clean drive (blue)
holds ~6000 rpm; the degraded drive (red) speeds up and slows down within each
revolution — this within-rev modulation is NURD. The NURD index `std(ω)/mean(ω)`
rises from 0.7% to 13.3%.*

![Spectral NURD](assets/nurd_spectrum.png)

*Harmonic content of `ω(t)` at multiples of the rotation frequency (`f_rot` =
100 Hz), normalized to DC. The dominant once-per-revolution (1×) component
flags the bind; the recovered ratios match the injected harmonic amplitudes
(`H_k ≈ a_k/2`).*

![Pullback linearity](assets/pullback_linearity.png)

*Pullback position vs time with a constant-velocity fit overlaid. The measured
speed (21.6 mm/s) exceeds the commanded 20.0 mm/s by +8.0% — a calibration bias
— while linearity stays high (R² = 0.99998).*

![Per-rotation NURD map](assets/nurd_per_rotation.png)

*NURD index computed for each completed revolution, against the `spec.yaml`
acceptance limit (5%). The clean drive stays near 0.7%; the degraded drive sits
~13% across every revolution, in the fail zone.*

## Install

```bash
pip install -e ".[dev]"      # numpy, scipy, pyyaml (+ pytest, matplotlib)
```

## Quickstart

```bash
# Two-scenario example: a clean in-spec drive (PASS) + a degraded drive (FAIL).
python examples/run_characterization.py

# Characterize a synthetic signal against a spec and stamp PASS/FAIL:
oct-drive-report --spec spec.yaml
```

```python
from oct_drive_test import nurd
from oct_drive_test.generator import generate_drive_signal, NurdHarmonic

sig = generate_drive_signal(...)         # inject known NURD/jitter/pullback (see example)
summary = nurd.nurd_summary(sig)
print(summary["nurd_index_pct_overall"], summary["dominant_harmonic_k"])
```

## Characterization report & spec

`spec.yaml` declares acceptance limits (nominal RPM and tolerance, max NURD index,
pullback accuracy, jitter ceiling). The report runner measures a dataset and
stamps **PASS/FAIL per limit** — the structure of a real V&V characterization record.

## Package layout

```
oct_drive_test/
├── generator.py   # synthetic drive-signal generator (ground-truth injection)
├── nurd.py        # NURD index + spectral / per-rotation NURD
├── rotation.py    # rotational stability: jitter, wow/flutter, RPM stats
├── pullback.py    # pullback speed accuracy & uniformity
├── angular.py     # angular position error
├── spec.py        # spec.yaml loading + pass/fail characterization report
├── backlash.py    # torque-cable wind-up / backlash (stretch)
└── aline.py       # A-line trigger-interval timing (stretch)
```

## Testing

```bash
pytest -q          # ground-truth validation of every metric
```

## License

MIT © Eric Rosenfeld
