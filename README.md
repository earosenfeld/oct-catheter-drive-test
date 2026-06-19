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
