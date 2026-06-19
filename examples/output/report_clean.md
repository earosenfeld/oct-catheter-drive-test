# OCT Drive Characterization Report

**Spec:** OCT catheter drive — example acceptance spec  
**Overall result:** **PASS**

_Engineering bench characterization of drive mechanics only. No clinical, diagnostic, or medical-efficacy claim is made._

## Acceptance limits

| Parameter | Measured | Limit | Check | Result |
|---|---|---|---|---|
| NURD index | 0.7213 % | 5 % | <= limit | **PASS** |
| RPM mean error | 0.0005845 % (abs) | 2 % (abs) | |RPM error| <= limit | **PASS** |
| Rotational jitter (period std) | 0.1283 % | 2 % | <= limit | **PASS** |
| Pullback speed error | 0.5 % (abs) | 3 % (abs) | |speed error| <= limit | **PASS** |

## Supporting measurements

### NURD
- Overall NURD index: 0.7213 %
- Per-rotation NURD (mean / max): 0.7004 % / 1.018 %
- Dominant NURD harmonic: 1x (ratio 0.004899)
- Revolutions analyzed: 100

### Rotational stability
- RPM mean: 6000
- RPM std: 7.701
- Period std / peak-to-peak: 0.1283 % / 0.6384 %
- Wow / flutter RMS: 0.004598 % / 0.1282 %

### Pullback
- Commanded / measured speed: 20 / 20.1 mm/s
- Speed error: 0.5 %
- Speed CV: 1.14e-10 %; max deviation: 4.757e-10 %
- Linearity R^2: 1
