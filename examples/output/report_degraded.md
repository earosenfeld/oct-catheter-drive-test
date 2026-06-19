# OCT Drive Characterization Report

**Spec:** OCT catheter drive — example acceptance spec  
**Overall result:** **FAIL**

_Engineering bench characterization of drive mechanics only. No clinical, diagnostic, or medical-efficacy claim is made._

## Acceptance limits

| Parameter | Measured | Limit | Check | Result |
|---|---|---|---|---|
| NURD index | 13.45 % | 5 % | <= limit | **FAIL** |
| RPM mean error | 0.01234 % (abs) | 2 % (abs) | |RPM error| <= limit | **PASS** |
| Rotational jitter (period std) | 0.7237 % | 2 % | <= limit | **PASS** |
| Pullback speed error | 7.995 % (abs) | 3 % (abs) | |speed error| <= limit | **FAIL** |

## Supporting measurements

### NURD
- Overall NURD index: 13.45 %
- Per-rotation NURD (mean / max): 13.42 % / 14.66 %
- Dominant NURD harmonic: 1x (ratio 0.08946)
- Revolutions analyzed: 100

### Rotational stability
- RPM mean: 6001
- RPM std: 43.46
- Period std / peak-to-peak: 0.7237 % / 3.685 %
- Wow / flutter RMS: 0.1258 % / 0.7127 %

### Pullback
- Commanded / measured speed: 20 / 21.6 mm/s
- Speed error: 7.995 %
- Speed CV: 3.576 %; max deviation: 6.886 %
- Linearity R^2: 1
