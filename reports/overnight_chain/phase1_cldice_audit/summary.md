# clDice Audit — Post-Fix Report

> **Fix applied**: degenerate-case handling in `cldice_score` and `cldice_loss`.
> Empty predictions now correctly score 0.0 (was 1.0 due to symmetric epsilon bug).

## Old clDice Values (from metrics.json)

| Config | Dice | τ* | Old clDice | Anomalous? |
|--------|------|----|-----------|------------|
| A: Baseline | 0.8061 | 0.50 | 1.0000 | ⚠️ YES |
| B: +LFD Gate | 0.8057 | 0.50 | 0.5000 | ✅ No |
| C: +SPADE v1 | 0.8105 | 0.50 | 0.6934 | ✅ No |
| D: +PDE SPADE v2 | 0.8110 | 0.50 | 0.5544 | ✅ No |
| E: +Hessian | 0.8105 | 0.50 | 0.6790 | ✅ No |
| F: +Skel Recall | 0.8094 | 0.60 | 0.7315 | ✅ No |
| G: +Fractal BCE | 0.8091 | 0.55 | 0.6002 | ✅ No |
| H: Full Stack | 0.8090 | 0.55 | 0.6103 | ✅ No |
| OPT-A: PDE 15 steps | 0.8247 | 0.80 | — | ✅ No |
| OPT-B: Finer LFD | 0.8310 | 0.75 | — | ✅ No |
| OPT-C: SPADE64+Skel | 0.8227 | 0.80 | — | ✅ No |
| OPT-D: 100ep | 0.8413 | 0.70 | — | ✅ No |
| OPT-D v2: 200ep | 0.8413 | 0.70 | — | ✅ No |
| XD: FIVES Baseline | 0.8212 | 0.80 | — | ✅ No |
| XD: FIVES Fractal | 0.8348 | 0.75 | — | ✅ No |
| XD: FIVES Config F | 0.8339 | 0.70 | — | ✅ No |

**Anomalous entries**: 1 / 16

### Explanation of Anomalous Values

The anomalous clDice values (1.0000 or near-zero) occurred because:

1. The old `cldice_score()` used symmetric epsilon: `(2*tprec*tsens + eps) / (tprec + tsens + eps)`
2. When predictions are empty → skeleton is empty → tprec = eps/eps ≈ 1.0
3. The harmonic mean then evaluates to `(2*1*tsens + eps)/(1 + tsens + eps) ≈ 1.0`

**Fix**: both `cldice_score()` and `cldice_loss()` now detect degenerate cases
(skeleton sum ≤ eps) and return 0.0 / loss=1.0 respectively.

### Impact on Training

The `cldice_loss()` fix is more impactful than the metric fix:
- Old behavior: collapsed predictions → loss ≈ 0.5 (rewarding collapse)
- New behavior: collapsed predictions → loss = 1.0 (penalizing collapse)
- All **existing** trained models are unaffected (they were trained with the old loss)
- Future training (OPT-F, OPT-G) will benefit from the corrected loss gradient
