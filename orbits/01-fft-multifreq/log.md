---
issue: 2
parents: []
eval_version: eval-v1
metric: 0.0002617728
---

# Research Notes — orbit/01-fft-multifreq.r1 (replica)

## Result
- **Test MSE = 0.0002617728** on the clean held-out test set (3/3
  seeds identical — evaluator is deterministic).
- Target was MSE < 0.005 → **met by ~20× margin**.

## Closed form
```
y(x) = C0 + C1·x + C2·x² + A1·sin(x) + A5·sin(5x)

C0 = -0.02303991369753
C1 = -0.00070483289662
C2 =  0.10277599471247
A1 =  1.00541989330001
A5 =  0.30331892810807
```

## Approach (independent of primary)
1. Smooth-polynomial trend fit to the 60 training points.
2. FFT of the polynomial residual exposed two narrow-band peaks at
   ω ≈ 1 and ω ≈ 5. Zero-padded + 3-ω brute-force refinement
   confirmed the exact frequencies.
3. Both peaks carried phase ≈ +π/2 → pure sines, no cosine
   component.
4. Fit the basis `{1, x, x², sin(x), sin(5x)}` via one-shot
   `np.linalg.lstsq` on the 60 training points. No iterative
   optimization — just a single linear solve on a fixed basis.
5. Kept the raw numeric coefficients (no symbolic snap-to-round).
   Train RMS ≈ 0.043, flush with σ = 0.05.

## Difference vs. primary
- Primary **snaps** coefficients to the clean round values
  (1, 0.3, 0.1, and drops the near-zero constant / linear / small
  quadratic deviation) — this is why primary achieves MSE ≈ 0
  on the clean test set.
- Replica **keeps** the fitted coefficients as-is; training-noise
  errors of order σ/√N ≈ 6·10⁻³ per coefficient leak into a
  non-zero test MSE, but still ~20× below the target.
- Both independently rediscover the same underlying symbolic
  structure — strong cross-validation of the discovery.

## Constraints
- No sklearn, no scipy.optimize, no curve_fit. Only `np.fft.rfft`
  and `np.linalg.lstsq`.

## Artifacts
- `figures/narrative.png` — discovery story: data → residual → FFT
  peaks → fitted closed form.
- `figures/results.png` — predicted curve vs. training points and
  clean test curve.
- `figures/fft_analysis.png` — residual spectrum with peaks marked.
