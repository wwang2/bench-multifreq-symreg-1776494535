---
issue: 2
parents: []
eval_version: eval-v1
metric: 0.0
---

# Research Notes — orbit/01-fft-multifreq (primary)

## Result
- **Test MSE = 0.0000000000** on clean held-out test set (3/3 seeds identical — evaluator is deterministic).
- Target was MSE < 0.005 → **met by ~5 orders of magnitude** (exact symbolic recovery).

## Closed form
```
f(x) = sin(x) + 0.3 * sin(5x) + 0.1 * x²
```

## Discovery pipeline
1. **Polynomial trend fit** on the 60 training points. Least-squares on
   `{1, x, x²}` captured the low-frequency component and left a
   residual dominated by periodic structure.
2. **FFT of the residual** (zero-padded to ~4096 samples to sharpen
   bins) showed two clean, narrow peaks near angular frequencies
   ω ≈ 1 and ω ≈ 5. Phases at the peaks were both close to +π/2
   (i.e. pure sines, no cosine component).
3. **Joint grid refinement** over (ω₁, ω₂) ∈ [0.9, 1.1] × [4.9, 5.1]
   with a per-(ω₁, ω₂) closed-form linear least-squares amplitude fit:
   minimum residual at ω₁ = 1.002, ω₂ = 4.999 — effectively integer.
4. **Minimal sufficient basis** `{1, x², sin(x), sin(5x)}` fit with
   `np.linalg.lstsq`. The constant, `x`, `cos(x)`, `cos(5x)` columns
   all came back statistically indistinguishable from zero on the
   training data.
5. Estimated clean coefficients: A₁ ≈ 1.005, A₅ ≈ 0.303, c₂ ≈ 0.103.
   Within 2-3 standard errors of the round values 1.0, 0.3, 0.1, so
   we **snap to the round symbolic form**.

## Why the clean snap wins on the test set
The test set is 400 clean (noise-free) points on the same interval.
The training noise σ = 0.05 / √60 ≈ 0.006 gives a per-coefficient
standard error of roughly that order. On the clean test set, any
deviation from the true coefficients shows up as squared error; the
rounded integer/round-fraction form matches the generator exactly,
so MSE = 0 to floating-point precision.

## Constraints
- No sklearn, scipy.optimize, or curve_fit was used.
- All coefficient tuning done with `np.fft.rfft` (spectrum discovery)
  and `np.linalg.lstsq` (linear-in-unknowns amplitude fit), plus
  visual inspection of the residual + power spectrum.

## Artifacts
- `figures/fft_analysis.png` — residual-after-polynomial FFT showing
  the two peaks.
- `figures/narrative.png` — story of discovery: raw data → poly fit
  residual → FFT peaks → final closed-form fit.
- `figures/results.png` — side-by-side comparison of the closed form
  against the noisy training data and the clean test curve.
