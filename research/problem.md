# Multi-Frequency Symbolic Regression

## Problem Statement
60 noisy training points `(x, y)` at `research/eval/train_data.csv`, with
`x ∈ [-4, 4]` (n=60 evenly spaced) and additive Gaussian noise `σ=0.05`.
Propose a closed-form `f(x)` that best fits the underlying signal.
The signal is expected to contain **multiple frequency components**, some
with small amplitude; a naive low-frequency fit will miss structure worth
several percent of MSE.

## Constraints
- No `sklearn`, no `scipy.optimize`, no `curve_fit`.
- Tune coefficients by inspection. FFT of residuals is permitted (and
  encouraged) for discovering frequencies.

## Solution Interface
`orbits/<name>/solution.py` must export `f(x: np.ndarray) -> np.ndarray`.

## Success Metric
MSE on a clean held-out test set of 400 evenly-spaced points over the same
`x ∈ [-4, 4]` interval (minimize). **Target: MSE < 0.005.**

## Budget
Max 3 orbits. Each orbit spawns 2 cross-validation replicas
(`execution.parallel_agents=2`) that propose independently.
