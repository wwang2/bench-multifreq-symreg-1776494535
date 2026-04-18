"""FFT-guided analysis of the multi-frequency symbolic regression data."""
import numpy as np

data = np.loadtxt(
    "/tmp/git-evolve-bench-multifreq-symreg-1776494535/.worktrees/01-fft-multifreq/research/eval/train_data.csv",
    delimiter=",", skiprows=1
)
x = data[:, 0]
y = data[:, 1]
N = len(x)
print(f"N={N}, x∈[{x.min():.3f}, {x.max():.3f}]")
dx = x[1] - x[0]
print(f"dx = {dx:.6f}, span = {x[-1] - x[0]:.3f}")

# Baseline: simple polynomial fits
print("\n-- Polynomial baseline MSEs on training data --")
for deg in [1, 2, 3, 4, 5, 6]:
    coeffs = np.polyfit(x, y, deg)
    yhat = np.polyval(coeffs, x)
    mse = np.mean((y - yhat) ** 2)
    print(f"poly deg={deg}: train MSE = {mse:.5f}  coeffs = {coeffs}")

# FFT on raw y (periodic assumption for discovery)
# Do FFT on y values. Because x is evenly spaced in [-4, 4], we can compute FFT.
Y = np.fft.rfft(y)
mags = np.abs(Y) / N
freqs = np.fft.rfftfreq(N, d=dx)  # cycles per unit x
print("\n-- FFT of raw y (angular freq ω = 2π f) --")
order = np.argsort(-mags)
for k in order[:12]:
    print(f"k={k:3d}  f(cyc/x)={freqs[k]:.4f}  ω={2*np.pi*freqs[k]:.4f}  |Y|/N={mags[k]:.4f}")

# Residual after quadratic/cubic polynomial fit:
for deg in [2, 3, 4]:
    coeffs = np.polyfit(x, y, deg)
    yhat = np.polyval(coeffs, x)
    resid = y - yhat
    R = np.fft.rfft(resid)
    mags_r = np.abs(R) / N
    print(f"\n-- FFT of residual after deg={deg} poly fit --")
    order = np.argsort(-mags_r)
    for k in order[:10]:
        print(f"k={k:3d}  f(cyc/x)={freqs[k]:.4f}  ω={2*np.pi*freqs[k]:.4f}  |R|/N={mags_r[k]:.4f}")
