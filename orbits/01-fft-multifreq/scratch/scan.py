"""Frequency scan: for each candidate ω, fit model y = poly(deg) + A cos(ωx) + B sin(ωx)
and compute residual MSE."""
import numpy as np

data = np.loadtxt(
    "/tmp/git-evolve-bench-multifreq-symreg-1776494535/.worktrees/01-fft-multifreq/research/eval/train_data.csv",
    delimiter=",", skiprows=1
)
x = data[:, 0]
y = data[:, 1]
N = len(x)

def design(x, omegas, deg):
    cols = []
    for p in range(deg + 1):
        cols.append(x ** p)
    for w in omegas:
        cols.append(np.cos(w * x))
        cols.append(np.sin(w * x))
    return np.vstack(cols).T

def fit(x, y, omegas, deg):
    A = design(x, omegas, deg)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    mse = np.mean((y - pred) ** 2)
    return mse, coef

# Step 1: scan single ω with poly deg=2 background
print("-- scan single ω, poly deg=2 --")
omegas = np.linspace(0.1, 8.0, 400)
best = []
for w in omegas:
    mse, _ = fit(x, y, [w], deg=2)
    best.append((mse, w))
best.sort()
for mse, w in best[:10]:
    print(f"  ω={w:.4f}  MSE={mse:.5f}")

# Step 2: scan single ω with poly deg=3 background
print("\n-- scan single ω, poly deg=3 --")
omegas = np.linspace(0.1, 8.0, 400)
best = []
for w in omegas:
    mse, _ = fit(x, y, [w], deg=3)
    best.append((mse, w))
best.sort()
for mse, w in best[:10]:
    print(f"  ω={w:.4f}  MSE={mse:.5f}")

# Step 3: joint scan of two ω with poly deg=2
print("\n-- joint scan (ω1, ω2), poly deg=2 --")
omegas1 = np.linspace(0.3, 2.0, 60)
omegas2 = np.linspace(2.0, 8.0, 80)
best = []
for w1 in omegas1:
    for w2 in omegas2:
        mse, _ = fit(x, y, [w1, w2], deg=2)
        best.append((mse, w1, w2))
best.sort()
for mse, w1, w2 in best[:10]:
    print(f"  ω1={w1:.4f}  ω2={w2:.4f}  MSE={mse:.5f}")

# Step 4: look at best and get coefficients
print("\n-- best (ω1, ω2) with poly deg=3 coefficients --")
omegas1 = np.linspace(0.3, 2.0, 80)
omegas2 = np.linspace(2.0, 8.0, 120)
best_mse = np.inf
best_pair = None
for w1 in omegas1:
    for w2 in omegas2:
        mse, coef = fit(x, y, [w1, w2], deg=3)
        if mse < best_mse:
            best_mse = mse
            best_pair = (w1, w2, coef)
w1, w2, coef = best_pair
print(f"best: ω1={w1:.5f}  ω2={w2:.5f}  train MSE={best_mse:.5f}")
print(f"coefficients: {coef}")
# coef = [c0, c1, c2, c3, A1, B1, A2, B2]
c0, c1, c2, c3, A1, B1, A2, B2 = coef
amp1 = np.hypot(A1, B1); ph1 = np.arctan2(-B1, A1)
amp2 = np.hypot(A2, B2); ph2 = np.arctan2(-B2, A2)
print(f"  poly:  {c0:.5f} + {c1:.5f}*x + {c2:.5f}*x^2 + {c3:.5f}*x^3")
print(f"  comp1: {A1:.5f}*cos({w1:.4f}x) + {B1:.5f}*sin({w1:.4f}x) -> amp={amp1:.4f} phase={ph1:.4f}")
print(f"  comp2: {A2:.5f}*cos({w2:.4f}x) + {B2:.5f}*sin({w2:.4f}x) -> amp={amp2:.4f} phase={ph2:.4f}")
