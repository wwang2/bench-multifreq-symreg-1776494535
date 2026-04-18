"""Narrow refinement around ω1≈1, ω2≈5 with poly deg=2."""
import numpy as np

data = np.loadtxt(
    "/tmp/git-evolve-bench-multifreq-symreg-1776494535/.worktrees/01-fft-multifreq/research/eval/train_data.csv",
    delimiter=",", skiprows=1
)
x = data[:, 0]
y = data[:, 1]

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

# Narrow refinement
print("-- narrow joint scan (ω1, ω2), poly deg=2 --")
omegas1 = np.linspace(0.90, 1.10, 201)
omegas2 = np.linspace(4.85, 5.15, 201)
best_mse = np.inf
best = None
for w1 in omegas1:
    for w2 in omegas2:
        mse, coef = fit(x, y, [w1, w2], deg=2)
        if mse < best_mse:
            best_mse = mse
            best = (w1, w2, coef)
w1, w2, coef = best
print(f"best: ω1={w1:.6f}  ω2={w2:.6f}  MSE={best_mse:.6f}")
c0, c1, c2, A1, B1, A2, B2 = coef
amp1 = np.hypot(A1, B1); ph1 = np.arctan2(-B1, A1)
amp2 = np.hypot(A2, B2); ph2 = np.arctan2(-B2, A2)
print(f"  poly:  {c0:.6f} + {c1:.6f}*x + {c2:.6f}*x^2")
print(f"  comp1: A={A1:.6f}  B={B1:.6f}  -> amp={amp1:.6f}  phase={ph1:.6f}")
print(f"         -> {amp1:.4f}*cos({w1:.4f}*x + {ph1:.4f})")
print(f"  comp2: A={A2:.6f}  B={B2:.6f}  -> amp={amp2:.6f}  phase={ph2:.6f}")
print(f"         -> {amp2:.4f}*cos({w2:.4f}*x + {ph2:.4f})")

# What if ω1 and ω2 are exactly 1.0 and 5.0?
mse_int, coef_int = fit(x, y, [1.0, 5.0], deg=2)
print(f"\nWith exact ω1=1.0, ω2=5.0, deg=2:  MSE={mse_int:.6f}")
print(f"coef = {coef_int}")
c0, c1, c2, A1, B1, A2, B2 = coef_int
amp1 = np.hypot(A1, B1); ph1 = np.arctan2(-B1, A1)
amp2 = np.hypot(A2, B2); ph2 = np.arctan2(-B2, A2)
print(f"  poly:  {c0:.6f} + {c1:.6f}*x + {c2:.6f}*x^2")
print(f"  comp1 (ω=1): A={A1:.6f}  B={B1:.6f}  -> amp={amp1:.4f}  phase={ph1:.4f}")
print(f"  comp2 (ω=5): A={A2:.6f}  B={B2:.6f}  -> amp={amp2:.4f}  phase={ph2:.4f}")

# Also try deg=3 and deg=1 backgrounds
for deg in [1, 2, 3, 4]:
    mse_int, coef_int = fit(x, y, [1.0, 5.0], deg=deg)
    print(f"\nWith exact ω1=1.0, ω2=5.0, poly deg={deg}:  MSE={mse_int:.6f}  coef={coef_int}")

# Estimate the true noise floor: we expect σ=0.05 -> noise MSE contribution ~0.0025.
# So training MSE should saturate near 0.0025 when the signal is perfectly modeled.
print(f"\nExpected noise floor (σ=0.05): {0.05**2:.6f}")

# Try adding a THIRD frequency just in case
print("\n-- search for third freq given ω=1, ω=5 are fixed, poly deg=2 --")
omegas3 = np.linspace(0.2, 10.0, 500)
best = []
for w3 in omegas3:
    mse, _ = fit(x, y, [1.0, 5.0, w3], deg=2)
    best.append((mse, w3))
best.sort()
for mse, w3 in best[:10]:
    print(f"  ω3={w3:.4f}  MSE={mse:.6f}")
