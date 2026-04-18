"""Check candidate closed-forms on training data."""
import numpy as np

data = np.loadtxt(
    "/tmp/git-evolve-bench-multifreq-symreg-1776494535/.worktrees/01-fft-multifreq/research/eval/train_data.csv",
    delimiter=",", skiprows=1
)
x = data[:, 0]; y = data[:, 1]

def test(name, f):
    p = f(x)
    mse = np.mean((y - p) ** 2)
    print(f"{name:40s}  train_MSE = {mse:.6f}")

# Closed-form candidates
test("sin(x) + 0.3 sin(5x)", lambda x: np.sin(x) + 0.3*np.sin(5*x))
test("sin(x) + 0.3 sin(5x) + 0.1 x^2", lambda x: np.sin(x) + 0.3*np.sin(5*x) + 0.1*x**2)
test("sin(x) + 0.3 sin(5x) + 0.105 x^2", lambda x: np.sin(x) + 0.3*np.sin(5*x) + 0.105*x**2)
test("sin(x) + 0.3 sin(5x) + 0.105 x^2 - 0.03", lambda x: np.sin(x) + 0.3*np.sin(5*x) + 0.105*x**2 - 0.03)
test("sin(x) + 0.3 sin(5x) + x^2/10", lambda x: np.sin(x) + 0.3*np.sin(5*x) + x**2/10)
test("sin(x) + 0.3 sin(5x) + 0.105*(x^2 - C)", lambda x: np.sin(x) + 0.3*np.sin(5*x) + 0.105*(x**2 - 0.3))

# Fit only the amp/offset, keeping freqs and x^2 structure
def design(x):
    return np.vstack([np.ones_like(x), x, x**2, np.sin(x), np.cos(x), np.sin(5*x), np.cos(5*x)]).T

A = design(x)
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
pred = A @ coef
print(f"\nFull fit with basis [1, x, x^2, sin(x), cos(x), sin(5x), cos(5x)]")
print(f"  coef = {coef}")
print(f"  train MSE = {np.mean((y - pred)**2):.6f}")

# Same but without linear term (suspected zero)
def design2(x):
    return np.vstack([np.ones_like(x), x**2, np.sin(x), np.cos(x), np.sin(5*x), np.cos(5*x)]).T

A = design2(x)
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
pred = A @ coef
print(f"\nFull fit with basis [1, x^2, sin(x), cos(x), sin(5x), cos(5x)]")
print(f"  coef = {coef}")
print(f"  train MSE = {np.mean((y - pred)**2):.6f}")

# Try only sin (no cos)
def design3(x):
    return np.vstack([np.ones_like(x), x**2, np.sin(x), np.sin(5*x)]).T

A = design3(x)
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
pred = A @ coef
print(f"\nFit with basis [1, x^2, sin(x), sin(5x)] (sin-only + even poly)")
print(f"  coef = {coef}")
print(f"  train MSE = {np.mean((y - pred)**2):.6f}")
