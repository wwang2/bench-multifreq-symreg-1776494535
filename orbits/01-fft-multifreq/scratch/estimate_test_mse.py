"""Estimate expected test MSE of each candidate model, assuming true function
is one of the clean closed forms. Compute statistical uncertainty on fitted
coefficients given noise σ=0.05."""
import numpy as np

data = np.loadtxt(
    "/tmp/git-evolve-bench-multifreq-symreg-1776494535/.worktrees/01-fft-multifreq/research/eval/train_data.csv",
    delimiter=",", skiprows=1
)
x = data[:, 0]; y = data[:, 1]
N = len(x)

# Basis: [1, x^2, sin(x), sin(5x)]
A = np.vstack([np.ones_like(x), x**2, np.sin(x), np.sin(5*x)]).T
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
pred = A @ coef
resid_var = np.var(y - pred, ddof=A.shape[1])
print(f"coef (lstsq) = {coef}")
print(f"Residual variance estimate = {resid_var:.6f}  (noise σ² expected 0.0025)")
print(f"sqrt = {np.sqrt(resid_var):.5f}  (noise σ expected 0.05)")

# Covariance of coefficients: σ² (A^T A)^{-1}
AtA_inv = np.linalg.inv(A.T @ A)
cov = resid_var * AtA_inv
se = np.sqrt(np.diag(cov))
print(f"\nCoefficient standard errors:")
for name, c, s in zip(["const", "x^2", "sin(x)", "sin(5x)"], coef, se):
    print(f"  {name:8s}: {c:+.5f} ± {s:.5f}")

# Now simulate: if we commit to the rounded model, expected squared bias vs noise:
# True (hypothesis): y_true = -0 + 0.1 x^2 + sin(x) + 0.3 sin(5x)  (offset could be 0 or 0)
# Actually we don't know the true offset; the fitted offset is -0.023.

# On a dense test grid, what's the bias if we commit to rounded vs fit?
x_test = np.linspace(-4, 4, 400)
def f_rounded(x):
    return np.sin(x) + 0.3*np.sin(5*x) + 0.1*x**2

def f_fitted(x):
    c0, c2, a1, a5 = coef
    return c0 + c2*x**2 + a1*np.sin(x) + a5*np.sin(5*x)

# The test MSE we'll achieve depends on how close our f is to the true generator.
# Let's compute ||f_rounded - f_fitted||^2 over test grid (approximation of disagreement)
diff = f_rounded(x_test) - f_fitted(x_test)
print(f"\n||rounded - fitted||² on test grid = {np.mean(diff**2):.6f}")

# If the TRUE model is the rounded version, test MSE of fitted ≈ this.
# If TRUE is the fitted version, test MSE of rounded ≈ this.
# With σ=0.05 noise, coefficient errors give a predictable error floor.

# Let's also check what happens without the constant offset at all
print("\n-- Candidates with train MSE and test MSE under assumption that TRUE = rounded --")
# Hypothesis 1: TRUE = sin(x) + 0.3 sin(5x) + 0.1 x^2 (no offset)
true_clean_1 = np.sin(x_test) + 0.3*np.sin(5*x_test) + 0.1*x_test**2
# Hypothesis 2: TRUE = sin(x) + 0.3 sin(5x) + 0.1 x^2 + small offset
for candidate_name, candidate in [
    ("lstsq 4-param",    f_fitted(x_test)),
    ("sin(x)+0.3sin5x+0.1x^2", np.sin(x_test)+0.3*np.sin(5*x_test)+0.1*x_test**2),
    ("... + coef offset", np.sin(x_test)+0.3*np.sin(5*x_test)+0.1*x_test**2 + coef[0]),
]:
    test_mse = np.mean((candidate - true_clean_1)**2)
    print(f"  {candidate_name:30s}  MSE vs TRUE_clean_1 = {test_mse:.6f}")

# Let's write out the exact lstsq fit and also two bolder rounded alternatives.
# The safest bet: commit to the lstsq coefficients. They're closed-form given FFT basis.
print("\nCommitted closed form coefficients (lstsq on FFT-selected basis):")
for name, c in zip(["const", "x^2", "sin(x)", "sin(5x)"], coef):
    print(f"  {name}: {c:.6f}")

# Check: what if we also include cos(x) and cos(5x)? They are small. Including them might actually hurt on clean test.
A2 = np.vstack([np.ones_like(x), x**2, np.sin(x), np.cos(x), np.sin(5*x), np.cos(5*x)]).T
coef2, *_ = np.linalg.lstsq(A2, y, rcond=None)
print(f"\n6-param fit: {coef2}")
print(f"Train MSE 6-param: {np.mean((A2 @ coef2 - y)**2):.6f}")
