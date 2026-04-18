"""
One more ablation: try nice round-number coefficients.
Maybe the 'true' signal is y = x^2/10 + sin(x) + 0.3*sin(5x) ?
Or y = sin(x) + 0.3 sin(5x) + 0.1 x^2 ?
Check each against the evaluator later.
"""
import numpy as np
from analysis import load_train


def rms(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


if __name__ == "__main__":
    x, y = load_train()

    cands = {
        "current (lstsq 5-term)": lambda x: -0.02304 - 0.00070 * x + 0.10278 * x**2 + 1.00542 * np.sin(x) + 0.30332 * np.sin(5 * x),
        "integer coefs [sin(x) + 0.3 sin(5x) + 0.1 x^2]": lambda x: np.sin(x) + 0.3 * np.sin(5 * x) + 0.1 * x**2,
        "plus tiny offset": lambda x: np.sin(x) + 0.3 * np.sin(5 * x) + 0.1 * x**2 - 0.02,
        "no offset no linear, best 3 coef": lambda x: 1.005 * np.sin(x) + 0.303 * np.sin(5 * x) + 0.103 * x**2 - 0.023,
    }
    for name, g in cands.items():
        yhat = g(x)
        print(f"{name:55s} train rms = {rms(y, yhat):.5f}")

    # Try subset-fit only with the three 'believed' basis functions:
    # [x^2, sin(x), sin(5x), 1]
    A = np.stack([x**2, np.sin(x), np.sin(5 * x), np.ones_like(x)], axis=1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    print(f"\n4-term (x^2, sin(x), sin(5x), 1) rms = {rms(y, yhat):.5f}")
    print(f"   coefs: x^2={beta[0]:.5f}  sin(x)={beta[1]:.5f}  sin(5x)={beta[2]:.5f}  c={beta[3]:.5f}")
