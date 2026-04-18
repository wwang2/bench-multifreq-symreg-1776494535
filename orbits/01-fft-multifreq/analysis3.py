"""
Pin down the final closed form.
Hypothesis: y = a0 + a1*x + a2*x^2 + A1*sin(x) + B1*cos(x) + A2*sin(5x) + B2*cos(5x)

with A1 ~ 1.0, B1 ~ 0, A2 ~ 0.3, B2 ~ 0.

We solve once via lstsq on train to get final numeric coefficients, then
sanity-check by dropping the ≈0 terms.
"""
import numpy as np
from analysis import load_train, solve_multifreq


def fit_final(x, y, omegas, poly_deg):
    cols = []
    for p in range(poly_deg, -1, -1):
        cols.append(x**p)
    for w in omegas:
        cols.append(np.cos(w * x))
        cols.append(np.sin(w * x))
    A = np.stack(cols, axis=1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    rms = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return beta, rms


if __name__ == "__main__":
    x, y = load_train()

    # Final candidate: poly_deg 2 + sin/cos at omega=1 and omega=5
    beta, rms = fit_final(x, y, [1.0, 5.0], poly_deg=2)
    print(f"Full fit rms = {rms:.5f}")
    print(f"poly[x^2,x,1]={beta[:3]}")
    print(f"cos(x), sin(x), cos(5x), sin(5x) = {beta[3:]}")

    # Reconstruct closed form:
    a2, a1, a0 = beta[0], beta[1], beta[2]
    c1, s1, c5, s5 = beta[3], beta[4], beta[5], beta[6]
    print(f"\nClosed form:")
    print(f"  y = {a0:+.4f} + {a1:+.4f}*x + {a2:+.4f}*x^2")
    print(f"      + {c1:+.4f}*cos(x) + {s1:+.4f}*sin(x)")
    print(f"      + {c5:+.4f}*cos(5x) + {s5:+.4f}*sin(5x)")

    # As amplitude/phase:
    A1 = np.hypot(c1, s1); phi1 = np.arctan2(s1, c1)
    A5 = np.hypot(c5, s5); phi5 = np.arctan2(s5, c5)
    print(f"\nAmp/phase form:")
    print(f"  A1={A1:.4f}, phi1={phi1:+.4f}  → cos(x - {phi1:+.4f})")
    print(f"  A5={A5:.4f}, phi5={phi5:+.4f}  → cos(5x - {phi5:+.4f})")

    # Try leaving only the 'big' basis functions (set tiny coefs to zero)
    # keep poly terms and sin components only (since cos coefficients are tiny)
    cols = np.stack([x**2, x**1, np.ones_like(x), np.sin(x), np.sin(5 * x)], axis=1)
    beta2, *_ = np.linalg.lstsq(cols, y, rcond=None)
    yhat2 = cols @ beta2
    rms2 = np.sqrt(np.mean((y - yhat2) ** 2))
    print(f"\nReduced (no cos) fit rms = {rms2:.5f}")
    print(f"  coefs: {beta2}")

    # Add back x*sin(x) etc? Probably not — we're close to noise floor
    # Try the very lean (no x-linear) model:
    cols = np.stack([x**2, np.ones_like(x), np.sin(x), np.sin(5 * x)], axis=1)
    beta3, *_ = np.linalg.lstsq(cols, y, rcond=None)
    yhat3 = cols @ beta3
    rms3 = np.sqrt(np.mean((y - yhat3) ** 2))
    print(f"\nLean (a*x^2 + c + sin(x) + B*sin(5x)) fit rms = {rms3:.5f}")
    print(f"  coefs (x^2, const, sin(x), sin(5x)): {beta3}")

    # Also try adding a cos(x) small coefficient
    cols = np.stack([x**2, x**1, np.ones_like(x), np.sin(x), np.cos(x), np.sin(5 * x), np.cos(5 * x)], axis=1)
    beta4, *_ = np.linalg.lstsq(cols, y, rcond=None)
    yhat4 = cols @ beta4
    rms4 = np.sqrt(np.mean((y - yhat4) ** 2))
    print(f"\nFull 7-term fit rms = {rms4:.5f}")
    print(f"  coefs: {beta4}")
