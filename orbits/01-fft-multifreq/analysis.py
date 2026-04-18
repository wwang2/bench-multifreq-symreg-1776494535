"""
FFT-based exploration of the training data to find:
  1. polynomial (smooth) trend
  2. sinusoidal components (multiple frequencies)

We:
  - load train_data.csv
  - fit a low-order polynomial baseline
  - FFT residuals (real-valued, so numpy rfft is appropriate)
  - find peaks
  - then solve amplitude/phase for each candidate (freq, phase) pair via lstsq
"""
import numpy as np
import os
import csv

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(ROOT, "..", "..", "research", "eval", "train_data.csv"))


def load_train():
    x, y = [], []
    with open(DATA, "r") as fh:
        r = csv.reader(fh)
        next(r)
        for row in r:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return np.array(x), np.array(y)


def fit_poly(x, y, deg):
    """Fit polynomial of degree deg via lstsq (allowed)."""
    X = np.vander(x, deg + 1)  # columns: x^deg ... x^0
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def eval_poly(coef, x):
    return np.polyval(coef, x)


def fft_peaks(x, y_resid, top_k=8):
    """
    Real FFT of residuals — data is evenly spaced in x so we can use rfft.
    Returns list of (freq_cycles_per_xunit, angular_freq_omega, amplitude) sorted by amplitude desc.
    """
    n = len(x)
    dx = x[1] - x[0]
    # rfft only returns non-negative frequencies (good for real signal)
    Y = np.fft.rfft(y_resid)
    freqs = np.fft.rfftfreq(n, d=dx)  # cycles per x unit
    amps = np.abs(Y) * 2.0 / n  # 2-sided amplitude
    amps[0] = np.abs(Y[0]) / n  # DC doesn't double
    # Sort by amplitude, exclude DC
    order = np.argsort(-amps[1:]) + 1
    peaks = []
    for k in order[:top_k]:
        peaks.append(
            (float(freqs[k]), float(2 * np.pi * freqs[k]), float(amps[k]), float(np.angle(Y[k])))
        )
    return peaks, freqs, amps


def solve_multifreq(x, y, omegas, poly_deg=2):
    """
    Solve the linear system:
        y ~ sum(a_i * cos(omega_i * x) + b_i * sin(omega_i * x)) + poly(x)
    via lstsq. Returns (poly_coef, list of (a_i, b_i), residual RMS).
    """
    cols = []
    # polynomial columns (x^poly_deg ... x^0)
    for p in range(poly_deg, -1, -1):
        cols.append(x**p)
    for w in omegas:
        cols.append(np.cos(w * x))
        cols.append(np.sin(w * x))
    A = np.stack(cols, axis=1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    rms = float(np.sqrt(np.mean((y - yhat) ** 2)))
    poly_coef = beta[: poly_deg + 1]
    ab = []
    idx = poly_deg + 1
    for _ in omegas:
        ab.append((float(beta[idx]), float(beta[idx + 1])))
        idx += 2
    return poly_coef, ab, rms


def refine_omegas(x, y, omegas_init, poly_deg, grid_halfwidth=0.15, n_grid=15, rounds=3):
    """Coordinate-descent local grid search on each omega to reduce RMS."""
    omegas = list(omegas_init)
    for _ in range(rounds):
        improved = False
        for i in range(len(omegas)):
            w0 = omegas[i]
            best = (w0, None)
            for w in np.linspace(w0 - grid_halfwidth, w0 + grid_halfwidth, n_grid):
                test = omegas.copy()
                test[i] = float(w)
                _, _, rms = solve_multifreq(x, y, test, poly_deg=poly_deg)
                if best[1] is None or rms < best[1]:
                    best = (float(w), rms)
            if best[0] != w0:
                omegas[i] = best[0]
                improved = True
        grid_halfwidth *= 0.3
        if not improved:
            break
    return omegas


if __name__ == "__main__":
    x, y = load_train()
    print(f"n={len(x)}  x in [{x.min():.3f}, {x.max():.3f}]  y in [{y.min():.3f}, {y.max():.3f}]")

    # 1) Fit smooth quadratic trend, inspect residuals
    for deg in [0, 1, 2, 3, 4]:
        c = fit_poly(x, y, deg)
        yhat = eval_poly(c, x)
        rms = np.sqrt(np.mean((y - yhat) ** 2))
        print(f"poly deg {deg}: rms={rms:.4f}")

    # 2) FFT residuals after poly deg 2
    deg = 2
    c2 = fit_poly(x, y, deg)
    resid = y - eval_poly(c2, x)
    peaks, freqs, amps = fft_peaks(x, resid, top_k=10)
    print("\nTop FFT peaks (freq[cyc/xunit], omega[rad/xunit], amp, phase):")
    for fr, w, a, p in peaks:
        print(f"  f={fr:.4f}  omega={w:.4f}  amp={a:.4f}  phase={p:+.3f}")

    # 3) Joint solve with top-2 FFT peaks + quadratic trend
    omegas_init = [peaks[0][1], peaks[1][1]]
    poly_coef, ab, rms = solve_multifreq(x, y, omegas_init, poly_deg=2)
    print(f"\nInitial 2-freq fit: rms={rms:.4f}")
    print(f"  omegas: {omegas_init}")
    print(f"  poly (deg 2→0): {poly_coef}")
    print(f"  ab: {ab}")

    # 4) Refine omegas
    omegas_refined = refine_omegas(x, y, omegas_init, poly_deg=2, grid_halfwidth=0.3, n_grid=61, rounds=4)
    poly_coef_r, ab_r, rms_r = solve_multifreq(x, y, omegas_refined, poly_deg=2)
    print(f"\nRefined 2-freq fit: rms={rms_r:.4f}")
    print(f"  omegas refined: {omegas_refined}")
    print(f"  poly (deg 2→0): {poly_coef_r}")
    print(f"  ab: {ab_r}")

    # 5) Try 3 frequencies
    omegas3_init = [peaks[0][1], peaks[1][1], peaks[2][1]]
    omegas3_refined = refine_omegas(x, y, omegas3_init, poly_deg=2, grid_halfwidth=0.3, n_grid=61, rounds=4)
    poly_coef_3, ab_3, rms_3 = solve_multifreq(x, y, omegas3_refined, poly_deg=2)
    print(f"\nRefined 3-freq fit: rms={rms_3:.4f}")
    print(f"  omegas refined: {omegas3_refined}")
    print(f"  poly (deg 2→0): {poly_coef_3}")
    print(f"  ab: {ab_3}")

    # 6) Convert (a,b) to (A, phi) form: a*cos + b*sin = A*cos(wx - phi)
    print("\n2-freq expression (A cos(w x - phi) form):")
    for (a, b), w in zip(ab_r, omegas_refined):
        A = float(np.hypot(a, b))
        phi = float(np.arctan2(b, a))
        print(f"  A={A:.4f}  omega={w:.4f}  phi={phi:+.4f}")
