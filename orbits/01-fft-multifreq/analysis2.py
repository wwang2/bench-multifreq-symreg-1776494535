"""
Refinement 2: try a finer grid search + fit with different poly degrees
to identify the cleanest closed-form.
"""
import numpy as np
from analysis import load_train, solve_multifreq, refine_omegas


def brute_grid_search_3(x, y, w1_range, w2_range, w3_range, poly_deg=2):
    """Coarse brute-force over 3 omegas to find global minimum around our FFT peaks."""
    best = (None, np.inf)
    for w1 in w1_range:
        for w2 in w2_range:
            for w3 in w3_range:
                _, _, rms = solve_multifreq(x, y, [w1, w2, w3], poly_deg=poly_deg)
                if rms < best[1]:
                    best = ([w1, w2, w3], rms)
    return best


def fit_and_report(x, y, omegas, poly_deg, label=""):
    poly_coef, ab, rms = solve_multifreq(x, y, omegas, poly_deg=poly_deg)
    print(f"[{label}] omegas={omegas} poly_deg={poly_deg} rms={rms:.5f}")
    # Express each (a,b) as A cos(wx - phi)
    for i, ((a, b), w) in enumerate(zip(ab, omegas)):
        A = np.hypot(a, b)
        phi = np.arctan2(b, a)
        print(f"   #{i}: A={A:.4f} omega={w:.5f} phi={phi:+.4f}  (a={a:+.4f},b={b:+.4f})")
    print(f"   poly coefs (high→low): {poly_coef}")
    return poly_coef, ab, rms


if __name__ == "__main__":
    x, y = load_train()

    # We know noise std = 0.05 so rms floor ≈ 0.05.
    # Anything well below ~0.04 is suspicious (over-fitting noise on 60 points).

    # Start fresh: do a more aggressive global search via finer FFT
    # by zero-padding
    dx = x[1] - x[0]
    yfft = y - y.mean()
    pad = 8
    yz = np.zeros(len(x) * pad)
    yz[: len(x)] = yfft
    Y = np.fft.rfft(yz)
    freqs_z = np.fft.rfftfreq(len(yz), d=dx)
    amps_z = np.abs(Y) * 2.0 / len(x)
    order = np.argsort(-amps_z[1:]) + 1
    print("Zero-padded FFT top peaks (omega):")
    seen_omega = []
    for k in order[:25]:
        w = 2 * np.pi * freqs_z[k]
        # merge nearby peaks
        if any(abs(w - s) < 0.1 for s in seen_omega):
            continue
        seen_omega.append(w)
        print(f"   omega={w:.4f}  amp={amps_z[k]:.4f}")
        if len(seen_omega) >= 8:
            break

    # Try 3-freq: wide search around top FFT peaks
    # Based on FFT: omega1 ~ 0.8-1.3, omega2 ~ 1.5-1.9, omega3 ~ 4.7-5.2
    print("\nBrute-force 3-omega coarse grid ...")
    w1_range = np.linspace(0.5, 1.5, 21)
    w2_range = np.linspace(1.5, 2.5, 21)
    w3_range = np.linspace(4.5, 5.5, 21)
    best, rms_best = brute_grid_search_3(x, y, w1_range, w2_range, w3_range, poly_deg=2)
    print(f"  best coarse: omegas={best} rms={rms_best:.5f}")
    om_refined = refine_omegas(x, y, best, poly_deg=2, grid_halfwidth=0.05, n_grid=41, rounds=4)
    fit_and_report(x, y, om_refined, poly_deg=2, label="3-freq refined (p2)")

    # Try 3-freq with poly_deg=3 and 4
    for pd in [1, 2, 3, 4]:
        om = refine_omegas(x, y, best, poly_deg=pd, grid_halfwidth=0.1, n_grid=31, rounds=4)
        fit_and_report(x, y, om, poly_deg=pd, label=f"3-freq, poly_deg={pd}")

    # Try 4-freq — guard against overfit (noise σ=0.05 ⇒ expect rms≈0.05 on train)
    print("\n4-freq attempt:")
    for pd in [2]:
        # coarse search starting with 4 known-good peaks
        initial = [1.04, 1.78, 5.00, 3.10]  # add a 4th guess
        om4 = refine_omegas(x, y, initial, poly_deg=pd, grid_halfwidth=0.3, n_grid=61, rounds=5)
        fit_and_report(x, y, om4, poly_deg=pd, label=f"4-freq, poly_deg={pd}")

    # Try integer-ish omegas to see if signal is harmonic: omega = 1, 2, 5 ?
    print("\nInteger-ish omega hypotheses (2-freq and 3-freq):")
    for om in ([1.0], [1.0, 2.0], [1.0, 2.0, 5.0], [1.0, 5.0], [np.pi / 3]):
        fit_and_report(x, y, om, poly_deg=2, label=f"hyp omegas={om}")
