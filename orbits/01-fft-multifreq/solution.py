"""
orbit/01-fft-multifreq.r1 — FFT-based symbolic regression.

Approach (independent replica r1):

1) Fit smooth polynomial trend to the 60 training points.
2) Take FFT of the residuals to expose narrow-band structure.
3) Two dominant frequencies fell out of a zero-padded FFT + brute-force
   3-omega refinement: omega ≈ 1.0 and omega ≈ 5.0.  Both carry phase
   close to +pi/2 (pure-sine).
4) Hypothesis: the underlying noise-free signal is

      y(x) = c0 + c1 x + c2 x^2 + A1 sin(x) + A5 sin(5 x)

   with the cos(x), cos(5x), and linear-x coefficients all statistically
   indistinguishable from zero on train (|coef| well below noise floor).

5) Fit c0, c1, c2, A1, A5 once via closed-form numpy.linalg.lstsq on
   the training data (allowed — linear in the unknowns once the basis
   is fixed).  Bake the numeric coefficients into the closed form.

Train RMS at these coefficients ~= 0.043, which is flush with the
additive noise std (sigma = 0.05) — i.e. we have explained essentially
all of the deterministic signal.
"""

import numpy as np


# Closed-form coefficients locked in after FFT discovery.
# These come from numpy.linalg.lstsq on the 60 training points,
# fitting [1, x, x^2, sin(x), sin(5x)] — no iterative optimization,
# just one-shot linear algebra on a fixed basis.
C0 = -0.02303991369753  # constant
C1 = -0.00070483289662  # linear (negligible, kept for symmetry)
C2 = 0.10277599471247   # quadratic
A1 = 1.00541989330001   # amplitude of sin(x)
A5 = 0.30331892810807   # amplitude of sin(5 x)


def f(x: np.ndarray) -> np.ndarray:
    """
    Closed-form reconstruction of the noise-free signal discovered by
    FFT analysis of the training residuals.

    y(x) = C0 + C1*x + C2*x^2 + A1*sin(x) + A5*sin(5*x)
    """
    x = np.asarray(x, dtype=float)
    return C0 + C1 * x + C2 * x * x + A1 * np.sin(x) + A5 * np.sin(5.0 * x)
