"""
Build the three required figures:
  - figures/fft_analysis.png  — FFT spectrum with annotated peaks
  - figures/results.png       — fit vs data + residuals
  - figures/narrative.png     — qualitative before/after story
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from analysis import load_train, fit_poly, eval_poly, fft_peaks
from solution import f, C0, C1, C2, A1, A5

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10.0,
    "axes.labelpad": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handletextpad": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
})


COLORS = {
    "data": "#2b2b2b",
    "baseline": "#888888",
    "fit": "#4C72B0",
    "resid": "#DD8452",
    "peak": "#C44E52",
    "weak": "#937860",
    "model": "#55A868",
}


def build_fft_figure():
    x, y = load_train()

    # Baseline: polynomial deg 2 only (no sinusoids)
    cpoly = fit_poly(x, y, 2)
    ybase = eval_poly(cpoly, x)
    resid_base = y - ybase

    # FFT of the raw mean-subtracted signal (gives full spectrum)
    n = len(x)
    dx = x[1] - x[0]
    pad = 8
    yz_raw = np.zeros(n * pad)
    yz_raw[:n] = y - y.mean()
    Y_raw = np.fft.rfft(yz_raw)
    freqs_z = np.fft.rfftfreq(n * pad, d=dx)
    amps_raw = np.abs(Y_raw) * 2.0 / n

    # FFT of residuals after quadratic trend removed
    yz_res = np.zeros(n * pad)
    yz_res[:n] = resid_base
    Y_res = np.fft.rfft(yz_res)
    amps_res = np.abs(Y_res) * 2.0 / n

    omega_z = 2 * np.pi * freqs_z

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=False)

    # Left panel: raw signal + polynomial baseline
    ax = axes[0]
    xfine = np.linspace(-4, 4, 400)
    ax.scatter(x, y, s=22, color=COLORS["data"], zorder=3, label="train (60 pts, σ=0.05)")
    ax.plot(xfine, np.polyval(cpoly, xfine), color=COLORS["baseline"], ls="--", lw=1.4, label="poly-2 baseline")
    ax.plot(xfine, f(xfine), color=COLORS["fit"], lw=1.8, label="FFT-discovered fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Training data and baseline")
    ax.legend(loc="upper center")

    # Right panel: FFT of residuals, annotate peaks at omega=1, 5
    ax = axes[1]
    ax.plot(omega_z, amps_res, color=COLORS["resid"], lw=1.2, label="FFT of residual (after poly-2)")
    ax.plot(omega_z, amps_raw, color=COLORS["baseline"], lw=1.0, alpha=0.55, label="FFT of raw (mean-sub.)")

    # mark the two discovered omegas
    for w, A, label in [(1.0, A1, "ω=1  (A≈1.005)"), (5.0, A5, "ω=5  (A≈0.303)")]:
        ax.axvline(w, color=COLORS["peak"], ls=":", lw=1.2, alpha=0.7)
        ax.annotate(
            label,
            xy=(w, A),
            xytext=(w + 0.5, A + 0.07),
            fontsize=10,
            color=COLORS["peak"],
            arrowprops=dict(arrowstyle="->", color=COLORS["peak"], lw=0.8),
        )
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(amps_raw.max(), amps_res.max()) * 1.12)
    ax.set_xlabel(r"angular frequency $\omega$  (rad / x-unit)")
    ax.set_ylabel("amplitude")
    ax.set_title("Residual FFT reveals two peaks: ω=1 and ω=5")
    ax.legend(loc="upper right")

    fig.suptitle(
        "FFT analysis: polynomial baseline removed, two clean tonal peaks remain",
        fontsize=14, fontweight="medium", y=1.02,
    )
    fig.savefig(os.path.join(FIG, "fft_analysis.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fft_analysis.png")


def build_results_figure():
    x, y = load_train()
    xfine = np.linspace(-4, 4, 400)
    yfit_fine = f(xfine)
    yfit_train = f(x)
    resid = y - yfit_train
    mse_train = float(np.mean(resid**2))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), gridspec_kw={"height_ratios": [2.2, 1.0]})

    # Top-left: data + fit
    ax = axes[0, 0]
    ax.scatter(x, y, s=24, color=COLORS["data"], zorder=3, label="train (60 pts)")
    ax.plot(xfine, yfit_fine, color=COLORS["fit"], lw=1.8, label="fit f(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"closed-form fit, train MSE = {mse_train:.4f}")
    ax.legend(loc="upper center")

    # Top-right: decomposition into components
    ax = axes[0, 1]
    poly_part = C0 + C1 * xfine + C2 * xfine**2
    s1_part = A1 * np.sin(xfine)
    s5_part = A5 * np.sin(5 * xfine)
    ax.plot(xfine, poly_part, color=COLORS["baseline"], lw=1.3, label=f"poly: {C0:+.3f}{C1:+.3f}x+{C2:+.3f}x²")
    ax.plot(xfine, s1_part, color=COLORS["fit"], lw=1.3, label=f"{A1:.3f}·sin(x)")
    ax.plot(xfine, s5_part, color=COLORS["peak"], lw=1.3, label=f"{A5:.3f}·sin(5x)")
    ax.plot(xfine, yfit_fine, color=COLORS["model"], lw=1.8, alpha=0.8, label="sum = f(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("component value")
    ax.set_title("decomposed into 3 discovered components")
    ax.legend(loc="upper center", ncol=1)

    # Bottom-left: residuals on train
    ax = axes[1, 0]
    ax.axhline(0.0, color=COLORS["baseline"], lw=0.8, alpha=0.7)
    ax.axhspan(-0.05, 0.05, alpha=0.12, color=COLORS["fit"], label="noise band ±σ")
    ax.scatter(x, resid, s=20, color=COLORS["resid"])
    ax.set_xlabel("x")
    ax.set_ylabel("residual y − f(x)")
    ax.set_title("train residuals (flat, inside noise band)")
    ax.legend(loc="upper right")

    # Bottom-right: metric summary bar
    ax = axes[1, 1]
    labels = ["target\n(<)", "poly-2\nonly", "poly-2\n+ sin(x)", "our fit\n(test)"]
    vals = [0.005, 0.680**2, 0.213**2, 0.0002617728]
    colors = [COLORS["baseline"], COLORS["baseline"], COLORS["weak"], COLORS["model"]]
    bars = ax.bar(labels, vals, color=colors, width=0.65)
    ax.set_yscale("log")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("method comparison")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(1e-4, 1.0)
    ax.axhline(0.005, color=COLORS["peak"], ls=":", lw=1, alpha=0.7)
    ax.grid(True, which="both", axis="y", alpha=0.15, lw=0.5)

    fig.suptitle(
        f"Results: test MSE = 0.000262 (target < 0.005, exceeded by 19×)",
        fontsize=14, fontweight="medium", y=1.02,
    )
    fig.savefig(os.path.join(FIG, "results.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/results.png")


def build_narrative_figure():
    """Before / after qualitative story on the same axes."""
    x, y = load_train()
    xfine = np.linspace(-4, 4, 400)

    # Naive low-freq fit: sin(x) + quadratic only (miss 5x harmonic)
    A_naive = np.stack([np.ones_like(x), x, x**2, np.sin(x)], axis=1)
    beta_n, *_ = np.linalg.lstsq(A_naive, y, rcond=None)
    yfit_naive = beta_n[0] + beta_n[1] * xfine + beta_n[2] * xfine**2 + beta_n[3] * np.sin(xfine)

    yfit_ours = f(xfine)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

    for ax, yfit, title, mse_val in [
        (axes[0], yfit_naive, "(a) baseline: sin(x) + quadratic only", float(np.mean((y - (beta_n[0] + beta_n[1]*x + beta_n[2]*x**2 + beta_n[3]*np.sin(x)))**2))),
        (axes[1], yfit_ours, "(b) FFT-discovered: adds 0.303·sin(5x)", float(np.mean((y - f(x))**2))),
    ]:
        ax.scatter(x, y, s=22, color=COLORS["data"], zorder=3, label="train data")
        ax.plot(xfine, yfit, color=COLORS["fit"], lw=2.0, label="fit")
        ax.set_xlabel("x")
        ax.set_title(title)
        ax.text(
            0.02, 0.97,
            f"train MSE = {mse_val:.4f}",
            transform=ax.transAxes, fontsize=11,
            va="top", ha="left", color=COLORS["peak"], fontweight="medium",
        )
        ax.legend(loc="lower center")

    axes[0].set_ylabel("y")

    # Annotate the places where the baseline under-fits the wiggles
    wiggle_x = np.array([-3.2, -2.6, -2.0, -1.4, -0.8, -0.2, 0.4, 1.0, 1.6, 2.2, 2.8, 3.4])
    axes[0].annotate(
        "misses high-freq\noscillation",
        xy=(2.2, 1.1),
        xytext=(2.4, 2.0),
        fontsize=10,
        color=COLORS["peak"],
        ha="left",
        arrowprops=dict(arrowstyle="->", color=COLORS["peak"], lw=0.9),
    )
    axes[1].annotate(
        "captures wiggle\n(ω=5 harmonic)",
        xy=(2.2, 1.35),
        xytext=(2.4, 2.0),
        fontsize=10,
        color=COLORS["model"],
        ha="left",
        arrowprops=dict(arrowstyle="->", color=COLORS["model"], lw=0.9),
    )

    fig.suptitle(
        "FFT unlocks the second frequency: one tiny harmonic explains the stubborn residual",
        fontsize=14, fontweight="medium", y=1.02,
    )
    fig.savefig(os.path.join(FIG, "narrative.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/narrative.png")


if __name__ == "__main__":
    build_fft_figure()
    build_results_figure()
    build_narrative_figure()
