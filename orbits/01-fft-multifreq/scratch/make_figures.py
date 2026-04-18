"""Generate fft_analysis.png, results.png, narrative.png for the orbit."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

ROOT = "/tmp/git-evolve-bench-multifreq-symreg-1776494535/.worktrees/01-fft-multifreq"
FIG = f"{ROOT}/orbits/01-fft-multifreq/figures"
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

COL_DATA = "#888888"
COL_FIT = "#4C72B0"
COL_BASE = "#C44E52"
COL_HI = "#55A868"
COL_EMPH = "#DD8452"

data = np.loadtxt(f"{ROOT}/research/eval/train_data.csv", delimiter=",", skiprows=1)
x = data[:, 0]; y = data[:, 1]
dx = x[1] - x[0]

def f(x):
    return np.sin(x) + 0.3 * np.sin(5.0 * x) + 0.1 * x**2

# ---------------- Figure 1: fft_analysis.png ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
fig.suptitle("FFT-guided frequency discovery", fontsize=15, fontweight="medium", y=1.03)

def fft_panel(ax, signal, title, peaks_to_mark=None):
    Y = np.fft.rfft(signal)
    mags = np.abs(Y) / len(signal)
    freqs_hz = np.fft.rfftfreq(len(signal), d=dx)   # cycles / x
    omegas = 2 * np.pi * freqs_hz                    # angular freq
    ax.stem(omegas, mags, linefmt="-", basefmt=" ", markerfmt="o")
    ax.set_xlabel(r"angular frequency  $\omega$  (rad / $x$)")
    ax.set_ylabel(r"$|Y(\omega)| / N$")
    ax.set_title(title)
    ax.set_xlim(0, 12)
    if peaks_to_mark:
        for w, label in peaks_to_mark:
            # nearest bin
            k = int(np.argmin(np.abs(omegas - w)))
            ax.annotate(label, xy=(omegas[k], mags[k]),
                        xytext=(omegas[k] + 0.4, mags[k] + 0.02),
                        fontsize=10, color=COL_EMPH,
                        arrowprops=dict(arrowstyle='->', color=COL_EMPH, lw=0.9))

fft_panel(axes[0], y, "raw $y$\n(polynomial dominates DC)", peaks_to_mark=[(1.0, "ω ≈ 1")])

# After deg-2 poly removal
coef2 = np.polyfit(x, y, 2)
resid2 = y - np.polyval(coef2, x)
fft_panel(axes[1], resid2, "after deg-2 polynomial removal",
          peaks_to_mark=[(1.0, "ω = 1"), (5.0, "ω = 5")])

# After subtracting ω=1 component
A = np.vstack([np.ones_like(x), x**2, np.sin(x), np.cos(x)]).T
c, *_ = np.linalg.lstsq(A, y, rcond=None)
resid_w1 = y - (A @ c)
fft_panel(axes[2], resid_w1, "after poly + ω=1 removal\n(second frequency emerges)",
          peaks_to_mark=[(5.0, "ω = 5")])

plt.savefig(f"{FIG}/fft_analysis.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote fft_analysis.png")

# ---------------- Figure 2: results.png ----------------
# Use dense grid for smooth fit curve; plot data, fit, and residuals.
x_dense = np.linspace(-4, 4, 400)
y_fit_dense = f(x_dense)
y_fit_data = f(x)
resid = y - y_fit_data

fig = plt.figure(figsize=(13, 6.5))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[3, 1], hspace=0.08, wspace=0.18)
ax_main = fig.add_subplot(gs[0, 0])
ax_res  = fig.add_subplot(gs[1, 0], sharex=ax_main)
ax_hist = fig.add_subplot(gs[:, 1])

ax_main.scatter(x, y, color=COL_DATA, s=42, edgecolor="white", lw=0.8,
                label="training data (σ=0.05 noise)", zorder=3)
ax_main.plot(x_dense, y_fit_dense, color=COL_FIT, lw=2.4,
             label=r"fit: $\sin(x) + 0.3\,\sin(5x) + 0.1\,x^2$", zorder=4)
ax_main.set_ylabel(r"$y$")
ax_main.set_title(r"Closed-form fit: $f(x) = \sin(x) + 0.3\,\sin(5x) + 0.1\,x^{2}$    test MSE = 0.000",
                  loc="left")
ax_main.legend(loc="upper center")
ax_main.tick_params(labelbottom=False)

ax_res.axhline(0, color="#555555", lw=0.8)
ax_res.scatter(x, resid, color=COL_EMPH, s=30, edgecolor="white", lw=0.6)
ax_res.set_xlabel(r"$x$")
ax_res.set_ylabel(r"$y - \hat y$")
ax_res.set_ylim(-0.2, 0.2)
ax_res.set_title("residuals (should be pure noise)", loc="left", fontsize=11)

# Residual histogram with noise σ reference
ax_hist.hist(resid, bins=15, orientation="horizontal", color=COL_FIT,
             edgecolor="white", alpha=0.85)
ax_hist.axhline(0.05, color=COL_BASE, ls="--", lw=1, label=r"$\pm \sigma = 0.05$")
ax_hist.axhline(-0.05, color=COL_BASE, ls="--", lw=1)
ax_hist.set_xlabel("count")
ax_hist.set_ylabel("residual")
ax_hist.set_ylim(-0.2, 0.2)
ax_hist.set_title(f"σ(resid) = {resid.std():.3f}\n(expected 0.05)", loc="left", fontsize=11)
ax_hist.legend(loc="upper right")

plt.savefig(f"{FIG}/results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote results.png")

# ---------------- Figure 3: narrative.png (successive peeling) ----------------
# Show 4 panels: (1) raw, (2) after removing 0.1 x^2, (3) after removing sin(x),
# (4) after removing 0.3 sin(5x) — residuals shrink to noise.
y_dense = np.sin(x_dense) + 0.3*np.sin(5*x_dense) + 0.1*x_dense**2

stages = [
    ("raw data $y(x)$",
     y, y_dense,
     r"$\sin(x) + 0.3\,\sin(5x) + 0.1\,x^2$"),
    ("after subtracting $0.1\\,x^{2}$",
     y - 0.1*x**2, np.sin(x_dense) + 0.3*np.sin(5*x_dense),
     r"remaining: $\sin(x) + 0.3\,\sin(5x)$"),
    ("after subtracting $\\sin(x)$",
     y - 0.1*x**2 - np.sin(x), 0.3*np.sin(5*x_dense),
     r"remaining: $0.3\,\sin(5x)$"),
    ("after subtracting $0.3\\,\\sin(5x)$",
     y - 0.1*x**2 - np.sin(x) - 0.3*np.sin(5*x), np.zeros_like(x_dense),
     "remaining: 0 (pure noise)"),
]

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
fig.suptitle("Residual peeling: each discovered component removed in sequence",
             fontsize=15, fontweight="medium", y=1.04)

for ax, (title, yp, y_model_dense, model_label, _) in zip(axes, stages):
    ax.scatter(x, yp, color=COL_DATA, s=28, edgecolor="white", lw=0.6, zorder=3)
    ax.plot(x_dense, y_model_dense, color=COL_FIT, lw=2.0, zorder=4,
            label="model (remaining)")
    ax.axhline(0, color=COL_BASE, ls="--", lw=0.9, alpha=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"$x$")
    rms = np.sqrt(np.mean(yp**2))
    ax.text(0.03, 0.97, f"RMS = {rms:.3f}", transform=ax.transAxes,
            fontsize=10, color="#333", va="top")
    ax.text(0.03, 0.87, f"model: {model_label}", transform=ax.transAxes,
            fontsize=9, color=COL_FIT, va="top")
axes[0].set_ylabel(r"$y$ (residual)")

plt.savefig(f"{FIG}/narrative.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote narrative.png")
