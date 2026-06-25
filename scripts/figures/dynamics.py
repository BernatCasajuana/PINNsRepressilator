"""
Two-panel dynamics figure — no-noise PINN reconstruction.

Panel A: Oscillatory  β=5, n=3,   σ=0
Panel B: Stable       β=5, n=1.5, σ=0

Runs are cached under results/dynamics_figure/ the first time and reused
on subsequent calls (delete that directory to force a rerun).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.utils import make_synthetic_dataset, set_global_seed, finalize_figure
from pinns.inverse import run_inverse

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.useoffset": False,
})

FIGURE_PATH            = "figures/dynamics.png"
FIGURE_PATH_OSC        = "figures/dynamics_oscillatory.png"
OUTDIR_BASE            = "results/dynamics_figure/runs"
SEED         = 0
COLORS       = ("#0072B2", "#E69F00", "#009E73")
REPRESSORS   = ["Repressor 1", "Repressor 2", "Repressor 3"]

CASES = [
    {"beta": 5.0, "n": 3.0, "panel": "A", "title": r"Oscillatory ($\beta=5,\,n=3,\,\sigma=0$)"},
    {"beta": 5.0, "n": 1.5, "panel": "B", "title": r"Stable ($\beta=5,\,n=1.5,\,\sigma=0$)"},
]


def _run_or_load(case, seed):
    """Return (t, y_true, y_pred) — runs the PINN on first call, then loads from cache."""
    cache_dir = os.path.join(
        OUTDIR_BASE,
        f"beta{case['beta']}_n{case['n']}_noise0.0",
        "obs-1-2-3_count-1000",
        f"seed-{seed}",
    )
    cache_file = os.path.join(cache_dir, "trajectory_cache.npz")

    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["t"], data["y_true"], data["y_pred"]

    set_global_seed(seed)
    dataset = make_synthetic_dataset(beta=case["beta"], n=case["n"], noise_level=0.0, seed=seed)
    result = run_inverse(
        dataset_path=dataset,
        outdir_base=OUTDIR_BASE,
        beta_guess=4.0,
        n_guess=2.5,
        observation_stride=1,
        observed_components=[0, 1, 2],
        train_iterations=10000,
        random_seed=seed,
        save_checkpoint=True,
    )
    t      = dataset["t"]
    y_true = dataset["y_clean"]
    y_pred = result["y_pred"]

    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_file, t=t, y_true=y_true, y_pred=y_pred)
    return t, y_true, y_pred


def _plot_panel(ax, t, y_true, y_pred, case, xlim=None, show_panel=True, title=None):
    t_flat = t.flatten()
    for i in range(3):
        ax.plot(t_flat, y_true[:, i], "-",  color=COLORS[i], label=f"{REPRESSORS[i]} (Data)")
        ax.plot(t_flat, y_pred[:, i], "--", color=COLORS[i], label=f"{REPRESSORS[i]} (PINN)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Protein Concentration")
    ax.set_title(title if title is not None else case["title"])
    ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)
    if show_panel:
        ax.text(-0.02, 1.04, case["panel"], transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="bottom", ha="left",
                color="black", clip_on=False)


def main():
    os.makedirs("figures", exist_ok=True)
    _, axes = plt.subplots(2, 1, figsize=(12, 9))

    for ax, case in zip(axes, CASES):
        t, y_true, y_pred = _run_or_load(case, SEED)
        _plot_panel(ax, t, y_true, y_pred, case)

    finalize_figure(FIGURE_PATH)
    print(f"Saved: {FIGURE_PATH}")


def main_oscillatory():
    """Single-panel figure for the oscillatory case only."""
    os.makedirs("figures", exist_ok=True)
    _, ax = plt.subplots(1, 1, figsize=(12, 4.5))

    case = CASES[0]  # oscillatory
    t, y_true, y_pred = _run_or_load(case, SEED)
    _plot_panel(ax, t, y_true, y_pred, case, show_panel=False,
                title=r"Oscillatory Dynamics Prediction ($\beta=5,\,n=3,\,\sigma=0$)")

    finalize_figure(FIGURE_PATH_OSC)
    print(f"Saved: {FIGURE_PATH_OSC}")


if __name__ == "__main__":
    main()
    main_oscillatory()
