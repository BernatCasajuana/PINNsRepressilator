"""
Experiment 7 — Training convergence.

Question: How do the loss components and parameter estimates evolve during training?
Does 10000 iterations provide adequate convergence? Do β̂ and n̂ settle before the
loss plateaus?

Design:
  - Single canonical condition: β = 5.0, n = 3.0, noise = 0.05 (oscillatory regime,
    same noise level as all other experiments)
  - 5 seeds; 10000 Adam iterations per run
  - All three repressors observed; stride = 1; β₀ = 4.0, n₀ = 2.5
  - run_inverse returns parameter_evolution (shape (N_checkpoints, 2), one row per
    100 iterations) and loss_train (shape (N_steps, N_components))

Figure: 2×2 layout
  - Panel A (top-left):   total loss on a semilog y-axis, 5 seeds overlaid
  - Panel B (top-right):  individual loss components (L_eq, L_IC, L_obs) on semilog y, 5 seeds overlaid
  - Panel C (bottom-left): β̂ convergence — one line per β₀ initial guess (fixed n₀=2.5, seed=0)
                            β₀ ∈ {2, 3, 4, 5, 6, 7, 8}; dashed line = true β
  - Panel D (bottom-right): n̂ convergence — one line per n₀ initial guess (fixed β₀=4.0, seed=0)
                             n₀ ∈ {1.5, 2, 2.5, 3, 3.5, 4, 4.5}; dashed line = true n

Key finding expected: β̂ typically converges faster than n̂. Panels C/D reveal which
initial guesses converge reliably to the true values vs which diverge or stall.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.utils import (
    ensure_project_directories,
    finalize_figure,
    make_synthetic_dataset,
    write_csv,
    write_run_manifest,
)
from scripts.pinns.inverse import run_inverse

true_beta = 5.0
true_n = 3.0
noise_level = 0.05
seeds = [0, 1, 2, 3, 4]
train_iterations = 10000
results_dir = "results/exp7_convergence"
figure_path = "figures/exp7_convergence.png"

BETA_COLOR = "#0072B2"
N_COLOR = "#0072B2"
TOTAL_LOSS_COLOR = "#222222"
EQ_COLOR = "#C62828"
IC_COLOR = "#00897B"
OBS_COLOR = "#8C564B"


def main():
    ensure_project_directories()
    expected_runs = len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp7_convergence",
            "script_path": __file__,
            "results_dir": results_dir,
            "figure_path": figure_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "true_beta": true_beta,
            "true_n": true_n,
            "noise_level": noise_level,
            "note": (
                "parameter_evolution sampled every 100 iterations "
                "(controlled by SaveVariablesCallback period=100 in inverse.py)"
            ),
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )

    results_store = {}
    raw_rows = []

    for seed in seeds:
        dataset = make_synthetic_dataset(true_beta, true_n, noise_level=noise_level, seed=seed)
        result = run_inverse(
            dataset_path=dataset,
            outdir_base=os.path.join(results_dir, "runs"),
            beta_guess=4.0,
            n_guess=2.5,
            observation_stride=1,
            observed_components=[0, 1, 2],
            train_iterations=train_iterations,
            random_seed=seed,
            save_checkpoint=False,
        )
        results_store[seed] = result
        raw_rows.append(
            {
                "seed": seed,
                "noise_level": noise_level,
                "beta_rel_error": result["beta_rel_error"],
                "n_rel_error": result["n_rel_error"],
                "parameter_rel_error": result["parameter_rel_error"],
                "state_rmse": result["state_rmse"],
                "final_beta_hat": result["beta_estimated"],
                "final_n_hat": result["n_estimated"],
                "outdir": result["outdir"],
            }
        )

    write_csv(
        os.path.join(results_dir, "exp7_convergence_raw.csv"),
        raw_rows,
        [
            "seed", "noise_level",
            "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse",
            "final_beta_hat", "final_n_hat", "outdir",
        ],
    )

    # Varied initial-guess convergence for panels C/D (seed=0 dataset, exp5 grid)
    beta_guess_range = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    n_guess_range = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    fixed_dataset = make_synthetic_dataset(true_beta, true_n, noise_level=noise_level, seed=0)

    beta_convergence = {}
    for bg in beta_guess_range:
        res = run_inverse(
            dataset_path=fixed_dataset,
            outdir_base=os.path.join(results_dir, "runs_guess_beta", f"bg{bg}"),
            beta_guess=bg,
            n_guess=2.5,
            observation_stride=1,
            observed_components=[0, 1, 2],
            train_iterations=train_iterations,
            random_seed=0,
            save_checkpoint=False,
        )
        beta_convergence[bg] = res

    n_convergence = {}
    for ng in n_guess_range:
        res = run_inverse(
            dataset_path=fixed_dataset,
            outdir_base=os.path.join(results_dir, "runs_guess_n", f"ng{ng}"),
            beta_guess=4.0,
            n_guess=ng,
            observation_stride=1,
            observed_components=[0, 1, 2],
            train_iterations=train_iterations,
            random_seed=0,
            save_checkpoint=False,
        )
        n_convergence[ng] = res

    plt.rcParams['axes.formatter.useoffset'] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)

    ax_loss = axes[0, 0]
    ax_comp = axes[0, 1]
    ax_beta = axes[1, 0]
    ax_n    = axes[1, 1]

    from matplotlib.lines import Line2D

    for seed in seeds:
        result = results_store[seed]
        loss_train = result["loss_train"]
        iters = result.get("iteration_axis")
        if iters is not None:
            ax_loss.semilogy(iters, loss_train.sum(axis=1),
                             color=TOTAL_LOSS_COLOR, alpha=0.5, linewidth=0.9)
            if loss_train.shape[1] >= 9:
                ax_comp.semilogy(iters, loss_train[:, :3].sum(axis=1),
                                 color=EQ_COLOR, alpha=0.5, linewidth=0.9)
                ax_comp.semilogy(iters, loss_train[:, 3:6].sum(axis=1),
                                 color=IC_COLOR, alpha=0.5, linewidth=0.9, linestyle="--")
                ax_comp.semilogy(iters, loss_train[:, 6:].sum(axis=1),
                                 color=OBS_COLOR, alpha=0.5, linewidth=0.9, linestyle=":")

    for bg, res in beta_convergence.items():
        pe = res["parameter_evolution"]
        param_iters = res.get("param_evo_iterations")
        if param_iters is not None and pe.shape[0] > 0:
            ax_beta.plot(param_iters, pe[:, 0], color=BETA_COLOR, alpha=0.5, linewidth=0.9)

    ax_beta.axhline(true_beta, color="black", linestyle="--", linewidth=1.0)

    for ng, res in n_convergence.items():
        pe = res["parameter_evolution"]
        param_iters = res.get("param_evo_iterations")
        if param_iters is not None and pe.shape[0] > 0:
            ax_n.plot(param_iters, pe[:, 1], color=N_COLOR, alpha=0.5, linewidth=0.9)

    ax_n.axhline(true_n, color="black", linestyle="--", linewidth=1.0)

    for panel_ax, letter in zip([ax_loss, ax_comp, ax_beta, ax_n], "ABCD"):
        panel_ax.text(-0.02, 1.04, letter, transform=panel_ax.transAxes,
                      fontsize=15, fontweight='bold', va='bottom', ha='left', color='black', clip_on=False)

    ax_loss.set_xlabel("Iterations")
    ax_loss.set_ylabel("Loss (log scale)")
    ax_loss.set_title("Total Loss")
    ax_loss.legend(handles=[Line2D([0], [0], color=TOTAL_LOSS_COLOR, linewidth=1.2, label="Seeds")],
                   fontsize=11, loc="upper right")

    ax_comp.set_xlabel("Iterations")
    ax_comp.set_ylabel("Loss (log scale)")
    ax_comp.set_title("Loss Components")
    ax_comp.legend(handles=[
        Line2D([0], [0], color=EQ_COLOR, linewidth=1.2, label=r"$L_{eq}$"),
        Line2D([0], [0], color=IC_COLOR, linewidth=1.2, linestyle="--", label=r"$L_{IC}$"),
        Line2D([0], [0], color=OBS_COLOR, linewidth=1.2, linestyle=":", label=r"$L_{obs}$"),
    ], fontsize=11, loc="upper right")

    ax_beta.set_xlabel("Iterations")
    ax_beta.set_ylabel(r"$\hat{\beta}$")
    ax_beta.set_title(r"$\hat{\beta}$ Convergence")
    ax_beta.legend(handles=[
        Line2D([0], [0], color=BETA_COLOR, linewidth=1.2, alpha=0.7, label=r"$\beta_0$ = 2, 3, ..., 8"),
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label=rf"true $\beta$ = {true_beta}"),
    ], fontsize=11, handlelength=3)

    ax_n.set_xlabel("Iterations")
    ax_n.set_ylabel(r"$\hat{n}$")
    ax_n.set_title(r"$\hat{n}$ Convergence")
    ax_n.legend(handles=[
        Line2D([0], [0], color=N_COLOR, linewidth=1.2, alpha=0.7, label=r"$n_0$ = 1.5, 2, ..., 4.5"),
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label=rf"true $n$ = {true_n}"),
    ], fontsize=11, handlelength=3)

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
