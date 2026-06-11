"""
Experiment 8 — Training convergence.

Question: How do the loss components and parameter estimates evolve during training?
Does 10000 iterations provide adequate convergence? Do β̂ and n̂ settle before the
loss plateaus?

Design:
  - Single canonical condition: β = 5.0, n = 3.0, noise = 0.05 (oscillatory regime,
    same noise level as all other experiments)
  - 3 seeds; 10000 Adam iterations per run
  - All three repressors observed; stride = 1; β₀ = 4.0, n₀ = 2.5
  - run_inverse returns parameter_evolution (shape (N_checkpoints, 2), one row per
    100 iterations) and loss_train (shape (N_steps, N_components))

Figure: 2×2 layout
  - Panel A (top-left):   total loss on a semilog y-axis
  - Panel B (top-right):  individual loss components (L_eq, L_IC, L_obs) on semilog y
  - Panel C (bottom-left): β̂ convergence — estimated β over iterations
  - Panel D (bottom-right): n̂ convergence — estimated n over iterations

  All seeds overlaid as separate lines (alpha transparency).

Key finding expected: β̂ typically converges faster than n̂. The loss components reveal
which constraint drives early vs late training dynamics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.experiment_utils import (
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
seeds = [0, 1, 2]
train_iterations = 10000
results_dir = "results/exp8_convergence"
figure_path = "figures/exp8_convergence.png"

BETA_COLOR = "#4C78A8"
N_COLOR = "#F58518"
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
            "experiment_name": "exp8_convergence",
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
        os.path.join(results_dir, "exp8_convergence_raw.csv"),
        raw_rows,
        [
            "seed", "noise_level",
            "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse",
            "final_beta_hat", "final_n_hat", "outdir",
        ],
    )

    plt.rcParams['axes.formatter.useoffset'] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    fig.suptitle(
        rf"Training Convergence ($\beta$={true_beta}, $n$={true_n}, $\sigma$={noise_level}, {len(seeds)} seeds, {train_iterations} iterations)",
        fontsize=13,
    )

    seed_alphas = [0.8, 0.55, 0.35]

    ax_loss = axes[0, 0]
    ax_comp = axes[0, 1]
    ax_beta = axes[1, 0]
    ax_n    = axes[1, 1]

    for s_idx, seed in enumerate(seeds):
        result = results_store[seed]
        alpha = seed_alphas[s_idx % len(seed_alphas)]
        loss_train = result["loss_train"]
        iter_axis = result["iteration_axis"]
        param_evo = result["parameter_evolution"]
        param_iter = result["param_evo_iterations"]

        total_loss = loss_train.sum(axis=1)
        ax_loss.semilogy(iter_axis, total_loss, color=TOTAL_LOSS_COLOR, alpha=alpha, linewidth=1.2,
                         label=f"seed {seed}")

        if loss_train.shape[1] >= 9:
            eq_loss = loss_train[:, :3].sum(axis=1)
            ic_loss = loss_train[:, 3:6].sum(axis=1)
            obs_loss = loss_train[:, 6:].sum(axis=1)
            ax_comp.semilogy(iter_axis, eq_loss, color=EQ_COLOR, alpha=alpha, linewidth=1.0,
                             linestyle="-", label=r"$L_{eq}$" if s_idx == 0 else "_")
            ax_comp.semilogy(iter_axis, ic_loss, color=IC_COLOR, alpha=alpha, linewidth=1.0,
                             linestyle="--", label=r"$L_{IC}$" if s_idx == 0 else "_")
            ax_comp.semilogy(iter_axis, obs_loss, color=OBS_COLOR, alpha=alpha, linewidth=1.0,
                             linestyle=":", label=r"$L_{obs}$" if s_idx == 0 else "_")
        else:
            for k in range(loss_train.shape[1]):
                ax_comp.semilogy(iter_axis, loss_train[:, k], alpha=alpha, linewidth=0.8)

        if param_evo.shape[0] > 0:
            ax_beta.plot(param_iter, param_evo[:, 0], color=BETA_COLOR, alpha=alpha, linewidth=1.2)
            ax_n.plot(param_iter, param_evo[:, 1], color=N_COLOR, alpha=alpha, linewidth=1.2)

    ax_beta.axhline(true_beta, color="black", linestyle="--", linewidth=1.0,
                    alpha=0.7, label=rf"true $\beta$ = {true_beta}")
    ax_n.axhline(true_n, color="black", linestyle="--", linewidth=1.0,
                 alpha=0.7, label=rf"true $n$ = {true_n}")

    for panel_ax, letter in zip([ax_loss, ax_comp, ax_beta, ax_n], "ABCD"):
        panel_ax.text(-0.02, 1.04, letter, transform=panel_ax.transAxes,
                      fontsize=15, fontweight='bold', va='bottom', ha='left', color='black', clip_on=False)

    ax_loss.set_xlabel("Iterations")
    ax_loss.set_ylabel("Total Loss  (log)")
    ax_loss.set_title("Total Loss")
    ax_loss.legend(fontsize=8, loc="upper right")

    ax_comp.set_xlabel("Iterations")
    ax_comp.set_ylabel("Loss Component  (log)")
    ax_comp.set_title("Loss Components")
    ax_comp.legend(fontsize=8, loc="upper right")

    ax_beta.set_xlabel("Iterations")
    ax_beta.set_ylabel(r"$\hat{\beta}$")
    ax_beta.set_title(r"$\hat{\beta}$ Convergence")
    ax_beta.legend(fontsize=8)

    ax_n.set_xlabel("Iterations")
    ax_n.set_ylabel(r"$\hat{n}$")
    ax_n.set_title(r"$\hat{n}$ Convergence")
    ax_n.legend(fontsize=8)

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
