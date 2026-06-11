"""
Experiment 8 — Training convergence.

Question: How do the loss components and parameter estimates evolve during training?
Does 10000 iterations provide adequate convergence? Do β̂ and n̂ settle before the
trajectory RMSE plateaus?

Design:
  - Two canonical conditions that span easy and hard recovery:
      "osc_low_noise":   β = 5.0, n = 3.0, noise = 0.01 — oscillatory, low noise
      "osc_high_noise":  β = 5.0, n = 3.0, noise = 0.10 — oscillatory, high noise
  - 3 seeds per condition; 10000 Adam iterations per run (double the default)
  - All three repressors observed; stride = 1; β₀ = 4.0, n₀ = 2.5
  - run_inverse returns parameter_evolution (shape (N_checkpoints, 2), one row per
    100 iterations) and loss_train (shape (N_steps, N_components))

Figure: 2-row × 2-col layout per condition, shown as 2 side-by-side column blocks

  For each condition:
    Top-left:   total loss (sum of all components) on a semilog y-axis
    Top-right:  individual loss components on a semilog y-axis
    Bottom-left: β̂ convergence — estimated β over iterations, with dashed true-β line
    Bottom-right: n̂ convergence — estimated n over iterations, with dashed true-n line

  All seeds are overlaid as separate lines (with alpha transparency).

Key finding expected: β̂ typically converges faster than n̂. The parameter estimates
may overshoot and oscillate before settling. 10000 iterations should be sufficient for
the low-noise case but may still show slow n̂ drift at high noise.
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
conditions = [
    ("osc_low_noise",  {"beta": true_beta, "n": true_n, "noise": 0.01}),
    ("osc_high_noise", {"beta": true_beta, "n": true_n, "noise": 0.10}),
]
seeds = [0, 1, 2]
train_iterations = 10000
results_dir = "results/exp8_convergence"
figure_path = "figures/exp8_convergence.png"

BETA_COLOR = "#0072B2"
N_COLOR = "#E69F00"
TOTAL_LOSS_COLOR = "#333333"
EQ_COLOR = "#C62828"
IC_COLOR = "#00897B"
OBS_COLOR = "#8C564B"
CONDITION_LABELS = {
    "osc_low_noise": r"$\sigma=0.01$",
    "osc_high_noise": r"$\sigma=0.10$",
}


def main():
    ensure_project_directories()
    expected_runs = len(conditions) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp8_convergence",
            "script_path": __file__,
            "results_dir": results_dir,
            "figure_path": figure_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "conditions": [
                {"name": name, **params} for name, params in conditions
            ],
            "true_beta": true_beta,
            "true_n": true_n,
            "note": (
                "parameter_evolution sampled every 100 iterations "
                "(controlled by SaveVariablesCallback period=100 in inverse.py)"
            ),
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )

    # Collect per-run results indexed by (condition_name, seed)
    results_store = {}
    raw_rows = []

    for cond_name, params in conditions:
        for seed in seeds:
            dataset = make_synthetic_dataset(
                params["beta"], params["n"], noise_level=params["noise"], seed=seed
            )
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
            results_store[(cond_name, seed)] = result
            raw_rows.append(
                {
                    "condition": cond_name,
                    "noise": params["noise"],
                    "seed": seed,
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
            "condition", "noise", "seed",
            "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse",
            "final_beta_hat", "final_n_hat", "outdir",
        ],
    )

    # Build figure: one 2×2 block per condition, side by side
    n_conditions = len(conditions)
    fig, axes = plt.subplots(
        2, 2 * n_conditions,
        figsize=(8 * n_conditions, 9),
        squeeze=False,
    )
    fig.suptitle(
        rf"Exp 8 — Training convergence ($\beta$={true_beta}, $n$={true_n}, {len(seeds)} seeds, {train_iterations} iters)",
        fontsize=13,
    )

    seed_alphas = [0.8, 0.55, 0.35]

    for col_block, (cond_name, params) in enumerate(conditions):
        ax_loss = axes[0, col_block * 2]      # top-left: total + component loss
        ax_comp = axes[0, col_block * 2 + 1]  # top-right: individual components
        ax_beta = axes[1, col_block * 2]      # bottom-left: β̂ evolution
        ax_n    = axes[1, col_block * 2 + 1]  # bottom-right: n̂ evolution

        cond_label = CONDITION_LABELS.get(cond_name, cond_name)
        true_beta_c = params["beta"]
        true_n_c = params["n"]

        for s_idx, seed in enumerate(seeds):
            result = results_store.get((cond_name, seed))
            if result is None:
                continue
            alpha = seed_alphas[s_idx % len(seed_alphas)]
            loss_train = result["loss_train"]     # (N_steps, N_components)
            iter_axis = result["iteration_axis"]  # (N_steps,)
            param_evo = result["parameter_evolution"]  # (N_checkpoints, 2)
            param_iter = result["param_evo_iterations"]  # (N_checkpoints,)

            # Total loss
            total_loss = loss_train.sum(axis=1)
            ax_loss.semilogy(iter_axis, total_loss, color=TOTAL_LOSS_COLOR, alpha=alpha, linewidth=1.2)

            # Component losses
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

            # β̂ evolution
            if param_evo.shape[0] > 0:
                ax_beta.plot(param_iter, param_evo[:, 0], color=BETA_COLOR, alpha=alpha, linewidth=1.2)

            # n̂ evolution
            if param_evo.shape[0] > 0:
                ax_n.plot(param_iter, param_evo[:, 1], color=N_COLOR, alpha=alpha, linewidth=1.2)

        # True parameter reference lines
        ax_beta.axhline(true_beta_c, color="black", linestyle="--", linewidth=1.0,
                        alpha=0.7, label=rf"True $\beta$ = {true_beta_c}")
        ax_n.axhline(true_n_c, color="black", linestyle="--", linewidth=1.0,
                     alpha=0.7, label=rf"True $n$ = {true_n_c}")

        # Labels
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Total loss  (log)")
        ax_loss.set_title(f"Total loss — {cond_label}")

        ax_comp.set_xlabel("Iteration")
        ax_comp.set_ylabel("Loss component  (log)")
        ax_comp.set_title(f"Loss components — {cond_label}")
        if col_block == 0:
            ax_comp.legend(fontsize=8, loc="upper right")

        ax_beta.set_xlabel("Iteration")
        ax_beta.set_ylabel(r"$\hat{\beta}$")
        ax_beta.set_title(rf"$\hat{{\beta}}$ convergence — {cond_label}")
        ax_beta.legend(fontsize=8)

        ax_n.set_xlabel("Iteration")
        ax_n.set_ylabel(r"$\hat{n}$")
        ax_n.set_title(rf"$\hat{{n}}$ convergence — {cond_label}")
        ax_n.legend(fontsize=8)

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
