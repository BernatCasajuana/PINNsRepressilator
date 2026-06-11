"""
Experiment 7 — Physics loss weight sensitivity.

Question: How sensitive is inverse-PINN performance to the physics loss weight λ_f?

The total loss is: L = λ_f · L_eq  +  λ_0 · L_IC  +  λ_y · L_obs
This experiment sweeps λ_f while holding λ_0 = λ_y = 1.0 fixed to isolate the
contribution of the ODE residual penalty.

Design:
  - True parameters: β = 5.0, n = 3.0 (oscillatory regime)
  - Fixed noise level: 0.05; all three repressors observed; stride = 1 (dense)
  - λ_f sweep: {0.01, 0.1, 1.0, 10.0, 100.0} — 5 decades
  - λ_0 = λ_y = 1.0 (fixed)
  - 5 seeds per λ_f; 5000 Adam iterations per run
  - Initial guesses: β₀ = 4.0, n₀ = 2.5

  loss_weights format for run_inverse (all 3 observed components):
    [λ_f, λ_f, λ_f,   ← 3 ODE residual terms
     λ_0, λ_0, λ_0,   ← 3 IC terms
     λ_y, λ_y, λ_y]   ← 3 observation terms

Figure: 3-panel 1×3 layout (15×5)
  - Panel A: β and n relative error vs λ_f — overlaid lines with significance brackets
             for both β (lower) and n (upper, stacked above β brackets)
  - Panel B: combined parameter error + significance brackets (vs λ_f = 1.0 baseline)
  - Panel C: state RMSE + significance brackets (vs λ_f = 1.0 baseline)
  Panels B and C show brackets vs λ_f = 1.0; Panel A stacks β brackets then n brackets.

Key finding expected: very low λ_f (<0.1) lets the network ignore the ODE constraint,
degrading parameter recovery while fitting observations well.  Very high λ_f (>10) can
suppress the observation loss and prevent convergence.  λ_f ≈ 1 is expected to be near-optimal.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.experiment_utils import (
    aggregate_metrics,
    annotate_pairwise_comparisons,
    ensure_project_directories,
    finalize_figure,
    make_synthetic_dataset,
    metric_values_by_group,
    pairwise_significance,
    write_csv,
    write_run_manifest,
)
from scripts.pinns.inverse import run_inverse

true_beta = 5.0
true_n = 3.0
noise_level = 0.05
lambda_f_values = [0.01, 0.1, 1.0, 10.0, 100.0]
lambda_0 = 1.0
lambda_y = 1.0
seeds = [0, 1, 2, 3, 4]
train_iterations = 5000
results_dir = "results/exp7_loss_weights"
figure_path = "figures/exp7_loss_weights.png"

BETA_COLOR = "#009E73"
N_COLOR = "#D55E00"
COMBINED_COLOR = "#222222"
RMSE_COLOR = "#222222"


def _build_loss_weights(lf: float) -> list:
    """[eq1, eq2, eq3, ic1, ic2, ic3, obs1, obs2, obs3] for all-3-observed case."""
    return [lf, lf, lf, lambda_0, lambda_0, lambda_0, lambda_y, lambda_y, lambda_y]


def main():
    ensure_project_directories()
    expected_runs = len(lambda_f_values) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp7_loss_weights",
            "script_path": __file__,
            "results_dir": results_dir,
            "figure_path": figure_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "lambda_f_values": list(lambda_f_values),
            "lambda_0": lambda_0,
            "lambda_y": lambda_y,
            "true_beta": true_beta,
            "true_n": true_n,
            "noise_level": noise_level,
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )
    raw_rows = []

    for lf in lambda_f_values:
        loss_weights = _build_loss_weights(lf)
        for seed in seeds:
            dataset = make_synthetic_dataset(true_beta, true_n, noise_level=noise_level, seed=seed)
            result = run_inverse(
                dataset_path=dataset,
                outdir_base=os.path.join(results_dir, "runs"),
                beta_guess=4.0,
                n_guess=2.5,
                observation_stride=1,
                observed_components=[0, 1, 2],
                loss_weights=loss_weights,
                train_iterations=train_iterations,
                random_seed=seed,
                save_checkpoint=False,
            )
            raw_rows.append(
                {
                    "lambda_f": lf,
                    "seed": seed,
                    "beta_rel_error": result["beta_rel_error"],
                    "n_rel_error": result["n_rel_error"],
                    "parameter_rel_error": result["parameter_rel_error"],
                    "state_rmse": result["state_rmse"],
                    "outdir": result["outdir"],
                }
            )

    summary_rows = aggregate_metrics(
        raw_rows,
        group_keys=["lambda_f"],
        metric_keys=["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key=lambda row: row["lambda_f"])

    write_csv(
        os.path.join(results_dir, "exp7_loss_weights_raw.csv"),
        raw_rows,
        ["lambda_f", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(results_dir, "exp7_loss_weights_summary.csv"),
        summary_rows,
        [
            "lambda_f", "num_runs",
            "beta_rel_error_mean", "beta_rel_error_std",
            "n_rel_error_mean", "n_rel_error_std",
            "parameter_rel_error_mean", "parameter_rel_error_std",
            "state_rmse_mean", "state_rmse_std",
        ],
    )

    lf_values = [row["lambda_f"] for row in summary_rows]
    positions = list(range(len(lf_values)))
    lf_labels = [str(v) for v in lf_values]

    beta_means = [row["beta_rel_error_mean"] for row in summary_rows]
    beta_stds = [row["beta_rel_error_std"] for row in summary_rows]
    n_means = [row["n_rel_error_mean"] for row in summary_rows]
    n_stds = [row["n_rel_error_std"] for row in summary_rows]
    parameter_means = [row["parameter_rel_error_mean"] for row in summary_rows]
    parameter_stds = [row["parameter_rel_error_std"] for row in summary_rows]
    state_means = [row["state_rmse_mean"] for row in summary_rows]
    state_stds = [row["state_rmse_std"] for row in summary_rows]
    show_eb = len(seeds) > 1

    baseline_lf = 1.0
    lf_comparisons = [(baseline_lf, lf) for lf in lf_values if lf != baseline_lf]

    beta_vals_by_lf = metric_values_by_group(raw_rows, "lambda_f", "beta_rel_error")
    n_vals_by_lf = metric_values_by_group(raw_rows, "lambda_f", "n_rel_error")
    parameter_vals_by_lf = metric_values_by_group(raw_rows, "lambda_f", "parameter_rel_error")
    state_vals_by_lf = metric_values_by_group(raw_rows, "lambda_f", "state_rmse")

    parameter_significance = pairwise_significance(parameter_vals_by_lf, lf_comparisons)
    state_significance = pairwise_significance(state_vals_by_lf, lf_comparisons)
    beta_significance = pairwise_significance(beta_vals_by_lf, lf_comparisons)
    n_significance = pairwise_significance(n_vals_by_lf, lf_comparisons)

    plt.rcParams['axes.formatter.useoffset'] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def _label(ax, letter):
        ax.text(-0.02, 1.04, letter, transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='bottom', ha='left', color='black', clip_on=False)

    def _errorbar(ax, color, label, means, stds, marker="o"):
        ax.errorbar(
            positions, means,
            yerr=stds if show_eb else None,
            fmt=marker, linestyle="-", linewidth=1.0,
            capsize=4 if show_eb else 0,
            color=color, ecolor=color,
            markerfacecolor=color, markeredgecolor=color,
            zorder=4, label=label,
        )

    def _xformat(ax):
        ax.set_xticks(positions, lf_labels)
        ax.set_xlabel(r"Physics Loss Weight $\lambda_f$  ($\lambda_0 = \lambda_y = 1$)")

    def _highlight_baseline(ax):
        if baseline_lf in lf_values:
            bi = lf_values.index(baseline_lf)
            ax.axvspan(bi - 0.4, bi + 0.4, alpha=0.07, color="gray", zorder=0)

    position_by_lf = {lf: i for i, lf in enumerate(lf_values)}

    # Panel A: β and n recovery with significance vs λf=1 baseline
    _errorbar(axes[0], BETA_COLOR, r"$\beta$", beta_means, beta_stds, marker="o")
    _errorbar(axes[0], N_COLOR, r"$n$", n_means, n_stds, marker="s")
    _highlight_baseline(axes[0])
    _xformat(axes[0])
    beta_tops_pa = {lf: beta_means[i] + (beta_stds[i] if show_eb else 0.0)
                   for i, lf in enumerate(lf_values)}
    annotate_pairwise_comparisons(
        axes[0], x_positions=position_by_lf, top_values=beta_tops_pa,
        comparisons=lf_comparisons, significance=beta_significance, use_adjusted_p_value=True)
    n_tops_pa = {lf: n_means[i] + (n_stds[i] if show_eb else 0.0)
                for i, lf in enumerate(lf_values)}
    annotate_pairwise_comparisons(
        axes[0], x_positions=position_by_lf, top_values=n_tops_pa,
        comparisons=lf_comparisons, significance=n_significance, use_adjusted_p_value=True)
    axes[0].set_ylabel("Relative Error")
    axes[0].set_title(r"$\beta$ and $n$ Recovery")
    axes[0].legend(fontsize=9)
    _label(axes[0], 'A')

    # Panel B: combined error + significance + baseline
    _errorbar(axes[1], COMBINED_COLOR, None, parameter_means, parameter_stds)
    _highlight_baseline(axes[1])
    _xformat(axes[1])
    axes[1].set_ylabel("Param. Error")
    axes[1].set_title("Combined Parameter Recovery")
    parameter_tops = [m + (s if show_eb else 0.0) for m, s in zip(parameter_means, parameter_stds)]
    parameter_top_by_lf = {lf: t for lf, t in zip(lf_values, parameter_tops)}
    annotate_pairwise_comparisons(
        axes[1], x_positions=position_by_lf, top_values=parameter_top_by_lf,
        comparisons=lf_comparisons, significance=parameter_significance, use_adjusted_p_value=True)
    _label(axes[1], 'B')

    # Panel C: state RMSE + significance + baseline
    _errorbar(axes[2], COMBINED_COLOR, None, state_means, state_stds)
    _highlight_baseline(axes[2])
    _xformat(axes[2])
    axes[2].set_ylabel("State RMSE")
    axes[2].set_title("Trajectory Reconstruction")
    state_tops = [m + (s if show_eb else 0.0) for m, s in zip(state_means, state_stds)]
    state_top_by_lf = {lf: t for lf, t in zip(lf_values, state_tops)}
    annotate_pairwise_comparisons(
        axes[2], x_positions=position_by_lf, top_values=state_top_by_lf,
        comparisons=lf_comparisons, significance=state_significance, use_adjusted_p_value=True)
    _label(axes[2], 'C')

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
