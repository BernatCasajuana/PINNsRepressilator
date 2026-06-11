"""
Experiment 4 — Sampling density.

Question: How sparse can the observations be before parameter recovery degrades?

Design:
  - True parameters: β = 5.0, n = 3.0 (oscillatory regime)
  - Fixed noise level: 0.05 (relative to peak-to-peak amplitude)
  - All three repressors observed (x1, x2, x3)
  - Observation counts: 10, 25, 50, 100 — evenly spaced across the 1000-point grid
    (as a percentage of the full grid: 1%, 2.5%, 5%, 10%)
  - 5 seeds per count; 10000 Adam iterations per run
  - Initial guesses: β₀ = 4.0, n₀ = 2.5

Figure: 4-panel 2×2 layout
  - (0,0): β relative error vs observation count + seed dots
  - (0,1): n relative error vs observation count + seed dots
  - (1,0): combined parameter error + significance brackets (vs 100-point baseline)
  - (1,1): state RMSE + seed dots + horizontal σ reference line

X-axis labels show count and % of full grid for context.
Significance: two-sided Mann–Whitney U, Holm–Bonferroni corrected, vs 100-point baseline.

Key finding expected: sparse sampling (<25 points) removes complete oscillatory phases,
costing critical phase information for identifying n.  State RMSE degrades more gracefully
because the ODE residual provides a global constraint between observations.
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
    evenly_spaced_observation_indices,
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
observation_counts = [10, 25, 50, 100]
total_grid_points = 1000
seeds = [0, 1, 2, 3, 4]
train_iterations = 10000
results_dir = "results/exp4_sampling_density"
figure_path = "figures/exp4_sampling_density.png"

BETA_COLOR = "#4C78A8"
N_COLOR = "#F58518"
COMBINED_COLOR = "#222222"
RMSE_COLOR = "#222222"


def main():
    ensure_project_directories()
    expected_runs = len(observation_counts) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp4_sampling_density",
            "script_path": __file__,
            "results_dir": results_dir,
            "figure_path": figure_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "observation_counts": list(observation_counts),
            "total_grid_points": total_grid_points,
            "true_beta": true_beta,
            "true_n": true_n,
            "noise_level": noise_level,
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )
    raw_rows = []

    for obs_count in observation_counts:
        for seed in seeds:
            dataset = make_synthetic_dataset(true_beta, true_n, noise_level=noise_level, seed=seed)
            observation_indices = evenly_spaced_observation_indices(
                len(dataset["t"]), obs_count
            )
            result = run_inverse(
                dataset_path=dataset,
                outdir_base=os.path.join(results_dir, "runs"),
                beta_guess=4.0,
                n_guess=2.5,
                observed_components=[0, 1, 2],
                train_iterations=train_iterations,
                observation_indices=observation_indices,
                random_seed=seed,
                save_checkpoint=True,
            )
            raw_rows.append(
                {
                    "observation_count": obs_count,
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
        group_keys=["observation_count"],
        metric_keys=["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key=lambda row: row["observation_count"])

    write_csv(
        os.path.join(results_dir, "exp4_sampling_density_raw.csv"),
        raw_rows,
        ["observation_count", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(results_dir, "exp4_sampling_density_summary.csv"),
        summary_rows,
        [
            "observation_count", "num_runs",
            "beta_rel_error_mean", "beta_rel_error_std",
            "n_rel_error_mean", "n_rel_error_std",
            "parameter_rel_error_mean", "parameter_rel_error_std",
            "state_rmse_mean", "state_rmse_std",
        ],
    )

    count_values = [row["observation_count"] for row in summary_rows]
    positions = list(range(len(count_values)))
    count_labels = [str(c) for c in count_values]

    beta_means = [row["beta_rel_error_mean"] for row in summary_rows]
    beta_stds = [row["beta_rel_error_std"] for row in summary_rows]
    n_means = [row["n_rel_error_mean"] for row in summary_rows]
    n_stds = [row["n_rel_error_std"] for row in summary_rows]
    parameter_means = [row["parameter_rel_error_mean"] for row in summary_rows]
    parameter_stds = [row["parameter_rel_error_std"] for row in summary_rows]
    state_means = [row["state_rmse_mean"] for row in summary_rows]
    state_stds = [row["state_rmse_std"] for row in summary_rows]
    show_eb = len(seeds) > 1

    baseline_count = max(count_values)
    count_comparisons = [(baseline_count, c) for c in count_values if c != baseline_count]

    beta_vals_by_count = metric_values_by_group(raw_rows, "observation_count", "beta_rel_error")
    n_vals_by_count = metric_values_by_group(raw_rows, "observation_count", "n_rel_error")
    parameter_vals_by_count = metric_values_by_group(raw_rows, "observation_count", "parameter_rel_error")
    state_vals_by_count = metric_values_by_group(raw_rows, "observation_count", "state_rmse")

    parameter_significance = pairwise_significance(parameter_vals_by_count, count_comparisons)
    state_significance = pairwise_significance(state_vals_by_count, count_comparisons)

    plt.rcParams['axes.formatter.useoffset'] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        rf"Sampling Density ($\beta$={true_beta}, $n$={true_n}, $\sigma$={noise_level}, {len(seeds)} seeds)",
        fontsize=13,
    )

    def _label(ax, letter):
        ax.text(-0.02, 1.10, letter, transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='bottom', ha='left', color='black', clip_on=False)

    def _errorbar(ax, color, label, means, stds):
        ax.errorbar(
            positions, means,
            yerr=stds if show_eb else None,
            fmt="o", linestyle="-", linewidth=1.5,
            capsize=4 if show_eb else 0,
            color=color, ecolor=color,
            markerfacecolor=color, markeredgecolor=color,
            zorder=4, label=label,
        )

    def _xformat(ax):
        ax.set_xticks(positions, count_labels)
        ax.set_xlabel("Observation Count")

    position_by_count = {c: i for i, c in enumerate(count_values)}

    # Panel A: β and n merged (blue=β, grey=n), no jitter
    _errorbar(axes[0], BETA_COLOR, r"$\beta$", beta_means, beta_stds)
    _errorbar(axes[0], N_COLOR, r"$n$", n_means, n_stds)
    _xformat(axes[0])
    axes[0].set_ylabel("Relative Error")
    axes[0].set_title(r"$\beta$ and $n$ Recovery vs Sampling Density")
    axes[0].legend(fontsize=9)
    _label(axes[0], 'A')

    # Panel B: combined error + significance
    _errorbar(axes[1], COMBINED_COLOR, None, parameter_means, parameter_stds)
    _xformat(axes[1])
    axes[1].set_ylabel("Combined Parameter Error  0.5(|Δβ|/β + |Δn|/n)")
    axes[1].set_title("Combined Parameter Recovery vs Sampling Density")
    parameter_tops = [m + (s if show_eb else 0.0) for m, s in zip(parameter_means, parameter_stds)]
    parameter_top_by_count = {c: t for c, t in zip(count_values, parameter_tops)}
    annotate_pairwise_comparisons(
        axes[1], x_positions=position_by_count, top_values=parameter_top_by_count,
        comparisons=count_comparisons, significance=parameter_significance, use_adjusted_p_value=True)
    _label(axes[1], 'B')

    # Panel C: state RMSE + significance
    _errorbar(axes[2], COMBINED_COLOR, None, state_means, state_stds)
    _xformat(axes[2])
    axes[2].set_ylabel("State RMSE  (vs clean trajectory)")
    axes[2].set_title("Trajectory Reconstruction vs Sampling Density")
    state_tops = [m + (s if show_eb else 0.0) for m, s in zip(state_means, state_stds)]
    state_top_by_count = {c: t for c, t in zip(count_values, state_tops)}
    annotate_pairwise_comparisons(
        axes[2], x_positions=position_by_count, top_values=state_top_by_count,
        comparisons=count_comparisons, significance=state_significance, use_adjusted_p_value=True)
    _label(axes[2], 'C')

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
