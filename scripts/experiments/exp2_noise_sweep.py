"""
Experiment 2 — Noise sensitivity.

Question: How does inverse-PINN parameter recovery degrade as observation noise increases?

Design:
  - True parameters: β = 5.0, n = 3.0 (canonical oscillatory regime)
  - All three repressors observed (x1, x2, x3), dense sampling (stride = 1, 1000 points)
  - Noise levels: 0.0, 0.01, 0.05, 0.10, 0.20 — each expressed as a fraction of the
    mean peak-to-peak signal amplitude so values are comparable across regimes
  - 5 seeds per noise level (independent data realisations)
  - 10000 Adam iterations per run; initial guesses β₀ = 4.0, n₀ = 2.5

Figure: 4-panel 2×2 layout
  - (0,0): β relative error vs noise — mean ± SD + individual seed dots
  - (0,1): n relative error vs noise — mean ± SD + individual seed dots
  - (1,0): combined parameter error vs noise + significance brackets (vs σ = 0)
  - (1,1): state RMSE vs noise + individual seed dots + σ reference line
            (σ reference = noise_level × signal_amplitude, showing scale of added noise)

Significance: two-sided Mann–Whitney U, Holm–Bonferroni corrected, vs σ = 0 baseline.

Key finding expected: parameter recovery is relatively stable across this noise range;
trajectory RMSE grows faster because the observation loss directly penalises trajectory
fit. The ODE residual regularises trajectories even when observations are corrupted.
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
noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]
seeds = [0, 1, 2, 3, 4]
observed_components = [0, 1, 2]
train_iterations = 10000
results_dir = "results/exp2_noise_sweep"
figure_path = "figures/exp2_noise_sweep.png"

BETA_COLOR = "#4C78A8"
N_COLOR = "#F58518"
COMBINED_COLOR = "#222222"
RMSE_COLOR = "#222222"


def main():
    ensure_project_directories()
    expected_runs = len(noise_levels) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp2_noise_sweep",
            "script_path": __file__,
            "results_dir": results_dir,
            "figure_path": figure_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "noise_levels": list(noise_levels),
            "observed_components": list(observed_components),
            "true_beta": true_beta,
            "true_n": true_n,
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )
    raw_rows = []

    for noise_level in noise_levels:
        for seed in seeds:
            dataset = make_synthetic_dataset(true_beta, true_n, noise_level=noise_level, seed=seed)
            result = run_inverse(
                dataset_path=dataset,
                outdir_base=os.path.join(results_dir, "runs"),
                beta_guess=4.0,
                n_guess=2.5,
                observation_stride=1,
                observed_components=observed_components,
                train_iterations=train_iterations,
                random_seed=seed,
                save_checkpoint=True,
            )
            raw_rows.append(
                {
                    "noise_level": noise_level,
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
        group_keys=["noise_level"],
        metric_keys=["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key=lambda row: row["noise_level"])

    write_csv(
        os.path.join(results_dir, "exp2_noise_sweep_raw.csv"),
        raw_rows,
        ["noise_level", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(results_dir, "exp2_noise_sweep_summary.csv"),
        summary_rows,
        [
            "noise_level", "num_runs",
            "beta_rel_error_mean", "beta_rel_error_std",
            "n_rel_error_mean", "n_rel_error_std",
            "parameter_rel_error_mean", "parameter_rel_error_std",
            "state_rmse_mean", "state_rmse_std",
        ],
    )

    noise_values = [row["noise_level"] for row in summary_rows]
    positions = list(range(len(noise_values)))
    noise_labels = [f"{v:.2f}" for v in noise_values]

    beta_means = [row["beta_rel_error_mean"] for row in summary_rows]
    beta_stds = [row["beta_rel_error_std"] for row in summary_rows]
    n_means = [row["n_rel_error_mean"] for row in summary_rows]
    n_stds = [row["n_rel_error_std"] for row in summary_rows]
    parameter_means = [row["parameter_rel_error_mean"] for row in summary_rows]
    parameter_stds = [row["parameter_rel_error_std"] for row in summary_rows]
    state_means = [row["state_rmse_mean"] for row in summary_rows]
    state_stds = [row["state_rmse_std"] for row in summary_rows]
    show_eb = len(seeds) > 1

    baseline_noise = 0.0 if 0.0 in noise_values else noise_values[0]
    noise_comparisons = [(baseline_noise, v) for v in noise_values if v != baseline_noise]

    beta_vals_by_noise = metric_values_by_group(raw_rows, "noise_level", "beta_rel_error")
    n_vals_by_noise = metric_values_by_group(raw_rows, "noise_level", "n_rel_error")
    parameter_vals_by_noise = metric_values_by_group(raw_rows, "noise_level", "parameter_rel_error")
    state_vals_by_noise = metric_values_by_group(raw_rows, "noise_level", "state_rmse")

    parameter_significance = pairwise_significance(parameter_vals_by_noise, noise_comparisons)
    state_significance = pairwise_significance(state_vals_by_noise, noise_comparisons)

    plt.rcParams['axes.formatter.useoffset'] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        rf"Noise Sensitivity ($\beta$={true_beta}, $n$={true_n}, {len(seeds)} seeds)",
        fontsize=13,
    )

    def _label(ax, letter):
        ax.text(-0.02, 1.04, letter, transform=ax.transAxes,
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
        ax.set_xticks(positions, noise_labels)
        ax.set_xlabel("Relative Noise Level")

    position_by_noise = {v: i for i, v in enumerate(noise_values)}

    # Panel A: β and n merged (blue=β, grey=n), no jitter
    _errorbar(axes[0], BETA_COLOR, r"$\beta$", beta_means, beta_stds)
    _errorbar(axes[0], N_COLOR, r"$n$", n_means, n_stds)
    _xformat(axes[0])
    axes[0].set_ylabel("Relative Error")
    axes[0].set_title(r"$\beta$ and $n$ Recovery vs Noise")
    axes[0].legend(fontsize=9)
    _label(axes[0], 'A')

    # Panel B: combined parameter error + significance
    _errorbar(axes[1], COMBINED_COLOR, None, parameter_means, parameter_stds)
    _xformat(axes[1])
    axes[1].set_ylabel("Combined Parameter Error  0.5(|Δβ|/β + |Δn|/n)")
    axes[1].set_title("Combined Parameter Recovery vs Noise")
    parameter_tops = [m + (s if show_eb else 0.0) for m, s in zip(parameter_means, parameter_stds)]
    parameter_top_by_noise = {v: t for v, t in zip(noise_values, parameter_tops)}
    annotate_pairwise_comparisons(
        axes[1], x_positions=position_by_noise, top_values=parameter_top_by_noise,
        comparisons=noise_comparisons, significance=parameter_significance, use_adjusted_p_value=True)
    _label(axes[1], 'B')

    # Panel C: state RMSE + significance
    _errorbar(axes[2], COMBINED_COLOR, None, state_means, state_stds)
    _xformat(axes[2])
    axes[2].set_ylabel("State RMSE  (vs clean trajectory)")
    axes[2].set_title("Trajectory Reconstruction vs Noise")
    state_tops = [m + (s if show_eb else 0.0) for m, s in zip(state_means, state_stds)]
    state_top_by_noise = {v: t for v, t in zip(noise_values, state_tops)}
    annotate_pairwise_comparisons(
        axes[2], x_positions=position_by_noise, top_values=state_top_by_noise,
        comparisons=noise_comparisons, significance=state_significance, use_adjusted_p_value=True)
    _label(axes[2], 'C')

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
