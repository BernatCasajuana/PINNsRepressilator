"""
Experiment 1: inverse sensitivity to observation noise.
Question: how does inverse-PINN recovery degrade as observation noise increases?
Design: all three repressors are observed, dense sampling is used, and the relative noise level is swept over `0.0, 0.01, 0.05, 0.10, 0.20`.
Output: a two-panel figure with parameter recovery error and state reconstruction error versus noise.
"""

# Import necessary libraries, utilities, and set up paths
import os
import sys
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

# Experiment parameters and output paths
true_beta = 5.0
true_n = 3.0
noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]
seeds = [0, 1, 2, 3, 4]
observed_components = [0, 1, 2]
train_iterations = 10000
results_dir = "results/exp_noise_sweep"
figure_path = "figures/exp_noise_sweep.png"
parameter_color = "#0072B2"
state_color = "#7F7F7F"

# Main experiment loop
def main():
    ensure_project_directories()
    expected_runs = len(noise_levels) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp_noise_sweep",
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
            dataset = make_synthetic_dataset(true_beta, true_n, noise_level = noise_level, seed = seed)
            result = run_inverse(
                dataset_path = dataset,
                outdir_base = os.path.join(results_dir, "runs"),
                C1_guess = 4.0,
                C2_guess = 2.5,
                observation_stride = 1,
                observed_components = observed_components,
                train_iterations = train_iterations,
                random_seed = seed,
                save_checkpoint = True,
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
        group_keys = ["noise_level"],
        metric_keys = ["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key = lambda row: row["noise_level"])

    write_csv(
        os.path.join(results_dir, "noise_sweep_raw.csv"),
        raw_rows,
        ["noise_level", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(results_dir, "noise_sweep_summary.csv"),
        summary_rows,
        [
            "noise_level",
            "num_runs",
            "beta_rel_error_mean",
            "beta_rel_error_std",
            "n_rel_error_mean",
            "n_rel_error_std",
            "parameter_rel_error_mean",
            "parameter_rel_error_std",
            "state_rmse_mean",
            "state_rmse_std",
        ],
    )

    noise_values = [row["noise_level"] for row in summary_rows]
    positions = list(range(len(noise_values)))
    noise_labels = [f"{noise_value:.2f}" for noise_value in noise_values]
    parameter_means = [row["parameter_rel_error_mean"] for row in summary_rows]
    parameter_stds = [row["parameter_rel_error_std"] for row in summary_rows]
    state_means = [row["state_rmse_mean"] for row in summary_rows]
    state_stds = [row["state_rmse_std"] for row in summary_rows]
    show_error_bars = len(seeds) > 1

    baseline_noise = 0.0 if 0.0 in noise_values else noise_values[0]
    noise_comparisons = [
        (baseline_noise, noise_value)
        for noise_value in noise_values
        if noise_value != baseline_noise
    ]
    parameter_values_by_noise = metric_values_by_group(raw_rows, "noise_level", "parameter_rel_error")
    state_values_by_noise = metric_values_by_group(raw_rows, "noise_level", "state_rmse")
    parameter_significance = pairwise_significance(parameter_values_by_noise, noise_comparisons)
    state_significance = pairwise_significance(state_values_by_noise, noise_comparisons)

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    axes[0].errorbar(
        positions,
        parameter_means,
        yerr = parameter_stds if show_error_bars else None,
        fmt = "o",
        linestyle = "-",
        linewidth = 1.5,
        capsize = 4 if show_error_bars else 0,
        color = parameter_color,
        ecolor = parameter_color,
        markerfacecolor = parameter_color,
        markeredgecolor = parameter_color,
    )
    axes[0].set_xticks(positions, noise_labels)
    axes[0].set_xlabel("Relative Noise Level")
    axes[0].set_ylabel("Parameter Recovery Error")
    axes[0].set_title("Noise vs Parameter Recovery")

    axes[1].errorbar(
        positions,
        state_means,
        yerr = state_stds if show_error_bars else None,
        fmt = "o",
        linestyle = "-",
        linewidth = 1.5,
        capsize = 4 if show_error_bars else 0,
        color = state_color,
        ecolor = state_color,
        markerfacecolor = state_color,
        markeredgecolor = state_color,
    )
    axes[1].set_xticks(positions, noise_labels)
    axes[1].set_xlabel("Relative Noise Level")
    axes[1].set_ylabel("State Reconstruction RMSE")
    axes[1].set_title("Noise vs State Reconstruction")

    parameter_tops = [
        parameter_mean + (parameter_std if show_error_bars else 0.0)
        for parameter_mean, parameter_std in zip(parameter_means, parameter_stds)
    ]
    state_tops = [
        state_mean + (state_std if show_error_bars else 0.0)
        for state_mean, state_std in zip(state_means, state_stds)
    ]
    position_by_noise = {noise_value: position for position, noise_value in zip(positions, noise_values)}
    parameter_top_by_noise = {noise_value: top for noise_value, top in zip(noise_values, parameter_tops)}
    state_top_by_noise = {noise_value: top for noise_value, top in zip(noise_values, state_tops)}
    annotate_pairwise_comparisons(
        axes[0],
        x_positions=position_by_noise,
        top_values=parameter_top_by_noise,
        comparisons=noise_comparisons,
        significance=parameter_significance,
        use_adjusted_p_value=True,
    )
    annotate_pairwise_comparisons(
        axes[1],
        x_positions=position_by_noise,
        top_values=state_top_by_noise,
        comparisons=noise_comparisons,
        significance=state_significance,
        use_adjusted_p_value=True,
    )

    for axis in axes:
        axis.xaxis.get_offset_text().set_visible(False)
        axis.yaxis.get_offset_text().set_visible(False)

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
