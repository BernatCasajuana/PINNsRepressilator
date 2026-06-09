"""
Experiment 1: inverse sensitivity to observation noise.
Question: how does inverse-PINN recovery degrade as observation noise increases?
Design: all three repressors are observed, dense sampling is used, and the relative noise level is swept over `0.01, 0.05, 0.10`.
Output: a two-panel figure with parameter recovery error and state reconstruction error versus noise.
"""

# Import necessary libraries, utilities, and set up paths
import os
import sys
import matplotlib.pyplot as plt

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.experiment_utils import aggregate_metrics, ensure_project_directories, finalize_figure, make_synthetic_dataset, write_csv
from scripts.pinns.inverse import run_inverse

# Experiment parameters and output paths
true_beta = 5.0
true_n = 3.0
noise_levels = [0.01, 0.05, 0.10]
seeds = [0, 1]
observed_components = [0, 1, 2]
train_iterations = 10000
results_dir = "results/exp_noise_sweep"
figure_path = "figures/exp_noise_sweep.png"
parameter_color = "#1F77B4"
state_color = "#6C757D"

# Main experiment loop
def main():
    ensure_project_directories()
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
    noise_labels = [f"{noise_value:.3f}" for noise_value in noise_values]
    parameter_means = [row["parameter_rel_error_mean"] for row in summary_rows]
    parameter_stds = [row["parameter_rel_error_std"] for row in summary_rows]
    state_means = [row["state_rmse_mean"] for row in summary_rows]
    state_stds = [row["state_rmse_std"] for row in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    axes[0].errorbar(
        positions,
        parameter_means,
        yerr = parameter_stds,
        fmt = "o",
        linestyle = "none",
        capsize = 4,
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
        yerr = state_stds,
        fmt = "o",
        linestyle = "none",
        capsize = 4,
        color = state_color,
        ecolor = state_color,
        markerfacecolor = state_color,
        markeredgecolor = state_color,
    )
    axes[1].set_xticks(positions, noise_labels)
    axes[1].set_xlabel("Relative Noise Level")
    axes[1].set_ylabel("State Reconstruction RMSE")
    axes[1].set_title("Noise vs State Reconstruction")

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
