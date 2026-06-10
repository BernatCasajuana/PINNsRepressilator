"""
Experiment 3: inverse sensitivity to sampling density. 
Question: how sparse can the measurements be before recovery fails?
Design: noise is fixed, all three repressors are observed, and the number of observation points is varied over `10, 25, 100`.
Output: parameter and state errors versus the number of observation points.
"""

# Import necessary libraries, utilities, and set up paths
import os
import sys
import matplotlib.pyplot as plt

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from experiments.experiment_utils import (
    aggregate_metrics,
    ensure_project_directories,
    evenly_spaced_observation_indices,
    finalize_figure,
    make_synthetic_dataset,
    write_csv,
    write_run_manifest,
)
from scripts.pinns.inverse import run_inverse

# Experiment parameters and output paths
true_beta = 5.0
true_n = 3.0
noise_level = 0.05
observation_counts = [10, 25, 50, 100]
seeds = [0, 1, 2]
train_iterations = 10000
results_dir = "results/exp_sampling_density"
figure_path = "figures/exp_sampling_density.png"
parameter_color = "#0072B2"
state_color = "#E69F00"


def main():
    ensure_project_directories()
    expected_runs = len(observation_counts) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp_sampling_density",
            "script_path": __file__,
            "results_dir": results_dir,
            "figure_path": figure_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "observation_counts": list(observation_counts),
            "true_beta": true_beta,
            "true_n": true_n,
            "noise_level": noise_level,
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )
    raw_rows = []

    for observation_count in observation_counts:
        for seed in seeds:
            dataset = make_synthetic_dataset(true_beta, true_n, noise_level = noise_level, seed = seed)
            observation_indices = evenly_spaced_observation_indices(len(dataset["t"]), observation_count)
            result = run_inverse(
                dataset_path = dataset,
                outdir_base = os.path.join(results_dir, "runs"),
                C1_guess = 4.0,
                C2_guess = 2.5,
                observed_components = [0, 1, 2],
                train_iterations = train_iterations,
                observation_indices = observation_indices,
                random_seed = seed,
                save_checkpoint=True,
            )
            raw_rows.append(
                {
                    "observation_count": observation_count,
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
        group_keys = ["observation_count"],
        metric_keys = ["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key = lambda row: row["observation_count"])

    write_csv(
        os.path.join(results_dir, "sampling_density_raw.csv"),
        raw_rows,
        ["observation_count", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(results_dir, "sampling_density_summary.csv"),
        summary_rows,
        [
            "observation_count",
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

    observation_values = [row["observation_count"] for row in summary_rows]
    positions = list(range(len(observation_values)))
    observation_labels = [str(observation_value) for observation_value in observation_values]
    parameter_means = [row["parameter_rel_error_mean"] for row in summary_rows]
    parameter_stds = [row["parameter_rel_error_std"] for row in summary_rows]
    state_means = [row["state_rmse_mean"] for row in summary_rows]
    state_stds = [row["state_rmse_std"] for row in summary_rows]
    show_error_bars = len(seeds) > 1

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    axes[0].bar(
        positions,
        parameter_means,
        yerr = parameter_stds if show_error_bars else None,
        capsize = 4 if show_error_bars else 0,
        color = parameter_color,
        edgecolor = "black",
        linewidth = 0.5,
    )
    axes[0].set_xticks(positions, observation_labels)
    axes[0].set_xlabel("Number of Observation Points")
    axes[0].set_ylabel("Parameter Recovery Error")
    axes[0].set_title("Sampling Density vs Parameter Recovery")

    axes[1].bar(
        positions,
        state_means,
        yerr = state_stds if show_error_bars else None,
        capsize = 4 if show_error_bars else 0,
        color = state_color,
        edgecolor = "black",
        linewidth = 0.5,
    )
    axes[1].set_xticks(positions, observation_labels)
    axes[1].set_xlabel("Number of Observation Points")
    axes[1].set_ylabel("State Reconstruction RMSE")
    axes[1].set_title("Sampling Density vs State Reconstruction")

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
