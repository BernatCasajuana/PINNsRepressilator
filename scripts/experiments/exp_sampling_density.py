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

from experiments.experiment_utils import aggregate_metrics, ensure_project_directories, evenly_spaced_observation_indices, finalize_figure, make_synthetic_dataset, write_csv
from scripts.pinns.inverse import run_inverse

# Experiment parameters and output paths
true_beta = 5.0
true_n = 3.0
noise_level = 0.05
observation_counts = [10, 25, 100]
seeds = [0]
train_iterations = 3000
results_dir = "results/exp_sampling_density"
figure_path = "figures/exp_sampling_density.png"


def main():
    ensure_project_directories()
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

    observation_counts = [row["observation_count"] for row in summary_rows]
    parameter_means = [row["parameter_rel_error_mean"] for row in summary_rows]
    parameter_stds = [row["parameter_rel_error_std"] for row in summary_rows]
    state_means = [row["state_rmse_mean"] for row in summary_rows]
    state_stds = [row["state_rmse_std"] for row in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    axes[0].errorbar(observation_counts, parameter_means, yerr = parameter_stds, marker = "o", capsize = 4)
    axes[0].set_xlabel("Number of observation points")
    axes[0].set_ylabel("Parameter recovery error")
    axes[0].set_title("Sampling density vs parameter recovery")

    axes[1].errorbar(observation_counts, state_means, yerr = state_stds, marker = "o", capsize = 4)
    axes[1].set_xlabel("Number of observation points")
    axes[1].set_ylabel("State reconstruction RMSE")
    axes[1].set_title("Sampling density vs state reconstruction")

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
