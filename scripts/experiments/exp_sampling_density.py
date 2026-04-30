"""Experiment 3: inverse-PINN sensitivity to sampling density."""

import os
import sys

import matplotlib.pyplot as plt

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from experiments.experiment_utils import aggregate_metrics, ensure_project_directories, evenly_spaced_observation_indices, finalize_figure, make_synthetic_dataset, write_csv
from pinn.run_inverse import run_inverse


TRUE_BETA = 5.0
TRUE_N = 3.0
NOISE_LEVEL = 0.05
OBSERVATION_COUNTS = [10, 25, 50, 100, 200]
SEEDS = [0, 1, 2, 3, 4]
TRAIN_ITERATIONS = 10000
RESULTS_DIR = "results/exp_sampling_density"
FIGURE_PATH = "figures/exp_sampling_density.png"


def main():
    ensure_project_directories()
    raw_rows = []

    for observation_count in OBSERVATION_COUNTS:
        for seed in SEEDS:
            dataset = make_synthetic_dataset(TRUE_BETA, TRUE_N, noise_level=NOISE_LEVEL, seed=seed)
            observation_indices = evenly_spaced_observation_indices(len(dataset["t"]), observation_count)
            result = run_inverse(
                dataset_path=dataset,
                outdir_base=os.path.join(RESULTS_DIR, "runs"),
                C1_guess=4.0,
                C2_guess=2.5,
                observed_components=[0, 1, 2],
                train_iterations=TRAIN_ITERATIONS,
                observation_indices=observation_indices,
                random_seed=seed,
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
        group_keys=["observation_count"],
        metric_keys=["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key=lambda row: row["observation_count"])

    write_csv(
        os.path.join(RESULTS_DIR, "sampling_density_raw.csv"),
        raw_rows,
        ["observation_count", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "sampling_density_summary.csv"),
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].errorbar(observation_counts, parameter_means, yerr=parameter_stds, marker="o", capsize=4)
    axes[0].set_xlabel("Number of observation points")
    axes[0].set_ylabel("Parameter recovery error")
    axes[0].set_title("Sampling density vs parameter recovery")

    axes[1].errorbar(observation_counts, state_means, yerr=state_stds, marker="o", capsize=4)
    axes[1].set_xlabel("Number of observation points")
    axes[1].set_ylabel("State reconstruction RMSE")
    axes[1].set_title("Sampling density vs state reconstruction")

    finalize_figure(FIGURE_PATH)


if __name__ == "__main__":
    main()
