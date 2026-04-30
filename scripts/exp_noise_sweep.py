"""Experiment 1: inverse-PINN sensitivity to observation noise."""

import os

import matplotlib.pyplot as plt

from experiment_utils import aggregate_metrics, ensure_project_directories, finalize_figure, make_synthetic_dataset, write_csv
from run_inverse import run_inverse


TRUE_BETA = 5.0
TRUE_N = 3.0
NOISE_LEVELS = [0.01, 0.05, 0.10, 0.20, 0.30]
SEEDS = [0, 1, 2, 3, 4]
OBSERVED_COMPONENTS = [0, 1, 2]
TRAIN_ITERATIONS = 10000
RESULTS_DIR = "results/exp_noise_sweep"
FIGURE_PATH = "figures/exp_noise_sweep.png"


def main():
    ensure_project_directories()
    raw_rows = []

    for noise_level in NOISE_LEVELS:
        for seed in SEEDS:
            dataset = make_synthetic_dataset(TRUE_BETA, TRUE_N, noise_level=noise_level, seed=seed)
            result = run_inverse(
                dataset_path=dataset,
                outdir_base=os.path.join(RESULTS_DIR, "runs"),
                C1_guess=4.0,
                C2_guess=2.5,
                observation_stride=1,
                observed_components=OBSERVED_COMPONENTS,
                train_iterations=TRAIN_ITERATIONS,
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
        os.path.join(RESULTS_DIR, "noise_sweep_raw.csv"),
        raw_rows,
        ["noise_level", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "noise_sweep_summary.csv"),
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
    parameter_means = [row["parameter_rel_error_mean"] for row in summary_rows]
    parameter_stds = [row["parameter_rel_error_std"] for row in summary_rows]
    state_means = [row["state_rmse_mean"] for row in summary_rows]
    state_stds = [row["state_rmse_std"] for row in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].errorbar(noise_values, parameter_means, yerr=parameter_stds, marker="o", capsize=4)
    axes[0].set_xlabel("Relative noise level")
    axes[0].set_ylabel("Parameter recovery error")
    axes[0].set_title("Noise vs parameter recovery")

    axes[1].errorbar(noise_values, state_means, yerr=state_stds, marker="o", capsize=4)
    axes[1].set_xlabel("Relative noise level")
    axes[1].set_ylabel("State reconstruction RMSE")
    axes[1].set_title("Noise vs state reconstruction")

    finalize_figure(FIGURE_PATH)


if __name__ == "__main__":
    main()
