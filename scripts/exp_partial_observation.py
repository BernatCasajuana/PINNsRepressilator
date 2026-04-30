"""Experiment 2: inverse-PINN sensitivity to partial observation."""

import os

import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import aggregate_metrics, ensure_project_directories, finalize_figure, make_synthetic_dataset, write_csv
from run_inverse import run_inverse


TRUE_BETA = 5.0
TRUE_N = 3.0
NOISE_LEVEL = 0.05
SEEDS = [0, 1, 2, 3, 4]
OBSERVATION_DESIGNS = [
    ("x1,x2,x3", [0, 1, 2]),
    ("x1,x2", [0, 1]),
    ("x1,x3", [0, 2]),
    ("x1", [0]),
]
TRAIN_ITERATIONS = 10000
RESULTS_DIR = "results/exp_partial_observation"
FIGURE_PATH = "figures/exp_partial_observation.png"


def main():
    ensure_project_directories()
    raw_rows = []

    for design_name, observed_components in OBSERVATION_DESIGNS:
        for seed in SEEDS:
            dataset = make_synthetic_dataset(TRUE_BETA, TRUE_N, noise_level=NOISE_LEVEL, seed=seed)
            result = run_inverse(
                dataset_path=dataset,
                outdir_base=os.path.join(RESULTS_DIR, "runs"),
                C1_guess=4.0,
                C2_guess=2.5,
                observation_stride=1,
                observed_components=observed_components,
                train_iterations=TRAIN_ITERATIONS,
                random_seed=seed,
                save_checkpoint=True,
            )
            raw_rows.append(
                {
                    "design": design_name,
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
        group_keys=["design"],
        metric_keys=["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    design_order = [design_name for design_name, _ in OBSERVATION_DESIGNS]
    summary_rows.sort(key=lambda row: design_order.index(row["design"]))

    write_csv(
        os.path.join(RESULTS_DIR, "partial_observation_raw.csv"),
        raw_rows,
        ["design", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "partial_observation_summary.csv"),
        summary_rows,
        [
            "design",
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

    positions = np.arange(len(summary_rows))
    width = 0.35
    parameter_means = [row["parameter_rel_error_mean"] for row in summary_rows]
    parameter_stds = [row["parameter_rel_error_std"] for row in summary_rows]
    state_means = [row["state_rmse_mean"] for row in summary_rows]
    state_stds = [row["state_rmse_std"] for row in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(positions, parameter_means, yerr=parameter_stds, capsize=4)
    axes[0].set_xticks(positions, [row["design"] for row in summary_rows], rotation=20)
    axes[0].set_ylabel("Parameter recovery error")
    axes[0].set_title("Partial observation vs parameter recovery")

    axes[1].bar(positions, state_means, yerr=state_stds, capsize=4)
    axes[1].set_xticks(positions, [row["design"] for row in summary_rows], rotation=20)
    axes[1].set_ylabel("State reconstruction RMSE")
    axes[1].set_title("Partial observation vs state reconstruction")

    finalize_figure(FIGURE_PATH)


if __name__ == "__main__":
    main()
