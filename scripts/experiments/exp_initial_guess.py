"""Experiment 4: inverse-PINN sensitivity to initial parameter guesses."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from experiments.experiment_utils import aggregate_metrics, ensure_project_directories, finalize_figure, make_synthetic_dataset, write_csv
from pinn.run_inverse import run_inverse


TRUE_BETA = 5.0
TRUE_N = 3.0
NOISE_LEVEL = 0.05
BETA_GUESSES = [1.0, 3.0, 5.0, 7.0, 10.0]
N_GUESSES = [1.5, 2.0, 2.5, 3.0, 4.0]
SEEDS = [0, 1, 2, 3, 4]
TRAIN_ITERATIONS = 10000
RESULTS_DIR = "results/exp_initial_guess"
FIGURE_PATH = "figures/exp_initial_guess.png"


def main():
    ensure_project_directories()
    raw_rows = []

    for beta_guess in BETA_GUESSES:
        for n_guess in N_GUESSES:
            for seed in SEEDS:
                dataset = make_synthetic_dataset(TRUE_BETA, TRUE_N, noise_level=NOISE_LEVEL, seed=seed)
                result = run_inverse(
                    dataset_path=dataset,
                    outdir_base=os.path.join(RESULTS_DIR, "runs"),
                    C1_guess=beta_guess,
                    C2_guess=n_guess,
                    observation_stride=1,
                    observed_components=[0, 1, 2],
                    train_iterations=TRAIN_ITERATIONS,
                    random_seed=seed,
                    save_checkpoint=True,
                )
                raw_rows.append(
                    {
                        "beta_guess": beta_guess,
                        "n_guess": n_guess,
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
        group_keys=["beta_guess", "n_guess"],
        metric_keys=["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key=lambda row: (row["beta_guess"], row["n_guess"]))

    write_csv(
        os.path.join(RESULTS_DIR, "initial_guess_raw.csv"),
        raw_rows,
        ["beta_guess", "n_guess", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "initial_guess_summary.csv"),
        summary_rows,
        [
            "beta_guess",
            "n_guess",
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

    beta_heatmap = np.zeros((len(N_GUESSES), len(BETA_GUESSES)))
    n_heatmap = np.zeros((len(N_GUESSES), len(BETA_GUESSES)))
    for row in summary_rows:
        beta_index = BETA_GUESSES.index(row["beta_guess"])
        n_index = N_GUESSES.index(row["n_guess"])
        beta_heatmap[n_index, beta_index] = row["beta_rel_error_mean"]
        n_heatmap[n_index, beta_index] = row["n_rel_error_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    beta_image = axes[0].imshow(beta_heatmap, origin="lower", aspect="auto")
    axes[0].set_xticks(range(len(BETA_GUESSES)), [str(value) for value in BETA_GUESSES])
    axes[0].set_yticks(range(len(N_GUESSES)), [str(value) for value in N_GUESSES])
    axes[0].set_xlabel(r"Initial $\beta$")
    axes[0].set_ylabel(r"Initial $n$")
    axes[0].set_title(r"Relative error on $\beta$")
    fig.colorbar(beta_image, ax=axes[0])

    n_image = axes[1].imshow(n_heatmap, origin="lower", aspect="auto")
    axes[1].set_xticks(range(len(BETA_GUESSES)), [str(value) for value in BETA_GUESSES])
    axes[1].set_yticks(range(len(N_GUESSES)), [str(value) for value in N_GUESSES])
    axes[1].set_xlabel(r"Initial $\beta$")
    axes[1].set_ylabel(r"Initial $n$")
    axes[1].set_title(r"Relative error on $n$")
    fig.colorbar(n_image, ax=axes[1])

    finalize_figure(FIGURE_PATH)


if __name__ == "__main__":
    main()
