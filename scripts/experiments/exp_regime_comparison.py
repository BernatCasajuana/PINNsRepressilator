"""Experiment 5: inverse-PINN comparison between stable and oscillatory regimes."""

import os
import sys

import matplotlib.pyplot as plt

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from experiments.experiment_utils import aggregate_metrics, ensure_project_directories, finalize_figure, make_synthetic_dataset, write_csv
from pinns.run_inverse import run_inverse


REGIMES = [
    ("stable", {"beta": 5.0, "n": 1.5}),
    ("oscillatory", {"beta": 5.0, "n": 3.0}),
]
NOISE_LEVELS = [0.05, 0.10, 0.20]
SEEDS = [0, 1, 2, 3, 4]
TRAIN_ITERATIONS = 10000
RESULTS_DIR = "results/exp_regime_comparison"
FIGURE_PATH = "figures/exp_regime_comparison.png"


def main():
    ensure_project_directories()
    raw_rows = []

    for regime_name, parameters in REGIMES:
        for noise_level in NOISE_LEVELS:
            for seed in SEEDS:
                dataset = make_synthetic_dataset(
                    parameters["beta"],
                    parameters["n"],
                    noise_level=noise_level,
                    seed=seed,
                )
                result = run_inverse(
                    dataset_path=dataset,
                    outdir_base=os.path.join(RESULTS_DIR, "runs"),
                    C1_guess=4.0,
                    C2_guess=2.5,
                    observation_stride=1,
                    observed_components=[0, 1, 2],
                    train_iterations=TRAIN_ITERATIONS,
                    random_seed=seed,
                    save_checkpoint=True,
                )
                raw_rows.append(
                    {
                        "regime": regime_name,
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
        group_keys=["regime", "noise_level"],
        metric_keys=["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key=lambda row: (row["regime"], row["noise_level"]))

    write_csv(
        os.path.join(RESULTS_DIR, "regime_comparison_raw.csv"),
        raw_rows,
        ["regime", "noise_level", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(RESULTS_DIR, "regime_comparison_summary.csv"),
        summary_rows,
        [
            "regime",
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for regime_name, _ in REGIMES:
        regime_rows = [row for row in summary_rows if row["regime"] == regime_name]
        noise_values = [row["noise_level"] for row in regime_rows]
        parameter_means = [row["parameter_rel_error_mean"] for row in regime_rows]
        parameter_stds = [row["parameter_rel_error_std"] for row in regime_rows]
        state_means = [row["state_rmse_mean"] for row in regime_rows]
        state_stds = [row["state_rmse_std"] for row in regime_rows]

        axes[0].errorbar(noise_values, parameter_means, yerr=parameter_stds, marker="o", capsize=4, label=regime_name)
        axes[1].errorbar(noise_values, state_means, yerr=state_stds, marker="o", capsize=4, label=regime_name)

    axes[0].set_xlabel("Relative noise level")
    axes[0].set_ylabel("Parameter recovery error")
    axes[0].set_title("Regime comparison: parameter recovery")
    axes[0].legend()

    axes[1].set_xlabel("Relative noise level")
    axes[1].set_ylabel("State reconstruction RMSE")
    axes[1].set_title("Regime comparison: state reconstruction")
    axes[1].legend()

    finalize_figure(FIGURE_PATH)


if __name__ == "__main__":
    main()
