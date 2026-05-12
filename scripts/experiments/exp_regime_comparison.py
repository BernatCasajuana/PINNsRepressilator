"""
Experiment 5: inverse sensitivity to dynamical regime. 
Question: does the dynamical regime change the difficulty of PINN recovery?
Design: a stable regime and an oscillatory regime are compared across multiple noise levels.
Output: a regime comparison figure for parameter recovery error and state reconstruction error.
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
regimes = [
    ("stable", {"beta": 5.0, "n": 1.5}),
    ("oscillatory", {"beta": 5.0, "n": 3.0}),
]
noise_levels = [0.05, 0.10, 0.20]
seeds = [0, 1]
train_iterations = 5000
results_dir = "results/exp_regime_comparison"
figure_path = "figures/exp_regime_comparison.png"

# Main experiment loop
def main():
    ensure_project_directories()
    raw_rows = []

    for regime_name, parameters in regimes:
        for noise_level in noise_levels:
            for seed in seeds:
                dataset = make_synthetic_dataset(
                    parameters["beta"],
                    parameters["n"],
                    noise_level = noise_level,
                    seed = seed,
                )
                result = run_inverse(
                    dataset_path = dataset,
                    outdir_base = os.path.join(results_dir, "runs"),
                    C1_guess = 4.0,
                    C2_guess = 2.5,
                    observation_stride = 1,
                    observed_components = [0, 1, 2],
                    train_iterations = train_iterations,
                    random_seed = seed,
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
        group_keys = ["regime", "noise_level"],
        metric_keys = ["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key = lambda row: (row["regime"], row["noise_level"]))

    write_csv(
        os.path.join(results_dir, "regime_comparison_raw.csv"),
        raw_rows,
        ["regime", "noise_level", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(results_dir, "regime_comparison_summary.csv"),
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

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    for regime_name, _ in regimes:
        regime_rows = [row for row in summary_rows if row["regime"] == regime_name]
        noise_values = [row["noise_level"] for row in regime_rows]
        parameter_means = [row["parameter_rel_error_mean"] for row in regime_rows]
        parameter_stds = [row["parameter_rel_error_std"] for row in regime_rows]
        state_means = [row["state_rmse_mean"] for row in regime_rows]
        state_stds = [row["state_rmse_std"] for row in regime_rows]

        axes[0].errorbar(noise_values, parameter_means, yerr = parameter_stds, marker = "o", capsize = 4, label = regime_name)
        axes[1].errorbar(noise_values, state_means, yerr = state_stds, marker = "o", capsize = 4, label = regime_name)

    axes[0].set_xlabel("Relative noise level")
    axes[0].set_ylabel("Parameter recovery error")
    axes[0].set_title("Regime comparison: parameter recovery")
    axes[0].legend()

    axes[1].set_xlabel("Relative noise level")
    axes[1].set_ylabel("State reconstruction RMSE")
    axes[1].set_title("Regime comparison: state reconstruction")
    axes[1].legend()

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
