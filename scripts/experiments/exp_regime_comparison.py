"""
Experiment 5: inverse sensitivity to dynamical regime. 
Question: does the dynamical regime change the difficulty of PINN recovery?
Design: each regime (stable and oscillatory) is evaluated at two beta values with fixed n.
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
    ("stable_beta5", {"regime": "stable", "beta": 5.0, "n": 1.5}),
    ("stable_beta8", {"regime": "stable", "beta": 8.0, "n": 1.5}),
    ("oscillatory_beta5", {"regime": "oscillatory", "beta": 5.0, "n": 3.0}),
    ("oscillatory_beta8", {"regime": "oscillatory", "beta": 8.0, "n": 3.0}),
]
noise_levels = [0.05]
seeds = [0, 1]
train_iterations = 10000
results_dir = "results/exp_regime_comparison"
figure_path = "figures/exp_regime_comparison.png"

# Main experiment loop
def main():
    ensure_project_directories()
    raw_rows = []

    for case_name, parameters in regimes:
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
                        "case": case_name,
                        "regime": parameters["regime"],
                        "beta": parameters["beta"],
                        "n": parameters["n"],
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
        group_keys = ["case", "regime", "beta", "n", "noise_level"],
        metric_keys = ["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    case_order = [case_name for case_name, _ in regimes]
    summary_rows.sort(key = lambda row: (case_order.index(row["case"]), row["noise_level"]))

    write_csv(
        os.path.join(results_dir, "regime_comparison_raw.csv"),
        raw_rows,
        [
            "case",
            "regime",
            "beta",
            "n",
            "noise_level",
            "seed",
            "beta_rel_error",
            "n_rel_error",
            "parameter_rel_error",
            "state_rmse",
            "outdir",
        ],
    )
    write_csv(
        os.path.join(results_dir, "regime_comparison_summary.csv"),
        summary_rows,
        [
            "case",
            "regime",
            "beta",
            "n",
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
    for case_name, parameters in regimes:
        case_rows = [row for row in summary_rows if row["case"] == case_name]
        noise_values = [row["noise_level"] for row in case_rows]
        parameter_means = [row["parameter_rel_error_mean"] for row in case_rows]
        parameter_stds = [row["parameter_rel_error_std"] for row in case_rows]
        state_means = [row["state_rmse_mean"] for row in case_rows]
        state_stds = [row["state_rmse_std"] for row in case_rows]
        label = f"{parameters['regime']} (beta={parameters['beta']}, n={parameters['n']})"

        axes[0].errorbar(noise_values, parameter_means, yerr = parameter_stds, marker = "o", capsize = 4, label = label)
        axes[1].errorbar(noise_values, state_means, yerr = state_stds, marker = "o", capsize = 4, label = label)

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
