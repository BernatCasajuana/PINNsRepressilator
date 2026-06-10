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
from matplotlib.patches import Patch

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.experiment_utils import (
    aggregate_metrics,
    ensure_project_directories,
    finalize_figure,
    make_synthetic_dataset,
    write_csv,
    write_run_manifest,
)
from scripts.pinns.inverse import run_inverse

# Experiment parameters and output paths
regimes = [
    ("stable_beta5", {"regime": "stable", "beta": 5.0, "n": 1.5}),
    ("stable_beta8", {"regime": "stable", "beta": 8.0, "n": 1.5}),
    ("oscillatory_beta5", {"regime": "oscillatory", "beta": 5.0, "n": 3.0}),
    ("oscillatory_beta8", {"regime": "oscillatory", "beta": 8.0, "n": 3.0}),
]
noise_levels = [0.05]
seeds = [0, 1, 2]
train_iterations = 10000
results_dir = "results/exp_regime_comparison"
figure_path = "figures/exp_regime_comparison.png"
stable_color = "#0072B2"
oscillatory_color = "#E69F00"

# Main experiment loop
def main():
    ensure_project_directories()
    expected_runs = len(regimes) * len(noise_levels) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp_regime_comparison",
            "script_path": __file__,
            "results_dir": results_dir,
            "figure_path": figure_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "noise_levels": list(noise_levels),
            "regimes": [
                {
                    "case": case_name,
                    "regime": parameters["regime"],
                    "beta": parameters["beta"],
                    "n": parameters["n"],
                }
                for case_name, parameters in regimes
            ],
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )
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

    rows_by_case = {row["case"]: row for row in summary_rows}
    case_labels = []
    parameter_means = []
    parameter_stds = []
    state_means = []
    state_stds = []
    bar_colors = []

    for case_name, parameters in regimes:
        row = rows_by_case.get(case_name)
        if row is None:
            continue
        case_labels.append(rf"$\beta$={parameters['beta']:.1f}, $n$={parameters['n']:.1f}")
        parameter_means.append(row["parameter_rel_error_mean"])
        parameter_stds.append(row["parameter_rel_error_std"])
        state_means.append(row["state_rmse_mean"])
        state_stds.append(row["state_rmse_std"])
        bar_colors.append(stable_color if parameters["regime"] == "stable" else oscillatory_color)

    positions = list(range(len(case_labels)))
    show_error_bars = len(seeds) > 1

    fig, axes = plt.subplots(1, 2, figsize = (14, 5))
    axes[0].bar(
        positions,
        parameter_means,
        yerr = parameter_stds if show_error_bars else None,
        capsize = 4 if show_error_bars else 0,
        color = bar_colors,
        edgecolor = "black",
        linewidth = 0.5,
    )
    axes[0].set_xticks(positions, case_labels, rotation = 20, ha = "right")
    axes[0].set_ylabel("Parameter Recovery Error")
    axes[0].set_title("Parameter Recovery")

    axes[1].bar(
        positions,
        state_means,
        yerr = state_stds if show_error_bars else None,
        capsize = 4 if show_error_bars else 0,
        color = bar_colors,
        edgecolor = "black",
        linewidth = 0.5,
    )
    axes[1].set_xticks(positions, case_labels, rotation = 20, ha = "right")
    axes[1].set_ylabel("State Reconstruction RMSE")
    axes[1].set_title("State Reconstruction")

    legend_handles = [
        Patch(facecolor = stable_color, edgecolor = "black", linewidth = 0.5, label = "Stable"),
        Patch(facecolor = oscillatory_color, edgecolor = "black", linewidth = 0.5, label = "Oscillatory"),
    ]
    axes[0].legend(handles = legend_handles, title = "Regime")

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
