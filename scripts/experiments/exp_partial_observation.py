"""
Experiment 2: inverse sensitivity to partial observation. 
Question: how much performance is lost when fewer repressors are measured?
Design: noise is fixed and four observation designs are compared: all three repressors, `x1,x2`, `x1,x3`, and `x1` only.
Output: a grouped comparison of parameter and state errors across observation designs.
"""

# Import necessary libraries, utilities, and set up paths
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.experiment_utils import aggregate_metrics, ensure_project_directories, finalize_figure, make_synthetic_dataset, write_csv
from scripts.pinns.inverse import run_inverse

# Experiment parameters and output paths
true_beta = 5.0
true_n = 3.0
noise_level = 0.05
seeds = [0, 1, 2, 3, 4]
observation_designs = [
    ("x1,x2,x3", [0, 1, 2]),
    ("x1,x2", [0, 1]),
    ("x1,x3", [0, 2]),
    ("x1", [0]),
]
train_iterations = 10000
results_dir = "results/exp_partial_observation"
figure_path = "figures/exp_partial_observation.png"

# Main experiment loop
def main():
    ensure_project_directories()
    raw_rows = []

    for design_name, observed_components in observation_designs:
        for seed in seeds:
            dataset = make_synthetic_dataset(true_beta, true_n, noise_level = noise_level, seed = seed)
            result = run_inverse(
                dataset_path = dataset,
                outdir_base = os.path.join(results_dir, "runs"),
                C1_guess = 4.0,
                C2_guess = 2.5,
                observation_stride = 1,
                observed_components = observed_components,
                train_iterations = train_iterations,
                random_seed = seed,
                save_checkpoint = True,
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
    design_order = [design_name for design_name, _ in observation_designs]
    summary_rows.sort(key = lambda row: design_order.index(row["design"]))

    write_csv(
        os.path.join(results_dir, "partial_observation_raw.csv"),
        raw_rows,
        ["design", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(results_dir, "partial_observation_summary.csv"),
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

    fig, axes = plt.subplots(1, 2, figsize = (13, 5))
    axes[0].bar(positions, parameter_means, yerr = parameter_stds, capsize = 4)
    axes[0].set_xticks(positions, [row["design"] for row in summary_rows], rotation = 20)
    axes[0].set_ylabel("Parameter recovery error")
    axes[0].set_title("Partial observation vs parameter recovery")

    axes[1].bar(positions, state_means, yerr = state_stds, capsize = 4)
    axes[1].set_xticks(positions, [row["design"] for row in summary_rows], rotation = 20)
    axes[1].set_ylabel("State reconstruction RMSE")
    axes[1].set_title("Partial observation vs state reconstruction")

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
