"""
Experiment 2: inverse sensitivity to partial observation. 
Question: how much performance is lost when fewer repressors are measured?
Design: noise is fixed and three observation designs are compared: `x1,x2,x3`, `x1,x2`, and `x1`.
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

from experiments.experiment_utils import (
    aggregate_metrics,
    annotate_pairwise_comparisons,
    ensure_project_directories,
    finalize_figure,
    make_synthetic_dataset,
    metric_values_by_group,
    pairwise_significance,
    write_csv,
    write_run_manifest,
)
from scripts.pinns.inverse import run_inverse

# Experiment parameters and output paths
true_beta = 5.0
true_n = 3.0
noise_level = 0.05
seeds = [0, 1, 2, 3, 4]
observation_designs = [
    ("x1,x2,x3", [0, 1, 2]),
    ("x1,x2", [0, 1]),
    ("x1", [0]),
]
train_iterations = 10000
results_dir = "results/exp_partial_observation"
figure_path = "figures/exp_partial_observation.png"
design_colors = {
    "x1,x2,x3": "#0072B2",
    "x1,x2": "#7F7F7F",
    "x1": "#C7C7C7",
}

# Main experiment loop
def main():
    ensure_project_directories()
    expected_runs = len(observation_designs) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp_partial_observation",
            "script_path": __file__,
            "results_dir": results_dir,
            "figure_path": figure_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "observation_designs": [
                {"design": design_name, "observed_components": list(observed_components)}
                for design_name, observed_components in observation_designs
            ],
            "true_beta": true_beta,
            "true_n": true_n,
            "noise_level": noise_level,
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )
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
    bar_colors = [design_colors[row["design"]] for row in summary_rows]
    parameter_means = [row["parameter_rel_error_mean"] for row in summary_rows]
    parameter_stds = [row["parameter_rel_error_std"] for row in summary_rows]
    state_means = [row["state_rmse_mean"] for row in summary_rows]
    state_stds = [row["state_rmse_std"] for row in summary_rows]
    show_error_bars = len(seeds) > 1
    bar_width = 0.55

    baseline_design = design_order[0]
    design_comparisons = [(baseline_design, design_name) for design_name in design_order[1:]]
    parameter_values_by_design = metric_values_by_group(raw_rows, "design", "parameter_rel_error")
    state_values_by_design = metric_values_by_group(raw_rows, "design", "state_rmse")
    parameter_significance = pairwise_significance(parameter_values_by_design, design_comparisons)
    state_significance = pairwise_significance(state_values_by_design, design_comparisons)

    fig, axes = plt.subplots(1, 2, figsize = (13, 5))
    axes[0].bar(
        positions,
        parameter_means,
        width = bar_width,
        yerr = parameter_stds if show_error_bars else None,
        capsize = 4 if show_error_bars else 0,
        color = bar_colors,
        edgecolor = "black",
        linewidth = 0.5,
    )
    axes[0].set_xticks(positions, [row["design"] for row in summary_rows], rotation = 20)
    axes[0].set_xlabel("Observation Design")
    axes[0].set_ylabel("Parameter Recovery Error")
    axes[0].set_title("Partial Observation vs Parameter Recovery")

    axes[1].bar(
        positions,
        state_means,
        width = bar_width,
        yerr = state_stds if show_error_bars else None,
        capsize = 4 if show_error_bars else 0,
        color = bar_colors,
        edgecolor = "black",
        linewidth = 0.5,
    )
    axes[1].set_xticks(positions, [row["design"] for row in summary_rows], rotation = 20)
    axes[1].set_xlabel("Observation Design")
    axes[1].set_ylabel("State Reconstruction RMSE")
    axes[1].set_title("Partial Observation vs State Reconstruction")

    parameter_tops = [
        parameter_mean + (parameter_std if show_error_bars else 0.0)
        for parameter_mean, parameter_std in zip(parameter_means, parameter_stds)
    ]
    state_tops = [
        state_mean + (state_std if show_error_bars else 0.0)
        for state_mean, state_std in zip(state_means, state_stds)
    ]
    position_by_design = {design_name: position for position, design_name in zip(positions, design_order)}
    parameter_top_by_design = {design_name: top for design_name, top in zip(design_order, parameter_tops)}
    state_top_by_design = {design_name: top for design_name, top in zip(design_order, state_tops)}
    annotate_pairwise_comparisons(
        axes[0],
        x_positions=position_by_design,
        top_values=parameter_top_by_design,
        comparisons=design_comparisons,
        significance=parameter_significance,
        use_adjusted_p_value=True,
    )
    annotate_pairwise_comparisons(
        axes[1],
        x_positions=position_by_design,
        top_values=state_top_by_design,
        comparisons=design_comparisons,
        significance=state_significance,
        use_adjusted_p_value=True,
    )

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
