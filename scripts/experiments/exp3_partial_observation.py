"""
Experiment 3 — Partial observation.

Question: How much performance is lost when only a subset of repressors is measured?

Design:
  - True parameters: β = 5.0, n = 3.0 (oscillatory regime)
  - Fixed noise level: 0.05 (relative to peak-to-peak amplitude)
  - Dense sampling (stride = 1, 1000 points); 5 seeds per design
  - Three observation designs compared:
      "3/3"  components [0, 1, 2]:  all repressors observed  — fully constrained system
      "2/3"  components [0, 1]:     two repressors observed  — x3 inferred from ODE
      "1/3"  component  [0]:        one repressor observed   — x2, x3 inferred from ODE
  - Initial guesses: β₀ = 4.0, n₀ = 2.5

Figure: 2-panel layout (10×5), line plot with mean ± SD
  - Panel A: combined parameter error + significance brackets (vs 3/3 baseline)
  - Panel B: state RMSE + significance brackets (vs 3/3 baseline)

  A combined 2×2 figure (exp3_4_observability) is also produced by exp4_sampling_density.py,
  placing partial-observation and sampling-density panels side by side for the paper.

Significance: two-sided Mann–Whitney U, Holm–Bonferroni corrected, vs full-observation baseline.

Key finding expected: losing the phase-coupled third repressor (x3) inflates variance
sharply because the Hill coefficient n is identified largely through the phase relationship
between repressors.  A single observed repressor typically fails to identify n reliably.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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

true_beta = 5.0
true_n = 3.0
noise_level = 0.05
seeds = [0, 1, 2, 3, 4]
# (display_label, observed_component_indices)
observation_designs = [
    ("3/3", [0, 1, 2]),
    ("2/3", [0, 1]),
    ("1/3", [0]),
]
train_iterations = 10000
results_dir = "results/exp3_partial_observation"
figure_path = "figures/exp3_partial_observation.png"

COLORS = {
    "3/3": "#0072B2",
    "2/3": "#56B4E9",
    "1/3": "#BBBBBB",
}


def main():
    ensure_project_directories()
    expected_runs = len(observation_designs) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp3_partial_observation",
            "script_path": __file__,
            "results_dir": results_dir,
            "figure_path": figure_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "observation_designs": [
                {"label": label, "observed_components": list(components)}
                for label, components in observation_designs
            ],
            "true_beta": true_beta,
            "true_n": true_n,
            "noise_level": noise_level,
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )
    raw_rows = []

    for design_label, observed_components in observation_designs:
        for seed in seeds:
            dataset = make_synthetic_dataset(true_beta, true_n, noise_level=noise_level, seed=seed)
            result = run_inverse(
                dataset_path=dataset,
                outdir_base=os.path.join(results_dir, "runs"),
                beta_guess=4.0,
                n_guess=2.5,
                observation_stride=1,
                observed_components=observed_components,
                train_iterations=train_iterations,
                random_seed=seed,
                save_checkpoint=True,
            )
            raw_rows.append(
                {
                    "design": design_label,
                    "seed": seed,
                    "beta_rel_error": result["beta_rel_error"],
                    "n_rel_error": result["n_rel_error"],
                    "parameter_rel_error": result["parameter_rel_error"],
                    "state_rmse": result["state_rmse"],
                    "outdir": result["outdir"],
                }
            )

    design_order = list(reversed([label for label, _ in observation_designs]))
    summary_rows = aggregate_metrics(
        raw_rows,
        group_keys=["design"],
        metric_keys=["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key=lambda row: design_order.index(row["design"]))

    write_csv(
        os.path.join(results_dir, "exp3_partial_observation_raw.csv"),
        raw_rows,
        ["design", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(results_dir, "exp3_partial_observation_summary.csv"),
        summary_rows,
        [
            "design", "num_runs",
            "beta_rel_error_mean", "beta_rel_error_std",
            "n_rel_error_mean", "n_rel_error_std",
            "parameter_rel_error_mean", "parameter_rel_error_std",
            "state_rmse_mean", "state_rmse_std",
        ],
    )

    positions = np.arange(len(summary_rows))
    bar_colors = [COLORS[row["design"]] for row in summary_rows]
    beta_means = [row["beta_rel_error_mean"] for row in summary_rows]
    beta_stds = [row["beta_rel_error_std"] for row in summary_rows]
    n_means = [row["n_rel_error_mean"] for row in summary_rows]
    n_stds = [row["n_rel_error_std"] for row in summary_rows]
    parameter_means = [row["parameter_rel_error_mean"] for row in summary_rows]
    parameter_stds = [row["parameter_rel_error_std"] for row in summary_rows]
    state_means = [row["state_rmse_mean"] for row in summary_rows]
    state_stds = [row["state_rmse_std"] for row in summary_rows]
    show_eb = len(seeds) > 1
    bar_width = 0.55

    baseline_design = "3/3"
    design_comparisons = [(baseline_design, d) for d in design_order if d != baseline_design]
    beta_vals_by_design = metric_values_by_group(raw_rows, "design", "beta_rel_error")
    n_vals_by_design = metric_values_by_group(raw_rows, "design", "n_rel_error")
    parameter_vals_by_design = metric_values_by_group(raw_rows, "design", "parameter_rel_error")
    state_vals_by_design = metric_values_by_group(raw_rows, "design", "state_rmse")
    beta_significance = pairwise_significance(beta_vals_by_design, design_comparisons)
    n_significance = pairwise_significance(n_vals_by_design, design_comparisons)
    parameter_significance = pairwise_significance(parameter_vals_by_design, design_comparisons)
    state_significance = pairwise_significance(state_vals_by_design, design_comparisons)

    plt.rcParams['axes.formatter.useoffset'] = False

    COMBINED_COLOR = "#222222"

    positions = list(range(len(design_order)))
    position_by_design = {d: i for i, d in enumerate(design_order)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    def _label(ax, letter):
        ax.text(-0.02, 1.04, letter, transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='bottom', ha='left', color='black', clip_on=False)

    def _errorbar(ax, color, label, means, stds):
        ax.errorbar(
            positions, means,
            yerr=stds if show_eb else None,
            fmt="o", linestyle="-", linewidth=1.0,
            capsize=4 if show_eb else 0,
            color=color, ecolor=color,
            markerfacecolor=color, markeredgecolor=color,
            zorder=4, label=label,
        )

    def _xformat(ax):
        ax.set_xticks(positions, design_order)
        ax.set_xlabel("Observed Repressors")

    # Panel A: combined parameter error + significance
    _errorbar(axes[0], COMBINED_COLOR, None, parameter_means, parameter_stds)
    _xformat(axes[0])
    axes[0].set_ylabel("Param. Error")
    axes[0].set_title("Combined Parameter Recovery")
    parameter_tops = {d: parameter_means[i] + (parameter_stds[i] if show_eb else 0.0)
                     for i, d in enumerate(design_order)}
    annotate_pairwise_comparisons(
        axes[0], x_positions=position_by_design, top_values=parameter_tops,
        comparisons=design_comparisons, significance=parameter_significance, use_adjusted_p_value=True)
    _label(axes[0], 'A')

    # Panel B: state RMSE + significance
    _errorbar(axes[1], COMBINED_COLOR, None, state_means, state_stds)
    _xformat(axes[1])
    axes[1].set_ylabel("State RMSE")
    axes[1].set_title("Trajectory Reconstruction")
    state_tops = {d: state_means[i] + (state_stds[i] if show_eb else 0.0)
                 for i, d in enumerate(design_order)}
    annotate_pairwise_comparisons(
        axes[1], x_positions=position_by_design, top_values=state_tops,
        comparisons=design_comparisons, significance=state_significance, use_adjusted_p_value=True)
    _label(axes[1], 'B')

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
