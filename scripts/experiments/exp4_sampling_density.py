"""
Experiment 4 — Sampling density.

Question: How sparse can the observations be before parameter recovery degrades?

Design:
  - True parameters: β = 5.0, n = 3.0 (oscillatory regime)
  - Fixed noise level: 0.05 (relative to peak-to-peak amplitude)
  - All three repressors observed (x1, x2, x3)
  - Observation counts: 10, 25, 50, 100 — evenly spaced across the 1000-point grid
    (as a percentage of the full grid: 1%, 2.5%, 5%, 10%)
  - 5 seeds per count; 10000 Adam iterations per run
  - Initial guesses: β₀ = 4.0, n₀ = 2.5

Figure: 2-panel layout (10×5), line plot with mean ± SD
  - Panel A: combined parameter error + significance brackets (vs highest-count baseline)
  - Panel B: state RMSE + significance brackets (vs highest-count baseline)

  Also produces a combined 2×2 figure (exp3_4_observability.png) that juxtaposes
  partial-observation (exp3) and sampling-density results side by side:
    - (0,0)/(0,1): Parameter Recovery — observed repressors / sampling points (of 1000)
    - (1,0)/(1,1): Trajectory Reconstruction — observed repressors / sampling points (of 1000)

Significance: two-sided Mann–Whitney U, Holm–Bonferroni corrected, vs highest-count baseline.

Key finding expected: sparse sampling (<25 points) removes complete oscillatory phases,
costing critical phase information for identifying n.  State RMSE degrades more gracefully
because the ODE residual provides a global constraint between observations.
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
    evenly_spaced_observation_indices,
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
observation_counts = [10, 25, 50, 100]
total_grid_points = 1000
seeds = [0, 1, 2, 3, 4]
train_iterations = 10000
results_dir = "results/exp4_sampling_density"
figure_path = "figures/exp4_sampling_density.png"

BETA_COLOR = "#009E73"
N_COLOR = "#D55E00"
COMBINED_COLOR = "#222222"
RMSE_COLOR = "#222222"


def main():
    ensure_project_directories()
    expected_runs = len(observation_counts) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp4_sampling_density",
            "script_path": __file__,
            "results_dir": results_dir,
            "figure_path": figure_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "observation_counts": list(observation_counts),
            "total_grid_points": total_grid_points,
            "true_beta": true_beta,
            "true_n": true_n,
            "noise_level": noise_level,
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )
    raw_rows = []

    for obs_count in observation_counts:
        for seed in seeds:
            dataset = make_synthetic_dataset(true_beta, true_n, noise_level=noise_level, seed=seed)
            observation_indices = evenly_spaced_observation_indices(
                len(dataset["t"]), obs_count
            )
            result = run_inverse(
                dataset_path=dataset,
                outdir_base=os.path.join(results_dir, "runs"),
                beta_guess=4.0,
                n_guess=2.5,
                observed_components=[0, 1, 2],
                train_iterations=train_iterations,
                observation_indices=observation_indices,
                random_seed=seed,
                save_checkpoint=True,
            )
            raw_rows.append(
                {
                    "observation_count": obs_count,
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
        group_keys=["observation_count"],
        metric_keys=["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key=lambda row: row["observation_count"])

    write_csv(
        os.path.join(results_dir, "exp4_sampling_density_raw.csv"),
        raw_rows,
        ["observation_count", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(results_dir, "exp4_sampling_density_summary.csv"),
        summary_rows,
        [
            "observation_count", "num_runs",
            "beta_rel_error_mean", "beta_rel_error_std",
            "n_rel_error_mean", "n_rel_error_std",
            "parameter_rel_error_mean", "parameter_rel_error_std",
            "state_rmse_mean", "state_rmse_std",
        ],
    )

    count_values = [row["observation_count"] for row in summary_rows]
    positions = list(range(len(count_values)))
    count_labels = [str(c) for c in count_values]

    beta_means = [row["beta_rel_error_mean"] for row in summary_rows]
    beta_stds = [row["beta_rel_error_std"] for row in summary_rows]
    n_means = [row["n_rel_error_mean"] for row in summary_rows]
    n_stds = [row["n_rel_error_std"] for row in summary_rows]
    parameter_means = [row["parameter_rel_error_mean"] for row in summary_rows]
    parameter_stds = [row["parameter_rel_error_std"] for row in summary_rows]
    state_means = [row["state_rmse_mean"] for row in summary_rows]
    state_stds = [row["state_rmse_std"] for row in summary_rows]
    show_eb = len(seeds) > 1

    baseline_count = max(count_values)
    count_comparisons = [(baseline_count, c) for c in count_values if c != baseline_count]

    beta_vals_by_count = metric_values_by_group(raw_rows, "observation_count", "beta_rel_error")
    n_vals_by_count = metric_values_by_group(raw_rows, "observation_count", "n_rel_error")
    parameter_vals_by_count = metric_values_by_group(raw_rows, "observation_count", "parameter_rel_error")
    state_vals_by_count = metric_values_by_group(raw_rows, "observation_count", "state_rmse")

    parameter_significance = pairwise_significance(parameter_vals_by_count, count_comparisons)
    state_significance = pairwise_significance(state_vals_by_count, count_comparisons)
    beta_significance = pairwise_significance(beta_vals_by_count, count_comparisons)
    n_significance = pairwise_significance(n_vals_by_count, count_comparisons)

    plt.rcParams['axes.formatter.useoffset'] = False

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
        ax.set_xticks(positions, count_labels)
        ax.set_xlabel("Observation Count")

    position_by_count = {c: i for i, c in enumerate(count_values)}

    # Panel A: combined parameter error + significance
    _errorbar(axes[0], COMBINED_COLOR, None, parameter_means, parameter_stds)
    _xformat(axes[0])
    axes[0].set_ylabel("Param. Error")
    axes[0].set_title("Combined Parameter Recovery")
    parameter_tops = {c: parameter_means[i] + (parameter_stds[i] if show_eb else 0.0)
                     for i, c in enumerate(count_values)}
    annotate_pairwise_comparisons(
        axes[0], x_positions=position_by_count, top_values=parameter_tops,
        comparisons=count_comparisons, significance=parameter_significance, use_adjusted_p_value=True)
    _label(axes[0], 'A')

    # Panel B: state RMSE + significance
    _errorbar(axes[1], COMBINED_COLOR, None, state_means, state_stds)
    _xformat(axes[1])
    axes[1].set_ylabel("State RMSE")
    axes[1].set_title("Trajectory Reconstruction")
    state_tops = {c: state_means[i] + (state_stds[i] if show_eb else 0.0)
                 for i, c in enumerate(count_values)}
    annotate_pairwise_comparisons(
        axes[1], x_positions=position_by_count, top_values=state_tops,
        comparisons=count_comparisons, significance=state_significance, use_adjusted_p_value=True)
    _label(axes[1], 'B')

    finalize_figure(figure_path)

    # Combined observability figure (exp3 + exp4) if exp3 results are available
    exp3_results_base = os.path.join(
        os.path.dirname(os.path.abspath(results_dir)),
        "exp3_partial_observation",
    )
    exp3_summary_path = os.path.join(exp3_results_base, "exp3_partial_observation_summary.csv")
    exp3_raw_path = os.path.join(exp3_results_base, "exp3_partial_observation_raw.csv")

    if os.path.exists(exp3_summary_path) and os.path.exists(exp3_raw_path):
        import csv as _csv

        def _load_csv_dicts(path, numeric_keys):
            rows = []
            with open(path) as f:
                for row in _csv.DictReader(f):
                    parsed = dict(row)
                    for k in numeric_keys:
                        if k in parsed and parsed[k]:
                            try:
                                parsed[k] = float(parsed[k])
                            except ValueError:
                                pass
                    rows.append(parsed)
            return rows

        exp3_summary = _load_csv_dicts(exp3_summary_path, [
            "parameter_rel_error_mean", "parameter_rel_error_std",
            "state_rmse_mean", "state_rmse_std",
        ])
        exp3_raw = _load_csv_dicts(exp3_raw_path, ["parameter_rel_error", "state_rmse"])

        exp3_design_order = ["1/3", "2/3", "3/3"]
        exp3_summary = [r for r in exp3_summary if r.get("design") in exp3_design_order]
        exp3_summary.sort(key=lambda r: exp3_design_order.index(r["design"]))

        if exp3_summary:
            exp3_param_means = [r["parameter_rel_error_mean"] for r in exp3_summary]
            exp3_param_stds = [r["parameter_rel_error_std"] for r in exp3_summary]
            exp3_state_means = [r["state_rmse_mean"] for r in exp3_summary]
            exp3_state_stds = [r["state_rmse_std"] for r in exp3_summary]
            exp3_designs = [r["design"] for r in exp3_summary]
            exp3_positions = list(range(len(exp3_designs)))
            exp3_pos_by_design = {d: i for i, d in enumerate(exp3_designs)}

            exp3_baseline = "3/3"
            exp3_comparisons = [(exp3_baseline, d) for d in exp3_design_order if d != exp3_baseline]
            exp3_param_vals = metric_values_by_group(exp3_raw, "design", "parameter_rel_error")
            exp3_state_vals = metric_values_by_group(exp3_raw, "design", "state_rmse")
            exp3_param_sig = pairwise_significance(exp3_param_vals, exp3_comparisons)
            exp3_state_sig = pairwise_significance(exp3_state_vals, exp3_comparisons)

            combined_figure_path = figure_path.replace("exp4_sampling_density", "exp3_4_observability")
            fig2, ax2 = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)

            def _label2(ax, letter):
                ax.text(-0.02, 1.04, letter, transform=ax.transAxes,
                        fontsize=15, fontweight='bold', va='bottom', ha='left', color='black', clip_on=False)

            def _eb2(ax, xpos, means, stds, color):
                ax.errorbar(
                    xpos, means, yerr=stds if show_eb else None,
                    fmt="o", linestyle="-", linewidth=1.0,
                    capsize=4 if show_eb else 0,
                    color=color, ecolor=color,
                    markerfacecolor=color, markeredgecolor=color,
                    zorder=4,
                )

            _eb2(ax2[0, 0], exp3_positions, exp3_param_means, exp3_param_stds, COMBINED_COLOR)
            exp3_param_tops = {d: exp3_param_means[i] + (exp3_param_stds[i] if show_eb else 0.0)
                              for i, d in enumerate(exp3_designs)}
            annotate_pairwise_comparisons(
                ax2[0, 0], x_positions=exp3_pos_by_design, top_values=exp3_param_tops,
                comparisons=exp3_comparisons, significance=exp3_param_sig, use_adjusted_p_value=True)
            ax2[0, 0].set_xticks(exp3_positions, exp3_designs)
            ax2[0, 0].set_xlabel("Observed Repressors")
            ax2[0, 0].set_ylabel("Param. Error")
            ax2[0, 0].set_title("Parameter Recovery")
            _label2(ax2[0, 0], 'A')

            _eb2(ax2[0, 1], positions, parameter_means, parameter_stds, COMBINED_COLOR)
            annotate_pairwise_comparisons(
                ax2[0, 1], x_positions=position_by_count, top_values=parameter_tops,
                comparisons=count_comparisons, significance=parameter_significance, use_adjusted_p_value=True)
            ax2[0, 1].set_xticks(positions, count_labels)
            ax2[0, 1].set_xlabel("Sampling Points (of 1000)")
            ax2[0, 1].set_ylabel("Param. Error")
            ax2[0, 1].set_title("Parameter Recovery")
            _label2(ax2[0, 1], 'B')

            _eb2(ax2[1, 0], exp3_positions, exp3_state_means, exp3_state_stds, COMBINED_COLOR)
            exp3_state_tops = {d: exp3_state_means[i] + (exp3_state_stds[i] if show_eb else 0.0)
                              for i, d in enumerate(exp3_designs)}
            annotate_pairwise_comparisons(
                ax2[1, 0], x_positions=exp3_pos_by_design, top_values=exp3_state_tops,
                comparisons=exp3_comparisons, significance=exp3_state_sig, use_adjusted_p_value=True)
            ax2[1, 0].set_xticks(exp3_positions, exp3_designs)
            ax2[1, 0].set_xlabel("Observed Repressors")
            ax2[1, 0].set_ylabel("State RMSE")
            ax2[1, 0].set_title("Trajectory Reconstruction")
            _label2(ax2[1, 0], 'C')

            _eb2(ax2[1, 1], positions, state_means, state_stds, COMBINED_COLOR)
            annotate_pairwise_comparisons(
                ax2[1, 1], x_positions=position_by_count, top_values=state_tops,
                comparisons=count_comparisons, significance=state_significance, use_adjusted_p_value=True)
            ax2[1, 1].set_xticks(positions, count_labels)
            ax2[1, 1].set_xlabel("Sampling Points (of 1000)")
            ax2[1, 1].set_ylabel("State RMSE")
            ax2[1, 1].set_title("Trajectory Reconstruction")
            _label2(ax2[1, 1], 'D')

            finalize_figure(combined_figure_path)


if __name__ == "__main__":
    main()
