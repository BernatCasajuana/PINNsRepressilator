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

Figure: 4-panel 2×2 layout with bars + individual seed dots (strip plot overlay)
  - (0,0): β relative error per design
  - (0,1): n relative error per design
  - (1,0): combined parameter error + significance brackets (vs 3/3 baseline)
  - (1,1): state RMSE + significance brackets (vs 3/3 baseline)

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

    design_order = [label for label, _ in observation_designs]
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

    baseline_design = design_order[0]
    design_comparisons = [(baseline_design, d) for d in design_order[1:]]
    beta_vals_by_design = metric_values_by_group(raw_rows, "design", "beta_rel_error")
    n_vals_by_design = metric_values_by_group(raw_rows, "design", "n_rel_error")
    parameter_vals_by_design = metric_values_by_group(raw_rows, "design", "parameter_rel_error")
    state_vals_by_design = metric_values_by_group(raw_rows, "design", "state_rmse")
    beta_significance = pairwise_significance(beta_vals_by_design, design_comparisons)
    n_significance = pairwise_significance(n_vals_by_design, design_comparisons)
    parameter_significance = pairwise_significance(parameter_vals_by_design, design_comparisons)
    state_significance = pairwise_significance(state_vals_by_design, design_comparisons)

    plt.rcParams['axes.formatter.useoffset'] = False

    # 3 categories: blue (3/3 best), mid-grey (2/3), light-grey (1/3 worst)
    DESIGN_COLORS = {"3/3": "#444444", "2/3": "#888888", "1/3": "#CCCCCC"}
    design_bar_colors = [DESIGN_COLORS[d] for d in design_order]
    BLUE = "#4C78A8"
    GREY = "#F58518"

    x = np.arange(len(design_order))
    bar_w = 0.35
    bar_wide = 0.55
    rng = np.random.default_rng(42)
    jw = 0.10

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        rf"Partial Observation ($\beta$={true_beta}, $n$={true_n}, $\sigma$={noise_level}, {len(seeds)} seeds)",
        fontsize=13,
    )

    def _label(ax, letter):
        ax.text(-0.02, 1.10, letter, transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='bottom', ha='left', color='black', clip_on=False)

    def _strip(ax, x_base, vals):
        j = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(x_base + j, vals, color="black", alpha=0.65, s=18, zorder=5, linewidths=0)

    def _xformat(ax):
        ax.set_xticks(x, design_order)
        ax.set_xlabel("Observed Repressors")

    position_by_design = {d: i for i, d in enumerate(design_order)}

    # Panel A: β and n grouped bars (blue=β, grey=n), strip dots
    axes[0].bar(x - bar_w / 2, beta_means, width=bar_w,
                yerr=beta_stds if show_eb else None, capsize=4 if show_eb else 0,
                color=BLUE, edgecolor="white", linewidth=0.5, label=r"$\beta$", zorder=3)
    axes[0].bar(x + bar_w / 2, n_means, width=bar_w,
                yerr=n_stds if show_eb else None, capsize=4 if show_eb else 0,
                color=GREY, edgecolor="white", linewidth=0.5, label=r"$n$", zorder=3)
    for i, d in enumerate(design_order):
        _strip(axes[0], i - bar_w / 2, beta_vals_by_design.get(d, []))
        _strip(axes[0], i + bar_w / 2, n_vals_by_design.get(d, []))
    _xformat(axes[0])
    axes[0].set_ylabel("Relative Error")
    axes[0].set_title(r"$\beta$ and $n$ recovery vs Observation Design")
    axes[0].legend(fontsize=9)
    _label(axes[0], 'A')
    beta_pos_by_design = {d: i - bar_w / 2 for i, d in enumerate(design_order)}
    n_pos_by_design = {d: i + bar_w / 2 for i, d in enumerate(design_order)}
    beta_tops = [m + (s if show_eb else 0.0) for m, s in zip(beta_means, beta_stds)]
    n_tops = [m + (s if show_eb else 0.0) for m, s in zip(n_means, n_stds)]
    beta_top_by_design = {d: t for d, t in zip(design_order, beta_tops)}
    n_top_by_design = {d: t for d, t in zip(design_order, n_tops)}
    annotate_pairwise_comparisons(
        axes[0], x_positions=beta_pos_by_design, top_values=beta_top_by_design,
        comparisons=design_comparisons, significance=beta_significance, use_adjusted_p_value=True)
    annotate_pairwise_comparisons(
        axes[0], x_positions=n_pos_by_design, top_values=n_top_by_design,
        comparisons=design_comparisons, significance=n_significance, use_adjusted_p_value=True)

    # Panel B: combined error (3-category bar colors) + significance + strip
    axes[1].bar(x, parameter_means, width=bar_wide,
                yerr=parameter_stds if show_eb else None, capsize=4 if show_eb else 0,
                color=design_bar_colors, edgecolor="white", linewidth=0.5, zorder=3)
    for i, d in enumerate(design_order):
        _strip(axes[1], i, parameter_vals_by_design.get(d, []))
    _xformat(axes[1])
    axes[1].set_ylabel("Combined Parameter Error  0.5(|Δβ|/β + |Δn|/n)")
    axes[1].set_title("Combined Parameter Recovery vs Observation Design")
    parameter_tops = [m + (s if show_eb else 0.0) for m, s in zip(parameter_means, parameter_stds)]
    parameter_top_by_design = {d: t for d, t in zip(design_order, parameter_tops)}
    annotate_pairwise_comparisons(
        axes[1], x_positions=position_by_design, top_values=parameter_top_by_design,
        comparisons=design_comparisons, significance=parameter_significance, use_adjusted_p_value=True)
    _label(axes[1], 'B')

    # Panel C: state RMSE (3-category bar colors) + significance + strip
    axes[2].bar(x, state_means, width=bar_wide,
                yerr=state_stds if show_eb else None, capsize=4 if show_eb else 0,
                color=design_bar_colors, edgecolor="white", linewidth=0.5, zorder=3)
    for i, d in enumerate(design_order):
        _strip(axes[2], i, state_vals_by_design.get(d, []))
    _xformat(axes[2])
    axes[2].set_ylabel("State RMSE  (vs clean trajectory)")
    axes[2].set_title("Trajectory Reconstruction vs Observation Design")
    state_tops = [m + (s if show_eb else 0.0) for m, s in zip(state_means, state_stds)]
    state_top_by_design = {d: t for d, t in zip(design_order, state_tops)}
    annotate_pairwise_comparisons(
        axes[2], x_positions=position_by_design, top_values=state_top_by_design,
        comparisons=design_comparisons, significance=state_significance, use_adjusted_p_value=True)
    _label(axes[2], 'C')

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
