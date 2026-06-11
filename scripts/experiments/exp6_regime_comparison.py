"""
Experiment 6 — Dynamical regime comparison.

Question: Does the dynamical regime (stable vs oscillatory) affect the difficulty of
inverse-PINN parameter recovery?

Design:
  - Fixed noise level: 0.05; all three repressors observed; stride = 1 (dense)
  - 5 seeds per case; 10000 Adam iterations per run
  - Initial guesses: β₀ = 4.0, n₀ = 2.5 (same for all cases)
  - Four cases spanning two regimes and two β values:
      stable_beta5:       β = 5.0, n = 1.5  — steady state,  lower production
      stable_beta8:       β = 8.0, n = 1.5  — steady state,  higher production
      oscillatory_beta5:  β = 5.0, n = 3.0  — sustained oscillations, lower production
      oscillatory_beta8:  β = 8.0, n = 3.0  — sustained oscillations, higher production
  - The stable/oscillatory boundary is ~n = 2.0 for β = 5.0 (Müller et al.)

Figure: 2-panel layout (15×6) with bars (edgecolor black) + individual seed dots (strip plot overlay)
  - Panel A: combined parameter error for all four cases; legend shows Stable / Oscillatory
  - Panel B: state RMSE for all four cases
  Each panel has a small inset trajectory showing a representative stable or oscillatory time-series.
  Significance: stable_beta5 vs oscillatory_beta5 and stable_beta8 vs oscillatory_beta8.

Key finding expected: the stable regime produces lower RMSE because trajectories are
smoother; however, oscillatory dynamics provide richer information for n recovery —
the oscillatory regime may show lower *parameter* error despite higher *trajectory* error.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
    simulate_repressilator,
    write_csv,
    write_run_manifest,
)
from scripts.pinns.inverse import run_inverse

regimes = [
    ("stable_beta5",      {"regime": "stable",      "beta": 5.0, "n": 1.5}),
    ("stable_beta8",      {"regime": "stable",      "beta": 8.0, "n": 1.5}),
    ("oscillatory_beta5", {"regime": "oscillatory", "beta": 5.0, "n": 3.0}),
    ("oscillatory_beta8", {"regime": "oscillatory", "beta": 8.0, "n": 3.0}),
]
noise_level = 0.05
seeds = [0, 1, 2, 3, 4]
train_iterations = 10000
results_dir = "results/exp6_regime_comparison"
figure_path = "figures/exp6_regime_comparison.png"

STABLE_COLOR = "#0072B2"
OSCILLATORY_COLOR = "#E69F00"
TRAJ_COLORS = ("#0072B2", "#E69F00", "#009E73")


def main():
    ensure_project_directories()
    expected_runs = len(regimes) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp6_regime_comparison",
            "script_path": __file__,
            "results_dir": results_dir,
            "figure_path": figure_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "noise_level": noise_level,
            "regimes": [
                {"case": name, "regime": p["regime"], "beta": p["beta"], "n": p["n"]}
                for name, p in regimes
            ],
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )
    raw_rows = []

    for case_name, params in regimes:
        for seed in seeds:
            dataset = make_synthetic_dataset(
                params["beta"], params["n"], noise_level=noise_level, seed=seed
            )
            result = run_inverse(
                dataset_path=dataset,
                outdir_base=os.path.join(results_dir, "runs"),
                beta_guess=4.0,
                n_guess=2.5,
                observation_stride=1,
                observed_components=[0, 1, 2],
                train_iterations=train_iterations,
                random_seed=seed,
                save_checkpoint=True,
            )
            raw_rows.append(
                {
                    "case": case_name,
                    "regime": params["regime"],
                    "beta": params["beta"],
                    "n": params["n"],
                    "noise_level": noise_level,
                    "seed": seed,
                    "beta_rel_error": result["beta_rel_error"],
                    "n_rel_error": result["n_rel_error"],
                    "parameter_rel_error": result["parameter_rel_error"],
                    "state_rmse": result["state_rmse"],
                    "outdir": result["outdir"],
                }
            )

    case_order = [name for name, _ in regimes]
    summary_rows = aggregate_metrics(
        raw_rows,
        group_keys=["case", "regime", "beta", "n", "noise_level"],
        metric_keys=["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key=lambda row: case_order.index(row["case"]))

    write_csv(
        os.path.join(results_dir, "exp6_regime_comparison_raw.csv"),
        raw_rows,
        ["case", "regime", "beta", "n", "noise_level", "seed",
         "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(results_dir, "exp6_regime_comparison_summary.csv"),
        summary_rows,
        ["case", "regime", "beta", "n", "noise_level", "num_runs",
         "beta_rel_error_mean", "beta_rel_error_std",
         "n_rel_error_mean", "n_rel_error_std",
         "parameter_rel_error_mean", "parameter_rel_error_std",
         "state_rmse_mean", "state_rmse_std"],
    )

    rows_by_case = {row["case"]: row for row in summary_rows}
    case_labels, bar_colors, plotted_cases = [], [], []
    parameter_means, parameter_stds, state_means, state_stds = [], [], [], []

    for case_name, params in regimes:
        row = rows_by_case.get(case_name)
        if row is None:
            continue
        plotted_cases.append(case_name)
        case_labels.append(rf"$\beta$={params['beta']:.0f}, $n$={params['n']:.1f}")
        parameter_means.append(row["parameter_rel_error_mean"])
        parameter_stds.append(row["parameter_rel_error_std"])
        state_means.append(row["state_rmse_mean"])
        state_stds.append(row["state_rmse_std"])
        bar_colors.append(STABLE_COLOR if params["regime"] == "stable" else OSCILLATORY_COLOR)

    positions = list(range(len(case_labels)))
    show_eb = len(seeds) > 1

    regime_comparisons = [
        ("stable_beta5", "oscillatory_beta5"),
        ("stable_beta8", "oscillatory_beta8"),
    ]
    parameter_vals_by_case = metric_values_by_group(raw_rows, "case", "parameter_rel_error")
    state_vals_by_case = metric_values_by_group(raw_rows, "case", "state_rmse")
    parameter_significance = pairwise_significance(parameter_vals_by_case, regime_comparisons)
    state_significance = pairwise_significance(state_vals_by_case, regime_comparisons)
    case_to_position = {name: i for i, name in enumerate(plotted_cases)}

    # Build representative trajectories for insets
    inset_cases = [
        ("stable",      5.0, 1.5),
        ("oscillatory", 5.0, 3.0),
    ]
    inset_trajs = {}
    for regime_label, b, nn in inset_cases:
        t_ins, y_ins = simulate_repressilator(b, nn)
        inset_trajs[regime_label] = (t_ins, y_ins)

    plt.rcParams['axes.formatter.useoffset'] = False

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    rng = np.random.default_rng(42)
    jw = 0.13
    bar_width = 0.55

    def _barstrip(ax, means, stds, vals_by_case):
        ax.bar(
            positions, means,
            width=bar_width,
            yerr=stds if show_eb else None,
            capsize=4 if show_eb else 0,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        for i, case_name in enumerate(plotted_cases):
            sv = vals_by_case.get(case_name, [])
            j = rng.uniform(-jw, jw, len(sv))
            ax.scatter(
                np.array(positions[i]) + j, sv,
                color="black", s=22, alpha=0.7, zorder=5, linewidths=0,
            )

    def _xformat(ax):
        ax.set_xticks(positions, case_labels, rotation=20, ha="right")

    # Panel (0): combined parameter error
    _barstrip(axes[0], parameter_means, parameter_stds, parameter_vals_by_case)
    _xformat(axes[0])
    axes[0].set_ylabel("Param. Error")
    axes[0].set_title("Parameter Recovery")
    axes[0].text(-0.02, 1.04, 'A', transform=axes[0].transAxes,
                 fontsize=15, fontweight='bold', va='bottom', ha='left', color='black', clip_on=False)

    parameter_tops = [m + (s if show_eb else 0.0) for m, s in zip(parameter_means, parameter_stds)]
    parameter_top_by_case = {c: t for c, t in zip(plotted_cases, parameter_tops)}
    annotate_pairwise_comparisons(
        axes[0],
        x_positions=case_to_position,
        top_values=parameter_top_by_case,
        comparisons=regime_comparisons,
        significance=parameter_significance,
        use_adjusted_p_value=True,
    )

    # Panel (1): state RMSE
    _barstrip(axes[1], state_means, state_stds, state_vals_by_case)
    _xformat(axes[1])
    axes[1].set_ylabel("State RMSE")
    axes[1].set_title("Trajectory Reconstruction")
    axes[1].text(-0.02, 1.04, 'B', transform=axes[1].transAxes,
                 fontsize=15, fontweight='bold', va='bottom', ha='left', color='black', clip_on=False)

    state_tops = [m + (s if show_eb else 0.0) for m, s in zip(state_means, state_stds)]
    state_top_by_case = {c: t for c, t in zip(plotted_cases, state_tops)}
    annotate_pairwise_comparisons(
        axes[1],
        x_positions=case_to_position,
        top_values=state_top_by_case,
        comparisons=regime_comparisons,
        significance=state_significance,
        use_adjusted_p_value=True,
    )

    # Inset trajectory panels — small, positioned at upper right to avoid bar tops
    for ax_main, (regime_label, b, nn) in zip(axes, inset_cases):
        t_ins, y_ins = inset_trajs[regime_label]
        inset_ax = ax_main.inset_axes([0.03, 0.68, 0.26, 0.27])
        for k in range(3):
            inset_ax.plot(t_ins, y_ins[:, k], color=TRAJ_COLORS[k], linewidth=0.7)
        inset_ax.set_xlim(t_ins[0], t_ins[-1])
        inset_ax.set_title(
            rf"$\beta$={b:.0f}, $n$={nn:.1f}",
            fontsize=6, pad=5,
        )
        inset_ax.tick_params(labelbottom=False, labelleft=False, length=0)
        for spine in inset_ax.spines.values():
            spine.set_linewidth(0.5)

    legend_handles = [
        Patch(facecolor=STABLE_COLOR, alpha=0.85, edgecolor="none", label="Stable"),
        Patch(facecolor=OSCILLATORY_COLOR, alpha=0.85, edgecolor="none", label="Oscillatory"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=9, loc="upper right")

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
