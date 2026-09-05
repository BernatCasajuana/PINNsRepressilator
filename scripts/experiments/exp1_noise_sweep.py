"""
Experiment 1 — Noise sensitivity and baseline comparison.

Question: How does inverse-PINN parameter recovery degrade as observation noise increases,
and does it outperform classical least-squares ODE fitting?

Design:
  - True parameters: β = 5.0, n = 3.0 (canonical oscillatory regime)
  - All three repressors observed (x1, x2, x3), dense sampling (stride = 1, 1000 points)
  - Noise levels: 0.0, 0.01, 0.05, 0.10, 0.20 — expressed as a fraction of the
    mean peak-to-peak signal amplitude so values are comparable across regimes
  - 5 seeds per noise level (independent data realisations)
  - 10000 Adam iterations per run; initial guesses β₀ = 4.0, n₀ = 2.5
  - For each (noise, seed), a classical ODE fit is also run as a baseline:
    scipy.optimize.minimize (L-BFGS-B) + scipy.integrate.odeint, same initial guess

Figure: 4-panel 2×2 layout (12×9), line plots with mean ± SD
  - Panel A (0,0): β relative error vs noise (PINN) + JT trend annotation
  - Panel B (0,1): n relative error vs noise (PINN) + JT trend annotation
  - Panel C (1,0): combined parameter error — PINN vs L-BFGS-B
  - Panel D (1,1): state RMSE — PINN vs L-BFGS-B

Significance (panels A/B): Jonckheere-Terpstra (JT) test for monotonic trend across
ordered noise levels (0 → 0.20). JT is preferred over pairwise Mann–Whitney here
because the noise levels form an ordered sequence and the scientific question is
whether degradation is monotonic, not which specific pairs differ. A single two-sided
p-value is shown as a text annotation in each panel. Panels C/D compare PINN vs
L-BFGS-B at each noise level (uncorrected two-sided Mann–Whitney U); significance is
not marked on the figure and is reported in the text/results instead.

Key finding expected: parameter recovery is relatively stable across this noise range;
trajectory RMSE grows faster because the observation loss directly penalises trajectory
fit. The ODE residual regularises trajectories even when observations are corrupted.
PINN is expected to outperform classical fitting at higher noise levels due to physics
regularisation.
"""

import os
import sys
import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.utils import (
    aggregate_metrics,
    ensure_project_directories,
    finalize_figure,
    format_p_value_label,
    jonckheere_terpstra_p_value,
    make_synthetic_dataset,
    mann_whitney_p_value,
    metric_values_by_group,
    p_value_to_stars,
    write_csv,
    write_run_manifest,
)
from datasets.data import protein_repressilator_rhs
from scripts.pinns.inverse import run_inverse

true_beta = 5.0
true_n = 3.0
noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]
seeds = [0, 1, 2, 3, 4]
observed_components = [0, 1, 2]
train_iterations = 10000
results_dir = "results/exp1_noise_sweep"
figure_path = "figures/exp1_noise_sweep.png"

PINN_COLOR = "#0072B2"
CLASSICAL_COLOR = "#E69F00"
_X0 = [1.0, 1.0, 1.2]


def run_classical_fitting(dataset, beta_guess=4.0, n_guess=2.5):
    """Fit β and n by minimising observed-data MSE via numerical ODE integration."""
    t_obs = dataset["t"].flatten()
    y_obs = dataset["y"]
    true_b = float(dataset["beta"])
    true_n_ = float(dataset["n"])

    def _obj(log_params):
        b = np.exp(log_params[0])
        nn = np.exp(log_params[1])
        try:
            y_pred = scipy.integrate.odeint(
                protein_repressilator_rhs, _X0, t_obs,
                args=(b, nn), rtol=1e-7, atol=1e-9,
            )
        except Exception:
            return 1e10
        return float(np.mean((y_pred - y_obs) ** 2))

    res = scipy.optimize.minimize(
        _obj,
        [np.log(beta_guess), np.log(n_guess)],
        method="L-BFGS-B",
        bounds=[(-2.0, 5.0), (-1.0, 3.0)],
        options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-9},
    )
    beta_est = float(np.exp(res.x[0]))
    n_est = float(np.exp(res.x[1]))

    beta_rel_err = abs(beta_est - true_b) / true_b
    n_rel_err = abs(n_est - true_n_) / true_n_

    y_final = scipy.integrate.odeint(
        protein_repressilator_rhs, _X0, t_obs, args=(beta_est, n_est),
    )
    state_rmse = float(np.sqrt(np.mean((y_final - dataset["y_clean"]) ** 2)))

    return {
        "beta_rel_error": beta_rel_err,
        "n_rel_error": n_rel_err,
        "parameter_rel_error": 0.5 * (beta_rel_err + n_rel_err),
        "state_rmse": state_rmse,
    }


def main():
    ensure_project_directories()
    expected_runs = len(noise_levels) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp1_noise_sweep",
            "script_path": __file__,
            "results_dir": results_dir,
            "figure_path": figure_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "noise_levels": list(noise_levels),
            "observed_components": list(observed_components),
            "true_beta": true_beta,
            "true_n": true_n,
            "classical_method": "L-BFGS-B (scipy.optimize.minimize + scipy.integrate.odeint)",
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )
    raw_rows = []

    for noise_level in noise_levels:
        for seed in seeds:
            dataset = make_synthetic_dataset(true_beta, true_n, noise_level=noise_level, seed=seed)
            result = run_inverse(
                dataset_path=dataset,
                outdir_base=os.path.join(results_dir, "runs", f"noise{noise_level}"),
                beta_guess=4.0,
                n_guess=2.5,
                observation_stride=1,
                observed_components=observed_components,
                train_iterations=train_iterations,
                random_seed=seed,
                save_checkpoint=True,
            )
            cls = run_classical_fitting(dataset)
            raw_rows.append(
                {
                    "noise_level": noise_level,
                    "seed": seed,
                    "beta_rel_error": result["beta_rel_error"],
                    "n_rel_error": result["n_rel_error"],
                    "parameter_rel_error": result["parameter_rel_error"],
                    "state_rmse": result["state_rmse"],
                    "classical_parameter_rel_error": cls["parameter_rel_error"],
                    "classical_state_rmse": cls["state_rmse"],
                    "outdir": result["outdir"],
                }
            )

    build_outputs(raw_rows, results_dir, figure_path)


def build_outputs(raw_rows, results_dir, figure_path):
    """Aggregate raw per-run rows into summary/significance CSVs and the figure.

    Split out from main() so the figure can be regenerated (e.g. after a
    styling tweak) from an existing exp1_noise_sweep_raw.csv without rerunning
    the full PINN/classical-fitting sweep.
    """
    summary_rows = aggregate_metrics(
        raw_rows,
        group_keys=["noise_level"],
        metric_keys=[
            "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse",
            "classical_parameter_rel_error", "classical_state_rmse",
        ],
    )
    summary_rows.sort(key=lambda row: row["noise_level"])

    write_csv(
        os.path.join(results_dir, "exp1_noise_sweep_raw.csv"),
        raw_rows,
        [
            "noise_level", "seed",
            "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse",
            "classical_parameter_rel_error", "classical_state_rmse",
            "outdir",
        ],
    )
    write_csv(
        os.path.join(results_dir, "exp1_noise_sweep_summary.csv"),
        summary_rows,
        [
            "noise_level", "num_runs",
            "beta_rel_error_mean", "beta_rel_error_std",
            "n_rel_error_mean", "n_rel_error_std",
            "parameter_rel_error_mean", "parameter_rel_error_std",
            "state_rmse_mean", "state_rmse_std",
            "classical_parameter_rel_error_mean", "classical_parameter_rel_error_std",
            "classical_state_rmse_mean", "classical_state_rmse_std",
        ],
    )

    noise_values = [row["noise_level"] for row in summary_rows]
    positions = list(range(len(noise_values)))
    noise_labels = [f"{v:.2f}" for v in noise_values]

    beta_means = [row["beta_rel_error_mean"] for row in summary_rows]
    beta_stds = [row["beta_rel_error_std"] for row in summary_rows]
    n_means = [row["n_rel_error_mean"] for row in summary_rows]
    n_stds = [row["n_rel_error_std"] for row in summary_rows]
    parameter_means = [row["parameter_rel_error_mean"] for row in summary_rows]
    parameter_stds = [row["parameter_rel_error_std"] for row in summary_rows]
    state_means = [row["state_rmse_mean"] for row in summary_rows]
    state_stds = [row["state_rmse_std"] for row in summary_rows]
    cls_param_means = [row["classical_parameter_rel_error_mean"] for row in summary_rows]
    cls_param_stds = [row["classical_parameter_rel_error_std"] for row in summary_rows]
    cls_state_means = [row["classical_state_rmse_mean"] for row in summary_rows]
    cls_state_stds = [row["classical_state_rmse_std"] for row in summary_rows]
    show_eb = len(seeds) > 1

    beta_vals_by_noise = metric_values_by_group(raw_rows, "noise_level", "beta_rel_error")
    n_vals_by_noise = metric_values_by_group(raw_rows, "noise_level", "n_rel_error")
    parameter_vals_by_noise = metric_values_by_group(raw_rows, "noise_level", "parameter_rel_error")
    state_vals_by_noise = metric_values_by_group(raw_rows, "noise_level", "state_rmse")
    cls_parameter_vals_by_noise = metric_values_by_group(raw_rows, "noise_level", "classical_parameter_rel_error")
    cls_state_vals_by_noise = metric_values_by_group(raw_rows, "noise_level", "classical_state_rmse")

    # JT trend test for panels A/B: do β/n errors monotonically increase with noise?
    jt_p_beta = jonckheere_terpstra_p_value([beta_vals_by_noise[v] for v in noise_values])
    jt_p_n    = jonckheere_terpstra_p_value([n_vals_by_noise[v]    for v in noise_values])

    # PINN vs L-BFGS-B at each noise level (panels C/D).
    parameter_pinn_vs_classical_p = {
        v: mann_whitney_p_value(parameter_vals_by_noise[v], cls_parameter_vals_by_noise[v])
        for v in noise_values
    }
    state_pinn_vs_classical_p = {
        v: mann_whitney_p_value(state_vals_by_noise[v], cls_state_vals_by_noise[v])
        for v in noise_values
    }

    significance_rows = [
        {
            "comparison": "beta_jt_trend",
            "noise_level": "all",
            "p_value": jt_p_beta,
            "stars": p_value_to_stars(jt_p_beta),
        },
        {
            "comparison": "n_jt_trend",
            "noise_level": "all",
            "p_value": jt_p_n,
            "stars": p_value_to_stars(jt_p_n),
        },
    ]
    for v in noise_values:
        for comparison_name, p_value in (
            ("parameter_pinn_vs_classical", parameter_pinn_vs_classical_p[v]),
            ("state_pinn_vs_classical", state_pinn_vs_classical_p[v]),
        ):
            significance_rows.append(
                {
                    "comparison": comparison_name,
                    "noise_level": v,
                    "p_value": p_value,
                    "stars": p_value_to_stars(p_value),
                }
            )
    write_csv(
        os.path.join(results_dir, "exp1_noise_sweep_significance.csv"),
        significance_rows,
        ["comparison", "noise_level", "p_value", "stars"],
    )

    plt.rcParams['axes.formatter.useoffset'] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)

    def _label(ax, letter):
        ax.text(-0.02, 1.04, letter, transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='bottom', ha='left', color='black', clip_on=False)

    def _ribbon(ax, color, label, means, stds, marker="o", linestyle="-"):
        m = np.array(means)
        s = np.array(stds)
        ax.plot(positions, m, linestyle=linestyle, marker=marker, color=color,
                linewidth=1.2, markersize=4, zorder=4, label=label)
        if show_eb:
            ax.fill_between(positions, m - s, m + s, color=color, alpha=0.15, zorder=3)

    LABEL_FONTSIZE = 14
    TITLE_FONTSIZE = 16
    TITLE_FONTSIZE_LONG = 13

    def _xformat(ax):
        ax.set_xticks(positions, noise_labels)
        ax.set_xlabel(r"Relative $\sigma$", fontsize=LABEL_FONTSIZE)

    def _jt_annotation(ax, p_value):
        label = f"JT: {format_p_value_label(p_value)} {p_value_to_stars(p_value)}".strip()
        ax.text(0.97, 0.97, label, transform=ax.transAxes, ha="right", va="top", fontsize=9.5)

    # Panel A: β recovery (PINN only) + JT trend annotation
    _ribbon(axes[0, 0], PINN_COLOR, None, beta_means, beta_stds, marker="o")
    _jt_annotation(axes[0, 0], jt_p_beta)
    _xformat(axes[0, 0])
    axes[0, 0].set_ylabel("Relative Error", fontsize=LABEL_FONTSIZE)
    axes[0, 0].set_title(r"$\beta$ Recovery", fontsize=TITLE_FONTSIZE)
    _label(axes[0, 0], 'A')

    # Panel B: n recovery (PINN only) + JT trend annotation
    _ribbon(axes[0, 1], PINN_COLOR, None, n_means, n_stds)
    _jt_annotation(axes[0, 1], jt_p_n)
    _xformat(axes[0, 1])
    axes[0, 1].set_ylabel("Relative Error", fontsize=LABEL_FONTSIZE)
    axes[0, 1].set_title(r"$n$ Recovery", fontsize=TITLE_FONTSIZE)
    _label(axes[0, 1], 'B')

    # Panel C: combined parameter error — PINN vs L-BFGS-B
    _ribbon(axes[1, 0], PINN_COLOR, "PINN", parameter_means, parameter_stds)
    _ribbon(axes[1, 0], CLASSICAL_COLOR, "L-BFGS-B",
            cls_param_means, cls_param_stds, linestyle="--", marker="s")
    _xformat(axes[1, 0])
    axes[1, 0].set_ylabel("Error", fontsize=LABEL_FONTSIZE)
    axes[1, 0].set_title("Parameter Recovery — PINN vs L-BFGS-B", fontsize=TITLE_FONTSIZE_LONG)
    axes[1, 0].legend(fontsize=8, handlelength=3)
    _label(axes[1, 0], 'C')

    # Panel D: state RMSE — PINN vs L-BFGS-B
    _ribbon(axes[1, 1], PINN_COLOR, "PINN", state_means, state_stds)
    _ribbon(axes[1, 1], CLASSICAL_COLOR, "L-BFGS-B",
            cls_state_means, cls_state_stds, linestyle="--", marker="s")
    _xformat(axes[1, 1])
    axes[1, 1].set_ylabel("State RMSE", fontsize=LABEL_FONTSIZE)
    axes[1, 1].set_title("Trajectory Reconstruction — PINN vs L-BFGS-B", fontsize=TITLE_FONTSIZE_LONG)
    axes[1, 1].legend(fontsize=8, handlelength=3)
    _label(axes[1, 1], 'D')

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
