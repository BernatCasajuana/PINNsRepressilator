"""
Experiment 3 — Sampling density.

Question: How sparse can the observations be before parameter recovery degrades,
and does the PINN's physics-guided interpolation give it an advantage over classical
ODE fitting when data is sparse?

Design:
  - True parameters: β = 5.0, n = 3.0 (oscillatory regime)
  - Fixed noise level: 0.05 (relative to peak-to-peak amplitude)
  - All three repressors observed (x1, x2, x3)
  - Observation counts: 10, 25, 50, 100 — evenly spaced across the 1000-point grid
    (as a percentage of the full grid: 1%, 2.5%, 5%, 10%)
  - 5 seeds per count; 10000 Adam iterations per run
  - Initial guesses: β₀ = 4.0, n₀ = 2.5

For each (count, seed), both the PINN and an L-BFGS-B baseline are run on identical data.
The L-BFGS-B baseline integrates the full ODE trajectory over the dense grid
(scipy.optimize.minimize + scipy.integrate.odeint) but computes MSE only at the
sparse observed time points. The PINN enforces the ODE residual at dense collocation
points across the full domain regardless of observation density.

Hypothesis: physics regularisation pays off precisely when data is incomplete — and
in this setting the incompleteness is temporal. L-BFGS-B integrates the full
trajectory (for numerical stability) but receives gradient signal only at the sparse
observed time points; the optimiser must infer the global shape of the oscillation
from a handful of samples. The PINN's ODE residual is evaluated at dense collocation
points across the full domain regardless of observation density, providing structural
information between observations that L-BFGS-B lacks. This is a genuine asymmetry,
unlike the partial-observation case where L-BFGS-B's exact integration already handles
unobserved species through the ODE coupling. If L-BFGS-B matches PINN performance,
exact integration at the sparse subset is sufficient and the collocation-based residual
adds no benefit.

Output: LaTeX table (tables/exp3_sampling_density.tex)
  - Rows: 100 (baseline), 50, 25, 10 time points
  - Columns: PINN and L-BFGS-B side by side for parameter error and state RMSE;
             p-value (Mann–Whitney U) comparing PINN vs L-BFGS-B at each count.

Combined output: tables/observability.tex
  - Two-section table covering both measurement observability dimensions:
      Section 1 (from exp2): observed species (3/3, 2/3, 1/3 repressors)
      Section 2 (this exp):  observation count (100, 50, 25, 10 time points)
  - Both methods shown (PINN and L-BFGS-B) side by side.
  - Generated when exp2 raw results are available; skipped otherwise.

Key finding expected: sparse sampling (<25 points) removes complete oscillatory phases,
costing critical phase information for identifying n. The PINN's advantage should grow
with sparsity because the collocation-based residual fills in structural information
between observations that L-BFGS-B lacks.
"""

import csv
import os
import sys
import numpy as np
import scipy.integrate
import scipy.optimize
from scipy.stats import mannwhitneyu

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.experiment_utils import (
    aggregate_metrics,
    ensure_project_directories,
    evenly_spaced_observation_indices,
    make_synthetic_dataset,
    metric_values_by_group,
    pairwise_significance,
    write_csv,
    write_run_manifest,
)
from data.generate_data import protein_repressilator_rhs
from scripts.pinns.inverse import run_inverse

true_beta = 5.0
true_n = 3.0
noise_level = 0.05
observation_counts = [10, 25, 50, 100]
total_grid_points = 1000
seeds = [0, 1, 2, 3, 4]
train_iterations = 10000
results_dir = "results/exp3_sampling_density"
figure_path = "figures/exp3_sampling_density.png"  # kept for pilot compatibility
table_path = "tables/exp3_sampling_density.tex"
observability_table_path = "tables/observability.tex"

_X0 = [1.0, 1.0, 1.2]


def run_classical_fitting(dataset, observation_indices, beta_guess=4.0, n_guess=2.5):
    """L-BFGS-B + odeint, MSE computed at sparse observation indices only."""
    t_full = dataset["t"].flatten()
    y_sparse = dataset["y"][observation_indices]  # all 3 species at sparse times
    true_b = float(dataset["beta"])
    true_n_ = float(dataset["n"])

    def _obj(log_params):
        b = np.exp(log_params[0])
        nn = np.exp(log_params[1])
        try:
            y_pred = scipy.integrate.odeint(
                protein_repressilator_rhs, _X0, t_full,
                args=(b, nn), rtol=1e-7, atol=1e-9,
            )
        except Exception:
            return 1e10
        return float(np.mean((y_pred[observation_indices] - y_sparse) ** 2))

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
        protein_repressilator_rhs, _X0, t_full, args=(beta_est, n_est),
    )
    state_rmse = float(np.sqrt(np.mean((y_final - dataset["y_clean"]) ** 2)))
    return {
        "parameter_rel_error": 0.5 * (beta_rel_err + n_rel_err),
        "state_rmse": state_rmse,
    }


def _fmt(mean, std):
    return f"${mean:.3f} \\pm {std:.3f}$"


def _fmt_p(p):
    if p is None:
        return "--"
    if p < 0.001:
        return r"$p < 0.001$ ***"
    stars = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    return f"$p = {p:.3f}$ {stars}".strip()


def _pinn_vs_cls_p(raw_rows, group_key, group_val):
    pinn = [r["parameter_rel_error"] for r in raw_rows if r[group_key] == group_val]
    cls  = [r["classical_parameter_rel_error"] for r in raw_rows if r[group_key] == group_val]
    if len(pinn) < 2:
        return None
    _, p = mannwhitneyu(pinn, cls, alternative="two-sided")
    return p


def _write_sampling_density_table(path, summary_rows, raw_rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows_by_count = {int(r["observation_count"]): r for r in summary_rows}
    table_order = [100, 50, 25, 10]

    header = (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\caption{Sampling density results. Parameter error and state RMSE "
        r"(mean $\pm$ SD, $n = 5$ seeds) for PINN and L-BFGS-B across four observation "
        r"counts (evenly spaced across 1000 time points, all 3 repressors observed). "
        r"L-BFGS-B integrates the full ODE trajectory but computes MSE only at the "
        r"observed time points; the PINN additionally enforces the ODE residual at dense "
        r"collocation points across the full domain. "
        r"$p$-values (two-sided Mann--Whitney U) compare PINN vs L-BFGS-B "
        r"parameter error at each count.}" + "\n"
        r"\label{tab:exp3_sampling_density}" + "\n"
        r"\begin{tabular}{lcccccc}" + "\n"
        r"\toprule" + "\n"
        r" & \multicolumn{2}{c}{Parameter error} & \multicolumn{2}{c}{State RMSE} & \\" + "\n"
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + "\n"
        r"Time points & PINN & L-BFGS-B & PINN & L-BFGS-B & $p$ (PINN vs L-BFGS-B) \\" + "\n"
        r"\midrule"
    )

    body_lines = []
    for count in table_order:
        row = rows_by_count.get(count)
        if row is None:
            continue
        pinn_param = _fmt(row["parameter_rel_error_mean"], row["parameter_rel_error_std"])
        pinn_rmse  = _fmt(row["state_rmse_mean"], row["state_rmse_std"])
        cls_param  = _fmt(row["classical_parameter_rel_error_mean"], row["classical_parameter_rel_error_std"])
        cls_rmse   = _fmt(row["classical_state_rmse_mean"], row["classical_state_rmse_std"])
        p = _pinn_vs_cls_p(raw_rows, "observation_count", count)
        label = "100 (baseline)" if count == 100 else str(count)
        body_lines.append(
            f"{label} & {pinn_param} & {cls_param} & {pinn_rmse} & {cls_rmse} & {_fmt_p(p)} \\\\"
        )

    footer = (
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for line in body_lines:
            f.write(line + "\n")
        f.write(footer + "\n")

    print(f"LaTeX table written to {path}")


def _load_exp2_raw(results_base="results"):
    csv_path = os.path.join(
        results_base, "exp2_partial_observation", "exp2_partial_observation_raw.csv"
    )
    if not os.path.exists(csv_path):
        return None
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "design": row["design"],
                "seed": int(row["seed"]),
                "parameter_rel_error": float(row["parameter_rel_error"]),
                "state_rmse": float(row["state_rmse"]),
                "classical_parameter_rel_error": float(row.get("classical_parameter_rel_error", "nan")),
                "classical_state_rmse": float(row.get("classical_state_rmse", "nan")),
            })
    return rows


def _write_combined_observability_table(
    path, partial_obs_raw, density_summary_rows, density_raw_rows,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # --- partial observation block ---
    partial_design_order = ["3/3", "2/3", "1/3"]
    partial_summary = aggregate_metrics(
        partial_obs_raw, group_keys=["design"],
        metric_keys=[
            "parameter_rel_error", "state_rmse",
            "classical_parameter_rel_error", "classical_state_rmse",
        ],
    )
    partial_by_design = {r["design"]: r for r in partial_summary}

    # --- sampling density block ---
    density_by_count = {int(r["observation_count"]): r for r in density_summary_rows}
    density_order = [100, 50, 25, 10]

    has_cls_partial = not np.isnan(
        partial_obs_raw[0].get("classical_parameter_rel_error", float("nan"))
    ) if partial_obs_raw else False

    header = (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\caption{Measurement observability. Parameter error and state RMSE "
        r"(mean $\pm$ SD, $n = 5$ seeds) for PINN and L-BFGS-B across two "
        r"observability dimensions: number of observed repressor species (top) and "
        r"number of observed time points (bottom). "
        r"Both methods use the full three-equation ODE. "
        r"L-BFGS-B minimises MSE only on the available measurements; "
        r"the PINN additionally enforces the ODE residual at dense collocation points. "
        r"$p$-values (two-sided Mann--Whitney U) compare PINN vs L-BFGS-B "
        r"parameter error at each condition.}" + "\n"
        r"\label{tab:observability}" + "\n"
        r"\begin{tabular}{lcccccc}" + "\n"
        r"\toprule" + "\n"
        r" & \multicolumn{2}{c}{Parameter error} & \multicolumn{2}{c}{State RMSE} & \\" + "\n"
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + "\n"
        r"Condition & PINN & L-BFGS-B & PINN & L-BFGS-B & $p$ (PINN vs L-BFGS-B) \\" + "\n"
        r"\midrule"
    )

    body_lines = []

    # Section 1: observed species
    body_lines.append(
        r"\multicolumn{6}{l}{\textit{Observed repressors (1000 time points)}} \\"
    )
    for design in partial_design_order:
        row = partial_by_design.get(design)
        if row is None:
            continue
        pinn_param = _fmt(row["parameter_rel_error_mean"], row["parameter_rel_error_std"])
        pinn_rmse  = _fmt(row["state_rmse_mean"], row["state_rmse_std"])
        if has_cls_partial:
            cls_param = _fmt(row["classical_parameter_rel_error_mean"], row["classical_parameter_rel_error_std"])
            cls_rmse  = _fmt(row["classical_state_rmse_mean"], row["classical_state_rmse_std"])
            p = _pinn_vs_cls_p(partial_obs_raw, "design", design)
            p_str = _fmt_p(p)
        else:
            cls_param = cls_rmse = p_str = "--"
        label = r"\quad 3/3 (baseline)" if design == "3/3" else rf"\quad {design}"
        body_lines.append(
            f"{label} & {pinn_param} & {cls_param} & {pinn_rmse} & {cls_rmse} & {p_str} \\\\"
        )

    body_lines.append(r"\midrule")

    # Section 2: observation count
    body_lines.append(
        r"\multicolumn{6}{l}{\textit{Observed time points (all 3 repressors)}} \\"
    )
    for count in density_order:
        row = density_by_count.get(count)
        if row is None:
            continue
        pinn_param = _fmt(row["parameter_rel_error_mean"], row["parameter_rel_error_std"])
        pinn_rmse  = _fmt(row["state_rmse_mean"], row["state_rmse_std"])
        cls_param  = _fmt(row["classical_parameter_rel_error_mean"], row["classical_parameter_rel_error_std"])
        cls_rmse   = _fmt(row["classical_state_rmse_mean"], row["classical_state_rmse_std"])
        p = _pinn_vs_cls_p(density_raw_rows, "observation_count", count)
        label = r"\quad 100 (baseline)" if count == 100 else rf"\quad {count}"
        body_lines.append(
            f"{label} & {pinn_param} & {cls_param} & {pinn_rmse} & {cls_rmse} & {_fmt_p(p)} \\\\"
        )

    footer = (
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for line in body_lines:
            f.write(line + "\n")
        f.write(footer + "\n")

    print(f"Combined observability table written to {path}")


def main():
    ensure_project_directories()
    os.makedirs("tables", exist_ok=True)
    expected_runs = len(observation_counts) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp3_sampling_density",
            "script_path": __file__,
            "results_dir": results_dir,
            "table_path": table_path,
            "observability_table_path": observability_table_path,
            "classical_method": "L-BFGS-B (scipy.optimize.minimize + scipy.integrate.odeint, MSE at sparse indices over full trajectory)",
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
            cls = run_classical_fitting(dataset, observation_indices)
            raw_rows.append(
                {
                    "observation_count": obs_count,
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

    summary_rows = aggregate_metrics(
        raw_rows,
        group_keys=["observation_count"],
        metric_keys=[
            "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse",
            "classical_parameter_rel_error", "classical_state_rmse",
        ],
    )
    summary_rows.sort(key=lambda row: row["observation_count"], reverse=True)

    write_csv(
        os.path.join(results_dir, "exp3_sampling_density_raw.csv"),
        raw_rows,
        [
            "observation_count", "seed",
            "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse",
            "classical_parameter_rel_error", "classical_state_rmse",
            "outdir",
        ],
    )
    write_csv(
        os.path.join(results_dir, "exp3_sampling_density_summary.csv"),
        summary_rows,
        [
            "observation_count", "num_runs",
            "beta_rel_error_mean", "beta_rel_error_std",
            "n_rel_error_mean", "n_rel_error_std",
            "parameter_rel_error_mean", "parameter_rel_error_std",
            "state_rmse_mean", "state_rmse_std",
            "classical_parameter_rel_error_mean", "classical_parameter_rel_error_std",
            "classical_state_rmse_mean", "classical_state_rmse_std",
        ],
    )

    _write_sampling_density_table(table_path, summary_rows, raw_rows)

    results_base = os.path.dirname(os.path.abspath(results_dir))
    partial_obs_raw = _load_exp2_raw(results_base=results_base)
    if partial_obs_raw is not None:
        _write_combined_observability_table(
            observability_table_path, partial_obs_raw, summary_rows, raw_rows,
        )
    else:
        print("Skipping combined observability table: exp2 raw results not found. Run exp2 first.")


if __name__ == "__main__":
    main()
