"""
Experiment 6 — Physics loss weight sensitivity.

Question: How sensitive is inverse-PINN performance to the physics loss weight λ_f?

The total loss is: L = λ_f · L_eq  +  λ_0 · L_IC  +  λ_y · L_obs
This experiment sweeps λ_f while holding λ_0 = λ_y = 1.0 fixed to isolate the
contribution of the ODE residual penalty.

Design:
  - True parameters: β = 5.0, n = 3.0 (oscillatory regime)
  - Fixed noise level: 0.05; all three repressors observed; stride = 1 (dense)
  - λ_f sweep: {0.01, 0.1, 1.0, 10.0, 100.0} — 5 decades
  - λ_0 = λ_y = 1.0 (fixed)
  - 5 seeds per λ_f; 10000 Adam iterations per run
  - Initial guesses: β₀ = 4.0, n₀ = 2.5

  loss_weights format for run_inverse (all 3 observed components):
    [λ_f, λ_f, λ_f,   ← 3 ODE residual terms
     λ_0, λ_0, λ_0,   ← 3 IC terms
     λ_y, λ_y, λ_y]   ← 3 observation terms

Output: LaTeX table (tables/exp6_loss_weights.tex)
  - Rows: λ_f ∈ {0.01, 0.1, 1.0 (baseline), 10.0, 100.0}
  - Columns: parameter error (mean ± SD), state RMSE (mean ± SD),
             p-value (Mann–Whitney U) vs λ_f = 1.0

Key finding expected: very low λ_f (<0.1) lets the network ignore the ODE constraint,
degrading parameter recovery while fitting observations well. Very high λ_f (>10) can
suppress the observation loss and prevent convergence. λ_f ≈ 1 is expected to be
near-optimal, confirming that balanced weighting between physics and data terms is
sufficient and the method is not highly sensitive to λ_f within the middle range.
"""

import os
import sys

from scipy.stats import mannwhitneyu

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.experiment_utils import (
    aggregate_metrics,
    ensure_project_directories,
    make_synthetic_dataset,
    write_csv,
    write_run_manifest,
)
from scripts.pinns.inverse import run_inverse

true_beta = 5.0
true_n = 3.0
noise_level = 0.05
lambda_f_values = [0.01, 0.1, 1.0, 10.0, 100.0]
lambda_0 = 1.0
lambda_y = 1.0
seeds = [0, 1, 2, 3, 4]
train_iterations = 10000
results_dir = "results/exp6_loss_weights"
table_path = "tables/exp6_loss_weights.tex"


def _build_loss_weights(lf: float) -> list:
    """[eq1, eq2, eq3, ic1, ic2, ic3, obs1, obs2, obs3] for all-3-observed case."""
    return [lf, lf, lf, lambda_0, lambda_0, lambda_0, lambda_y, lambda_y, lambda_y]


def _fmt(mean, std):
    return f"${mean:.3f} \\pm {std:.3f}$"


def _fmt_p(p):
    if p is None:
        return "--"
    if p < 0.001:
        return r"$p < 0.001$ ***"
    stars = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    return f"$p = {p:.3f}$ {stars}".strip()


def _write_latex_table(path, summary_rows, raw_rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    baseline_lf = 1.0
    rows_by_lf = {row["lambda_f"]: row for row in summary_rows}

    header = (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\caption{Physics loss weight sensitivity. $\beta$ error, $n$ error, and state RMSE "
        r"(mean $\pm$ SD, $n = 5$ seeds) for PINN across five values of the physics "
        r"loss weight $\lambda_f$ ($\lambda_0 = \lambda_y = 1$ fixed). "
        r"$p$-values (two-sided Mann--Whitney U) compare each $\lambda_f$ to the "
        r"baseline $\lambda_f = 1$.}" + "\n"
        r"\label{tab:exp6_loss_weights}" + "\n"
        r"\begin{tabular}{lccccc}" + "\n"
        r"\toprule" + "\n"
        r"$\lambda_f$ & $\beta$ error & $n$ error & State RMSE"
        r" & $p$ ($\beta$ vs $\lambda_f = 1$) & $p$ ($n$ vs $\lambda_f = 1$) \\" + "\n"
        r"\midrule"
    )

    body_lines = []
    for lf in lambda_f_values:
        row = rows_by_lf.get(lf)
        if row is None:
            continue
        beta_str = _fmt(row["beta_rel_error_mean"], row["beta_rel_error_std"])
        n_str    = _fmt(row["n_rel_error_mean"],    row["n_rel_error_std"])
        rmse_str = _fmt(row["state_rmse_mean"],     row["state_rmse_std"])
        if lf == baseline_lf:
            p_beta_str = p_n_str = "--"
        else:
            beta_lf   = [r["beta_rel_error"] for r in raw_rows if r["lambda_f"] == lf]
            beta_base = [r["beta_rel_error"] for r in raw_rows if r["lambda_f"] == baseline_lf]
            n_lf      = [r["n_rel_error"]    for r in raw_rows if r["lambda_f"] == lf]
            n_base    = [r["n_rel_error"]    for r in raw_rows if r["lambda_f"] == baseline_lf]
            p_beta = None
            p_n    = None
            if len(beta_lf) >= 2:
                _, p_beta = mannwhitneyu(beta_lf, beta_base, alternative="two-sided")
                _, p_n    = mannwhitneyu(n_lf,    n_base,    alternative="two-sided")
            p_beta_str = _fmt_p(p_beta)
            p_n_str    = _fmt_p(p_n)
        label = r"$1.0$ (baseline)" if lf == baseline_lf else f"${lf}$"
        body_lines.append(f"{label} & {beta_str} & {n_str} & {rmse_str} & {p_beta_str} & {p_n_str} \\\\")

    footer = (
        r"\bottomrule" + "\n"
        r"\multicolumn{6}{l}{\footnotesize *** $p < 0.001$, ** $p < 0.01$, * $p < 0.05$.} \\" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for line in body_lines:
            f.write(line + "\n")
        f.write(footer + "\n")

    print(f"LaTeX table written to {path}")


RAW_CSV_FIELDNAMES = ["lambda_f", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"]


def _load_completed_runs(csv_path):
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    import csv as _csv
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            completed.add((float(row["lambda_f"]), int(row["seed"])))
    return completed


def _append_raw_row(csv_path, row):
    import csv as _csv
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=RAW_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    ensure_project_directories()
    os.makedirs("tables", exist_ok=True)
    expected_runs = len(lambda_f_values) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp6_loss_weights",
            "script_path": __file__,
            "results_dir": results_dir,
            "table_path": table_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "lambda_f_values": list(lambda_f_values),
            "lambda_0": lambda_0,
            "lambda_y": lambda_y,
            "true_beta": true_beta,
            "true_n": true_n,
            "noise_level": noise_level,
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )

    raw_csv_path = os.path.join(results_dir, "exp6_loss_weights_raw.csv")
    completed = _load_completed_runs(raw_csv_path)
    raw_rows = []

    if completed:
        import csv as _csv
        with open(raw_csv_path, newline="") as f:
            for r in _csv.DictReader(f):
                raw_rows.append({
                    "lambda_f": float(r["lambda_f"]),
                    "seed": int(r["seed"]),
                    "beta_rel_error": float(r["beta_rel_error"]),
                    "n_rel_error": float(r["n_rel_error"]),
                    "parameter_rel_error": float(r["parameter_rel_error"]),
                    "state_rmse": float(r["state_rmse"]),
                    "outdir": r["outdir"],
                })

    for lf in lambda_f_values:
        loss_weights = _build_loss_weights(lf)
        for seed in seeds:
            if (lf, seed) in completed:
                print(f"Skipping already-completed run: lf={lf}, seed={seed}")
                continue
            dataset = make_synthetic_dataset(true_beta, true_n, noise_level=noise_level, seed=seed)
            result = run_inverse(
                dataset_path=dataset,
                outdir_base=os.path.join(results_dir, "runs", f"lf{lf}"),
                beta_guess=4.0,
                n_guess=2.5,
                observation_stride=1,
                observed_components=[0, 1, 2],
                loss_weights=loss_weights,
                train_iterations=train_iterations,
                random_seed=seed,
                save_checkpoint=False,
            )
            row = {
                "lambda_f": lf,
                "seed": seed,
                "beta_rel_error": result["beta_rel_error"],
                "n_rel_error": result["n_rel_error"],
                "parameter_rel_error": result["parameter_rel_error"],
                "state_rmse": result["state_rmse"],
                "outdir": result["outdir"],
            }
            raw_rows.append(row)
            _append_raw_row(raw_csv_path, row)

    summary_rows = aggregate_metrics(
        raw_rows,
        group_keys=["lambda_f"],
        metric_keys=["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key=lambda row: row["lambda_f"])

    write_csv(
        os.path.join(results_dir, "exp6_loss_weights_summary.csv"),
        summary_rows,
        [
            "lambda_f", "num_runs",
            "beta_rel_error_mean", "beta_rel_error_std",
            "n_rel_error_mean", "n_rel_error_std",
            "parameter_rel_error_mean", "parameter_rel_error_std",
            "state_rmse_mean", "state_rmse_std",
        ],
    )

    _write_latex_table(table_path, summary_rows, raw_rows)


if __name__ == "__main__":
    main()
