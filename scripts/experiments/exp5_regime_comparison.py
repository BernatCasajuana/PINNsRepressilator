"""
Experiment 5 — Dynamical regime comparison.

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

Output: LaTeX table (tables/exp5_regime_comparison.tex)
  - Rows: grouped by β, stable then oscillatory; β embedded in condition label
  - Columns: condition, β error (mean ± SD), n error (mean ± SD), state RMSE (mean ± SD),
             p(β) and p(n) — Holm–Bonferroni-adjusted Mann–Whitney U, stable vs oscillatory

Key finding expected: the stable regime produces lower RMSE because trajectories are
smoother; however, oscillatory dynamics provide richer information for n recovery —
the oscillatory regime may show lower *parameter* error despite higher *trajectory* error.
"""

import os
import sys
import numpy as np

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.utils import (
    aggregate_metrics,
    ensure_project_directories,
    make_synthetic_dataset,
    metric_values_by_group,
    pairwise_significance,
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
results_dir = "results/exp5_regime_comparison"
figure_path = "figures/exp5_regime_comparison.png"  # kept for pilot compatibility
table_path = "tables/exp5_regime_comparison.tex"


def _fmt_mean_std(mean, std):
    return f"${mean:.3f} \\pm {std:.3f}$"


def _stars(p):
    if p is None or not np.isfinite(p) or p >= 0.05:
        return ""
    return "***" if p < 0.001 else ("**" if p < 0.01 else "*")


def _sig_stars(sig_entry):
    if sig_entry is None:
        return ""
    return _stars(sig_entry.get("adjusted_p_value", float("nan")))


def _fmt_with_stars(mean, std, stars_str):
    base = f"${mean:.3f} \\pm {std:.3f}$"
    return base + (f"\\rlap{{\\textsuperscript{{{stars_str}}}}}" if stars_str else "")


def _write_latex_table(path, rows_by_case, beta_significance, n_significance, rmse_significance):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    table_cases = [
        ("stable_beta5",      "Stable",      5.0, 1.5),
        ("oscillatory_beta5", "Oscillatory", 5.0, 3.0),
        ("stable_beta8",      "Stable",      8.0, 1.5),
        ("oscillatory_beta8", "Oscillatory", 8.0, 3.0),
    ]
    comparisons_map = {
        "oscillatory_beta5": ("stable_beta5", "oscillatory_beta5"),
        "oscillatory_beta8": ("stable_beta8", "oscillatory_beta8"),
    }

    header = (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\caption{Regime comparison results. $\beta$ error, $n$ error, and state RMSE "
        r"(mean $\pm$ SD, $n = 5$ seeds) for stable and oscillatory dynamics at two "
        r"production rates ($\beta \in \{5, 8\}$, $n_{\text{stable}}=1.5$, "
        r"$n_{\text{osc}}=3.0$).}" + "\n"
        r"\label{tab:exp5_regime_comparison}" + "\n"
        r"{\small\setlength{\tabcolsep}{5pt}%" + "\n"
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccc}" + "\n"
        r"\toprule" + "\n"
        r"Condition & $\beta$ error & $n$ error & State RMSE \\" + "\n"
        r"\midrule"
    )

    body_lines = []
    for case_name, regime_label, beta_val, n_val in table_cases:
        row = rows_by_case.get(case_name)
        if row is None:
            continue
        if case_name in comparisons_map:
            comp = comparisons_map[case_name]
            sig_b = beta_significance.get(comp) or beta_significance.get(comp[::-1])
            sig_n = n_significance.get(comp)    or n_significance.get(comp[::-1])
            sig_r = rmse_significance.get(comp) or rmse_significance.get(comp[::-1])
            sb = _sig_stars(sig_b)
            sn = _sig_stars(sig_n)
            sr = _sig_stars(sig_r)
        else:
            sb = sn = sr = ""
        beta_cell = _fmt_with_stars(row["beta_rel_error_mean"], row["beta_rel_error_std"], sb)
        n_cell    = _fmt_with_stars(row["n_rel_error_mean"],    row["n_rel_error_std"],    sn)
        rmse_cell = _fmt_with_stars(row["state_rmse_mean"],     row["state_rmse_std"],     sr)
        condition = f"{regime_label} ($\\beta={beta_val:.0f}$)"
        body_lines.append(
            f"{condition} & {beta_cell} & {n_cell} & {rmse_cell} \\\\"
        )
        if case_name == "oscillatory_beta5":
            body_lines.append(r"\midrule")

    footer = (
        r"\bottomrule" + "\n"
        r"\multicolumn{4}{l}{\footnotesize Superscripts on oscillatory rows: "
        r"Holm--Bonferroni MWU, stable vs.\ oscillatory at same $\beta$.} \\" + "\n"
        r"\multicolumn{4}{l}{\footnotesize MWU: Mann--Whitney U; "
        r"*** $p < 0.001$, ** $p < 0.01$, * $p < 0.05$.} \\" + "\n"
        r"\end{tabular*}}" + "\n"
        r"\end{table}"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for line in body_lines:
            f.write(line + "\n")
        f.write(footer + "\n")

    print(f"LaTeX table written to {path}")


def main():
    ensure_project_directories()
    os.makedirs("tables", exist_ok=True)
    expected_runs = len(regimes) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp5_regime_comparison",
            "script_path": __file__,
            "results_dir": results_dir,
            "table_path": table_path,
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
                outdir_base=os.path.join(results_dir, "runs", case_name),
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
        os.path.join(results_dir, "exp5_regime_comparison_raw.csv"),
        raw_rows,
        ["case", "regime", "beta", "n", "noise_level", "seed",
         "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(results_dir, "exp5_regime_comparison_summary.csv"),
        summary_rows,
        ["case", "regime", "beta", "n", "noise_level", "num_runs",
         "beta_rel_error_mean", "beta_rel_error_std",
         "n_rel_error_mean", "n_rel_error_std",
         "parameter_rel_error_mean", "parameter_rel_error_std",
         "state_rmse_mean", "state_rmse_std"],
    )

    rows_by_case = {row["case"]: row for row in summary_rows}
    regime_comparisons = [
        ("stable_beta5", "oscillatory_beta5"),
        ("stable_beta8", "oscillatory_beta8"),
    ]
    beta_vals_by_case = metric_values_by_group(raw_rows, "case", "beta_rel_error")
    n_vals_by_case    = metric_values_by_group(raw_rows, "case", "n_rel_error")
    rmse_vals_by_case = metric_values_by_group(raw_rows, "case", "state_rmse")
    beta_significance = pairwise_significance(beta_vals_by_case, regime_comparisons)
    n_significance    = pairwise_significance(n_vals_by_case,    regime_comparisons)
    rmse_significance = pairwise_significance(rmse_vals_by_case, regime_comparisons)

    _write_latex_table(table_path, rows_by_case, beta_significance, n_significance, rmse_significance)


if __name__ == "__main__":
    main()
