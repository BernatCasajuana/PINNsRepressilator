"""
Experiment 2 — Partial observation.

Question: How much performance is lost when only a subset of repressors is measured,
and does the PINN's physics residual give it an advantage over classical ODE fitting
in this setting?

Design:
  - True parameters: β = 5.0, n = 3.0 (oscillatory regime)
  - Fixed noise level: 0.05 (relative to peak-to-peak amplitude)
  - Dense sampling (stride = 1, 1000 points); 5 seeds per design
  - Three observation designs compared:
      "3/3"  components [0, 1, 2]:  all repressors observed  — fully constrained system
      "2/3"  components [0, 1]:     two repressors observed  — x3 inferred from ODE
      "1/3"  component  [0]:        one repressor observed   — x2, x3 inferred from ODE
  - Initial guesses: β₀ = 4.0, n₀ = 2.5

For each (design, seed), both the PINN and an L-BFGS-B baseline are run on
identical data. The L-BFGS-B baseline integrates the full three-equation ODE
(scipy.optimize.minimize + scipy.integrate.odeint) but minimises MSE only on the
observed species. The PINN also receives only the observed species as an observation
loss but additionally enforces the ODE residual on all three equations at dense
collocation points throughout the domain.

Hypothesis: both methods use the full three-equation ODE to handle unobserved
species; however, they differ fundamentally. L-BFGS-B integrates the ODE exactly
— given β and n, x3 is always perfectly consistent with x1 and x2 through the ODE
coupling, so "unobserved" species are already implicitly constrained. The PINN
outputs x1, x2, x3 as independent neural network values at each time point and
enforces ODE coupling only approximately via the residual loss, giving x3 more
freedom to deviate. The real question is therefore not whether the PINN "knows"
more physics about unobserved species, but whether the neural-network
parameterisation provides a smoother optimisation landscape than direct parameter
search with exact ODE integration.

Physics regularisation pays off precisely when data is incomplete — but in this
setting the incompleteness is about which species are measured, not how densely.
If L-BFGS-B matches PINN even in the 1/3 setting, exact ODE coupling is
sufficient and the neural-network parameterisation adds no benefit here.

Output: LaTeX table (tables/exp2_partial_observation.tex)
  - Rows: 3/3 (baseline), 2/3, 1/3
  - Columns: PINN and L-BFGS-B side by side for parameter error and state RMSE;
             p-value (Mann–Whitney U) comparing PINN vs L-BFGS-B at each design.

Note: running exp3_sampling_density.py after this experiment produces a combined
tables/observability.tex merging both measurement observability dimensions
(observed species here; observation count in exp3) into one table.

Key finding expected: losing the phase-coupled third repressor (x3) inflates variance
sharply because the Hill coefficient n is identified largely through the phase relationship
between repressors.  A single observed repressor typically fails to identify n reliably.
The PINN's advantage over L-BFGS-B should be largest in the 1/3 setting.
"""

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
    make_synthetic_dataset,
    write_csv,
    write_run_manifest,
)
from data.generate_data import protein_repressilator_rhs
from scripts.pinns.inverse import run_inverse

true_beta = 5.0
true_n = 3.0
noise_level = 0.05
seeds = [0, 1, 2, 3, 4]
observation_designs = [
    ("3/3", [0, 1, 2]),
    ("2/3", [0, 1]),
    ("1/3", [0]),
]
train_iterations = 10000
results_dir = "results/exp2_partial_observation"
figure_path = "figures/exp2_partial_observation.png"  # kept for pilot compatibility
table_path = "tables/exp2_partial_observation.tex"

_X0 = [1.0, 1.0, 1.2]


def run_classical_fitting(dataset, observed_components, beta_guess=4.0, n_guess=2.5):
    """L-BFGS-B + odeint, MSE on observed species only."""
    t_obs = dataset["t"].flatten()
    y_obs = dataset["y"][:, observed_components]
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
        return float(np.mean((y_pred[:, observed_components] - y_obs) ** 2))

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


def _write_latex_table(path, summary_rows, raw_rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table_order = ["3/3", "2/3", "1/3"]
    rows_by_design = {r["design"]: r for r in summary_rows}

    header = (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\caption{Partial observation results. Parameter error and state RMSE "
        r"(mean $\pm$ SD, $n = 5$ seeds) for PINN and L-BFGS-B under three "
        r"observation designs. Both methods use the full three-equation ODE; "
        r"L-BFGS-B minimises MSE only on the observed species while the PINN "
        r"additionally enforces the ODE residual on all species at collocation points. "
        r"$p$-values (two-sided Mann--Whitney U) compare PINN vs L-BFGS-B "
        r"parameter error at each design.}" + "\n"
        r"\label{tab:exp2_partial_observation}" + "\n"
        r"\begin{tabular}{lcccccc}" + "\n"
        r"\toprule" + "\n"
        r" & \multicolumn{2}{c}{Parameter error} & \multicolumn{2}{c}{State RMSE} & \\" + "\n"
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + "\n"
        r"Repressors & PINN & L-BFGS-B & PINN & L-BFGS-B & $p$ (PINN vs L-BFGS-B) \\" + "\n"
        r"\midrule"
    )

    body_lines = []
    for design in table_order:
        row = rows_by_design.get(design)
        if row is None:
            continue
        pinn_param = _fmt(row["parameter_rel_error_mean"], row["parameter_rel_error_std"])
        pinn_rmse  = _fmt(row["state_rmse_mean"], row["state_rmse_std"])
        cls_param  = _fmt(row["classical_parameter_rel_error_mean"], row["classical_parameter_rel_error_std"])
        cls_rmse   = _fmt(row["classical_state_rmse_mean"], row["classical_state_rmse_std"])
        pinn_vals = [r["parameter_rel_error"] for r in raw_rows if r["design"] == design]
        cls_vals  = [r["classical_parameter_rel_error"] for r in raw_rows if r["design"] == design]
        p = None
        if len(pinn_vals) >= 2 and len(cls_vals) >= 2:
            _, p = mannwhitneyu(pinn_vals, cls_vals, alternative="two-sided")
        label = f"{design} (baseline)" if design == "3/3" else design
        body_lines.append(
            f"{label} & {pinn_param} & {cls_param} & {pinn_rmse} & {cls_rmse} & {_fmt_p(p)} \\\\"
        )

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


def main():
    ensure_project_directories()
    os.makedirs("tables", exist_ok=True)
    expected_runs = len(observation_designs) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp2_partial_observation",
            "script_path": __file__,
            "results_dir": results_dir,
            "table_path": table_path,
            "classical_method": "L-BFGS-B (scipy.optimize.minimize + scipy.integrate.odeint, MSE on observed species only)",
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
            cls = run_classical_fitting(dataset, observed_components)
            raw_rows.append(
                {
                    "design": design_label,
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

    design_order = [label for label, _ in observation_designs]
    summary_rows = aggregate_metrics(
        raw_rows,
        group_keys=["design"],
        metric_keys=[
            "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse",
            "classical_parameter_rel_error", "classical_state_rmse",
        ],
    )
    summary_rows.sort(key=lambda row: design_order.index(row["design"]))

    write_csv(
        os.path.join(results_dir, "exp2_partial_observation_raw.csv"),
        raw_rows,
        [
            "design", "seed",
            "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse",
            "classical_parameter_rel_error", "classical_state_rmse",
            "outdir",
        ],
    )
    write_csv(
        os.path.join(results_dir, "exp2_partial_observation_summary.csv"),
        summary_rows,
        [
            "design", "num_runs",
            "beta_rel_error_mean", "beta_rel_error_std",
            "n_rel_error_mean", "n_rel_error_std",
            "parameter_rel_error_mean", "parameter_rel_error_std",
            "state_rmse_mean", "state_rmse_std",
            "classical_parameter_rel_error_mean", "classical_parameter_rel_error_std",
            "classical_state_rmse_mean", "classical_state_rmse_std",
        ],
    )

    _write_latex_table(table_path, summary_rows, raw_rows)


if __name__ == "__main__":
    main()
