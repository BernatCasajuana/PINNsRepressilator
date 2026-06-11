"""
Experiment 1 — Forward vs inverse PINN gap.

Question: How much does knowing the true parameters help? What is the performance gap
between a forward PINN (β and n known, trajectory fitting only) and an inverse PINN
(β and n estimated jointly with the trajectory)?

Design:
  - True parameters: β = 5.0, n = 3.0 (oscillatory regime)
  - Three noise levels: 0.01, 0.05, 0.10
  - 3 seeds per condition; 5000 Adam iterations per run; dense observations (stride = 1)
  - For each (noise, seed) pair, two runs are compared:
      forward:  run_forward with true β, n — trajectory fit only (no parameter estimation)
      inverse:  run_inverse estimating β, n from scratch (β₀=4.0, n₀=2.5)
  - The forward run uses the noisy dataset (same as inverse) and sets run_lbfgs=False for
    an apples-to-apples comparison with the Adam-only inverse run

Figure: single-panel (7×5)
  - State RMSE for forward vs inverse PINNs across noise levels (line plot, mean ± SD)

The panel shows the identifiability cost: extra RMSE incurred by the need to
simultaneously identify parameters.

Key finding expected: forward PINNs achieve lower RMSE than inverse PINNs at the same
noise level — the parameter identification task adds extra optimisation difficulty.
However, the gap should shrink as noise increases (the ODE residual dominates).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.experiment_utils import (
    ensure_project_directories,
    finalize_figure,
    make_synthetic_dataset,
    write_csv,
    write_run_manifest,
)
from scripts.pinns.forward import run_forward
from scripts.pinns.inverse import run_inverse

true_beta = 5.0
true_n = 3.0
noise_levels = [0.0, 0.01, 0.05, 0.10]
seeds = [0, 1, 2]
train_iterations = 5000
results_dir = "results/exp1_forward_vs_inverse"
figure_path = "figures/exp1_forward_vs_inverse.png"

FORWARD_COLOR = "#0072B2"
INVERSE_COLOR = "#E69F00"


def _save_synthetic_dataset_as_npz(dataset: dict, path: str) -> None:
    """Write an in-memory dataset dict to a temporary .npz file for run_forward."""
    np.savez(
        path,
        t=dataset["t"],
        y=dataset["y"],
        y_clean=dataset["y_clean"],
        beta=dataset["beta"],
        n=dataset["n"],
        noise=dataset["noise"],
    )


def main():
    ensure_project_directories()
    os.makedirs(os.path.join(results_dir, "tmp"), exist_ok=True)
    expected_runs = len(noise_levels) * len(seeds) * 2  # forward + inverse
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp1_forward_vs_inverse",
            "script_path": __file__,
            "results_dir": results_dir,
            "figure_path": figure_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "noise_levels": list(noise_levels),
            "true_beta": true_beta,
            "true_n": true_n,
            "note": (
                "Forward run uses known beta/n; inverse run estimates them. "
                "Both use Adam only (run_lbfgs=False), stride=1, 5000 iterations."
            ),
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )

    raw_rows = []

    for noise_level in noise_levels:
        for seed in seeds:
            dataset = make_synthetic_dataset(true_beta, true_n, noise_level=noise_level, seed=seed)

            # Persist dataset to a temporary .npz so run_forward can load it
            tmp_path = os.path.join(
                results_dir, "tmp",
                f"tmp_beta{true_beta}_n{true_n}_noise{noise_level}_seed{seed}.npz",
            )
            _save_synthetic_dataset_as_npz(dataset, tmp_path)

            # Forward run (known parameters, trajectory fit only)
            fwd = run_forward(
                tmp_path,
                outdir_base=os.path.join(results_dir, "runs", "forward"),
                observation_stride=1,
                observed_components=[0, 1, 2],
                adam_epochs=train_iterations,
                run_lbfgs=False,
            )

            # Inverse run (estimate parameters)
            inv = run_inverse(
                dataset_path=dataset,
                outdir_base=os.path.join(results_dir, "runs", "inverse"),
                beta_guess=4.0,
                n_guess=2.5,
                observation_stride=1,
                observed_components=[0, 1, 2],
                train_iterations=train_iterations,
                random_seed=seed,
                save_checkpoint=False,
            )

            raw_rows.append(
                {
                    "noise_level": noise_level,
                    "seed": seed,
                    "forward_state_rmse": fwd["state_rmse"],
                    "inverse_state_rmse": inv["state_rmse"],
                    "beta_rel_error": inv["beta_rel_error"],
                    "n_rel_error": inv["n_rel_error"],
                    "parameter_rel_error": inv["parameter_rel_error"],
                    "forward_outdir": fwd["outdir"],
                    "inverse_outdir": inv["outdir"],
                }
            )

    write_csv(
        os.path.join(results_dir, "exp1_forward_vs_inverse_raw.csv"),
        raw_rows,
        [
            "noise_level", "seed",
            "forward_state_rmse", "inverse_state_rmse",
            "beta_rel_error", "n_rel_error", "parameter_rel_error",
            "forward_outdir", "inverse_outdir",
        ],
    )

    # Aggregate by noise level
    noise_values = sorted(set(row["noise_level"] for row in raw_rows))
    noise_labels = [f"{v:.2f}" for v in noise_values]

    def _group_mean_std(key):
        means, stds = [], []
        for nv in noise_values:
            vals = [row[key] for row in raw_rows if row["noise_level"] == nv]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        return means, stds

    fwd_rmse_means, fwd_rmse_stds = _group_mean_std("forward_state_rmse")
    inv_rmse_means, inv_rmse_stds = _group_mean_std("inverse_state_rmse")

    show_eb = len(seeds) > 1
    x = np.arange(len(noise_values))

    plt.rcParams['axes.formatter.useoffset'] = False

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(x, fwd_rmse_means, yerr=fwd_rmse_stds if show_eb else None,
                fmt="o", linestyle="-", linewidth=1.0,
                capsize=4 if show_eb else 0,
                color=FORWARD_COLOR, ecolor=FORWARD_COLOR,
                markerfacecolor=FORWARD_COLOR, markeredgecolor=FORWARD_COLOR,
                zorder=4, label="Forward (Known Params)")
    ax.errorbar(x, inv_rmse_means, yerr=inv_rmse_stds if show_eb else None,
                fmt="s", linestyle="-", linewidth=1.0,
                capsize=4 if show_eb else 0,
                color=INVERSE_COLOR, ecolor=INVERSE_COLOR,
                markerfacecolor=INVERSE_COLOR, markeredgecolor=INVERSE_COLOR,
                zorder=4, label="Inverse (Estimated Params)")
    ax.set_xticks(x, noise_labels)
    ax.set_xlabel("Relative Noise Level")
    ax.set_ylabel("State RMSE")
    ax.legend(fontsize=8)

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
