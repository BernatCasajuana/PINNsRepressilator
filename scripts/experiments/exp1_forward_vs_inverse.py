"""
Experiment 6 — Forward vs inverse PINN gap.

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

Figure: 3-panel layout (1×3)
  - Left:   state RMSE for forward vs inverse (grouped bars by noise level)
  - Centre: β relative error for inverse PINN only
  - Right:  n relative error for inverse PINN only

The left panel shows the identifiability cost: extra RMSE incurred by the need to
simultaneously identify parameters.  The centre/right panels contextualise whether
better trajectory fit translates to better parameter estimates.

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
noise_levels = [0.01, 0.05, 0.10]
seeds = [0, 1, 2]
train_iterations = 5000
results_dir = "results/exp6_forward_vs_inverse"
figure_path = "figures/exp6_forward_vs_inverse.png"

FORWARD_COLOR = "#009E73"
INVERSE_COLOR = "#0072B2"
BETA_COLOR = "#0072B2"
N_COLOR = "#E69F00"


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
            "experiment_name": "exp6_forward_vs_inverse",
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
        os.path.join(results_dir, "exp6_forward_vs_inverse_raw.csv"),
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
    beta_means, beta_stds = _group_mean_std("beta_rel_error")
    n_means, n_stds = _group_mean_std("n_rel_error")

    show_eb = len(seeds) > 1
    n_groups = len(noise_values)
    x = np.arange(n_groups)
    bar_w = 0.35
    rng = np.random.default_rng(42)
    jw = 0.07

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        rf"Exp 6 — Forward vs inverse PINN gap ($\beta$={true_beta}, $n$={true_n}, {len(seeds)} seeds)",
        fontsize=13,
    )

    # Panel (0): RMSE comparison — grouped bars
    axes[0].bar(
        x - bar_w / 2, fwd_rmse_means,
        width=bar_w,
        yerr=fwd_rmse_stds if show_eb else None,
        capsize=4 if show_eb else 0,
        color=FORWARD_COLOR, edgecolor="white", linewidth=0.5,
        label="Forward (known params)", zorder=3,
    )
    axes[0].bar(
        x + bar_w / 2, inv_rmse_means,
        width=bar_w,
        yerr=inv_rmse_stds if show_eb else None,
        capsize=4 if show_eb else 0,
        color=INVERSE_COLOR, edgecolor="white", linewidth=0.5,
        label="Inverse (estimated params)", zorder=3,
    )
    for i, nv in enumerate(noise_values):
        fwd_sv = [row["forward_state_rmse"] for row in raw_rows if row["noise_level"] == nv]
        inv_sv = [row["inverse_state_rmse"] for row in raw_rows if row["noise_level"] == nv]
        j = rng.uniform(-jw, jw, len(fwd_sv))
        axes[0].scatter(i - bar_w / 2 + j, fwd_sv, color="black", s=18, alpha=0.7, zorder=5, linewidths=0)
        axes[0].scatter(i + bar_w / 2 + j, inv_sv, color="black", s=18, alpha=0.7, zorder=5, linewidths=0)
    axes[0].set_xticks(x, noise_labels)
    axes[0].set_xlabel("Relative noise level")
    axes[0].set_ylabel("State RMSE  (vs clean trajectory)")
    axes[0].set_title("RMSE: forward vs inverse")
    axes[0].legend(fontsize=8)

    # Panel (1): β recovery (inverse only)
    axes[1].errorbar(
        x, beta_means,
        yerr=beta_stds if show_eb else None,
        fmt="o", linestyle="-", linewidth=1.5,
        capsize=4 if show_eb else 0,
        color=BETA_COLOR, ecolor=BETA_COLOR,
        markerfacecolor=BETA_COLOR, markeredgecolor=BETA_COLOR,
        zorder=4,
    )
    for i, nv in enumerate(noise_values):
        sv = [row["beta_rel_error"] for row in raw_rows if row["noise_level"] == nv]
        j = rng.uniform(-jw * 2, jw * 2, len(sv))
        axes[1].scatter(i + j, sv, alpha=0.5, s=20, color=BETA_COLOR, zorder=5, linewidths=0)
    axes[1].set_xticks(x, noise_labels)
    axes[1].set_xlabel("Relative noise level")
    axes[1].set_ylabel(r"$\hat{\beta}$ relative error")
    axes[1].set_title(r"$\beta$ recovery (inverse)")

    # Panel (2): n recovery (inverse only)
    axes[2].errorbar(
        x, n_means,
        yerr=n_stds if show_eb else None,
        fmt="o", linestyle="-", linewidth=1.5,
        capsize=4 if show_eb else 0,
        color=N_COLOR, ecolor=N_COLOR,
        markerfacecolor=N_COLOR, markeredgecolor=N_COLOR,
        zorder=4,
    )
    for i, nv in enumerate(noise_values):
        sv = [row["n_rel_error"] for row in raw_rows if row["noise_level"] == nv]
        j = rng.uniform(-jw * 2, jw * 2, len(sv))
        axes[2].scatter(i + j, sv, alpha=0.5, s=20, color=N_COLOR, zorder=5, linewidths=0)
    axes[2].set_xticks(x, noise_labels)
    axes[2].set_xlabel("Relative noise level")
    axes[2].set_ylabel(r"$\hat{n}$ relative error")
    axes[2].set_title(r"$n$ recovery (inverse)")

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
