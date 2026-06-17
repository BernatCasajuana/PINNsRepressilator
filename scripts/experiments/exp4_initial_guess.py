"""
Experiment 4 — Initial parameter guesses.

Question: How sensitive is inverse-PINN training to the initial guesses for β and n?

Design:
  - True parameters: β = 5.0, n = 3.0
  - Fixed noise level: 0.05; all three repressors observed; stride = 1 (dense)
  - 7×7 grid of initial guesses:
      β₀ ∈ {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}  (true β = 5.0 included)
      n₀ ∈ {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5}  (true n  = 3.0 included)
  - 1 seed per grid cell (49 total runs); 10000 iterations each
  - The grid covers broad offsets to reveal attraction basins and failure modes

Figure: single heatmap showing combined parameter error over the (β₀, n₀) grid
  - Combined error: 0.5 × (|Δβ|/β + |Δn|/n) per grid cell
  - Arrow annotation marking the true parameter location (β=5.0, n=3.0)
  - Dashed contour at the 10% combined error threshold

Key finding expected: the loss landscape is non-convex — some (β₀, n₀) combinations
trap the optimiser far from the true parameters. The contour reveals the reliable-recovery
basin, which may be surprisingly narrow or irregular.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.experiment_utils import (
    aggregate_metrics,
    ensure_project_directories,
    finalize_figure,
    make_synthetic_dataset,
    write_csv,
    write_run_manifest,
)
from scripts.pinns.inverse import run_inverse

true_beta = 5.0
true_n = 3.0
noise_level = 0.05
beta_guesses = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
n_guesses = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
seeds = [0]
train_iterations = 10000
results_dir = "results/exp4_initial_guess"
figure_path = "figures/exp4_initial_guess.png"

CONTOUR_THRESHOLD = 0.10  # 10% relative error contour


RAW_CSV_FIELDNAMES = ["beta_guess", "n_guess", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"]


def _load_completed_runs(csv_path):
    """Return a set of (bg, ng, seed) tuples already present in the raw CSV."""
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    import csv as _csv
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            completed.add((float(row["beta_guess"]), float(row["n_guess"]), int(row["seed"])))
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
    expected_runs = len(beta_guesses) * len(n_guesses) * len(seeds)
    write_run_manifest(
        os.path.join(results_dir, "run_manifest.json"),
        {
            "experiment_name": "exp4_initial_guess",
            "script_path": __file__,
            "results_dir": results_dir,
            "figure_path": figure_path,
            "train_iterations": train_iterations,
            "seeds": list(seeds),
            "beta_guesses": list(beta_guesses),
            "n_guesses": list(n_guesses),
            "true_beta": true_beta,
            "true_n": true_n,
            "noise_level": noise_level,
            "expected_runs": expected_runs,
            "expected_total_train_iterations": expected_runs * train_iterations,
        },
    )

    raw_csv_path = os.path.join(results_dir, "exp4_initial_guess_raw.csv")
    completed = _load_completed_runs(raw_csv_path)
    raw_rows = []

    # Reload already-completed rows so the summary/figure is correct on resume
    if completed:
        import csv as _csv
        with open(raw_csv_path, newline="") as f:
            for row in _csv.DictReader(f):
                raw_rows.append({
                    "beta_guess": float(row["beta_guess"]),
                    "n_guess": float(row["n_guess"]),
                    "seed": int(row["seed"]),
                    "beta_rel_error": float(row["beta_rel_error"]),
                    "n_rel_error": float(row["n_rel_error"]),
                    "parameter_rel_error": float(row["parameter_rel_error"]),
                    "state_rmse": float(row["state_rmse"]),
                    "outdir": row["outdir"],
                })

    for bg in beta_guesses:
        for ng in n_guesses:
            for seed in seeds:
                if (bg, ng, seed) in completed:
                    print(f"Skipping already-completed run: bg={bg}, ng={ng}, seed={seed}")
                    continue
                dataset = make_synthetic_dataset(true_beta, true_n, noise_level=noise_level, seed=seed)
                result = run_inverse(
                    dataset_path=dataset,
                    outdir_base=os.path.join(results_dir, "runs", f"bg{bg}_ng{ng}"),
                    beta_guess=bg,
                    n_guess=ng,
                    observation_stride=1,
                    observed_components=[0, 1, 2],
                    train_iterations=train_iterations,
                    random_seed=seed,
                    save_checkpoint=True,
                )
                row = {
                    "beta_guess": bg,
                    "n_guess": ng,
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
        group_keys=["beta_guess", "n_guess"],
        metric_keys=["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key=lambda row: (row["beta_guess"], row["n_guess"]))

    write_csv(
        os.path.join(results_dir, "exp4_initial_guess_summary.csv"),
        summary_rows,
        [
            "beta_guess", "n_guess", "num_runs",
            "beta_rel_error_mean", "beta_rel_error_std",
            "n_rel_error_mean", "n_rel_error_std",
            "parameter_rel_error_mean", "parameter_rel_error_std",
            "state_rmse_mean", "state_rmse_std",
        ],
    )

    combined_heatmap = np.full((len(n_guesses), len(beta_guesses)), np.nan)
    for row in summary_rows:
        bi = beta_guesses.index(row["beta_guess"])
        ni = n_guesses.index(row["n_guess"])
        combined_heatmap[ni, bi] = row["parameter_rel_error_mean"]

    # Star position in grid coordinates
    star_bx = beta_guesses.index(true_beta) if true_beta in beta_guesses else None
    star_nx = n_guesses.index(true_n) if true_n in n_guesses else None

    plt.rcParams['axes.formatter.useoffset'] = False

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.2))

    vmax = float(np.nanmax(combined_heatmap))

    im = ax.imshow(
        combined_heatmap, origin="lower", aspect="auto",
        cmap="viridis", vmin=0.0, vmax=vmax,
        interpolation="bicubic",
    )
    ax.set_xticks(range(len(beta_guesses)), [str(v) for v in beta_guesses])
    ax.set_yticks(range(len(n_guesses)), [str(v) for v in n_guesses])
    ax.set_xlabel(r"Initial $\beta_0$", fontsize=11)
    ax.set_ylabel(r"Initial $n_0$", fontsize=11)
    ax.set_title("Sensitivity to Initial Parameter Guess")

    if star_bx is not None and star_nx is not None:
        ax.annotate(
            "Truth",
            xy=(star_bx, star_nx),
            xytext=(star_bx + 0.6, star_nx + 0.6),
            fontsize=plt.rcParams["font.size"],
            color="black",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2, shrinkA=10),
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Combined Error")

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
