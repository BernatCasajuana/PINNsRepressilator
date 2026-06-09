"""
Experiment 4: inverse sensitivity to initial parameter guesses. 
Question: how sensitive is inverse-PINN training to the initial guesses for $\beta$ and $n$?
Design: the inverse problem is run over a 4x4 grid of initial guesses, spanning the true pair and nearby offsets: $\beta_0 \in \{4.0, 5.0, 6.0, 7.0\}$ and $n_0 \in \{2.0, 2.5, 3.0, 3.5\}$.
Output: heatmaps of the relative recovery error on $\beta$ and $n$ over the initial-guess grid.
"""

# Import necessary libraries, utilities, and set up paths
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.experiment_utils import aggregate_metrics, ensure_project_directories, finalize_figure, make_synthetic_dataset, write_csv
from scripts.pinns.inverse import run_inverse

# Experiment parameters and output paths
true_beta = 5.0
true_n = 3.0
noise_level = 0.05
beta_guesses = [4.0, 5.0, 6.0, 7.0]
n_guesses = [2.0, 2.5, 3.0, 3.5]
seeds = [0, 1]
train_iterations = 10000
results_dir = "results/exp_initial_guess"
figure_path = "figures/exp_initial_guess.png"
heatmap_cmap = mcolors.LinearSegmentedColormap.from_list(
    "paper_blues_exact_tones",
    ["#A6C8E0", "#5DA5DA", "#1F77B4"],
)

# Main experiment loop
def main():
    ensure_project_directories()
    raw_rows = []

    for beta_guess in beta_guesses:
        for n_guess in n_guesses:
            for seed in seeds:
                dataset = make_synthetic_dataset(true_beta, true_n, noise_level = noise_level, seed = seed)
                result = run_inverse(
                    dataset_path = dataset,
                    outdir_base = os.path.join(results_dir, "runs"),
                    C1_guess = beta_guess,
                    C2_guess = n_guess,
                    observation_stride = 1,
                    observed_components = [0, 1, 2],
                    train_iterations = train_iterations,
                    random_seed = seed,
                    save_checkpoint = True,
                )
                raw_rows.append(
                    {
                        "beta_guess": beta_guess,
                        "n_guess": n_guess,
                        "seed": seed,
                        "beta_rel_error": result["beta_rel_error"],
                        "n_rel_error": result["n_rel_error"],
                        "parameter_rel_error": result["parameter_rel_error"],
                        "state_rmse": result["state_rmse"],
                        "outdir": result["outdir"],
                    }
                )

    summary_rows = aggregate_metrics(
        raw_rows,
        group_keys = ["beta_guess", "n_guess"],
        metric_keys = ["beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse"],
    )
    summary_rows.sort(key = lambda row: (row["beta_guess"], row["n_guess"]))

    write_csv(
        os.path.join(results_dir, "initial_guess_raw.csv"),
        raw_rows,
        ["beta_guess", "n_guess", "seed", "beta_rel_error", "n_rel_error", "parameter_rel_error", "state_rmse", "outdir"],
    )
    write_csv(
        os.path.join(results_dir, "initial_guess_summary.csv"),
        summary_rows,
        [
            "beta_guess",
            "n_guess",
            "num_runs",
            "beta_rel_error_mean",
            "beta_rel_error_std",
            "n_rel_error_mean",
            "n_rel_error_std",
            "parameter_rel_error_mean",
            "parameter_rel_error_std",
            "state_rmse_mean",
            "state_rmse_std",
        ],
    )

    beta_heatmap = np.zeros((len(n_guesses), len(beta_guesses)))
    n_heatmap = np.zeros((len(n_guesses), len(beta_guesses)))
    for row in summary_rows:
        beta_index = beta_guesses.index(row["beta_guess"])
        n_index = n_guesses.index(row["n_guess"])
        beta_heatmap[n_index, beta_index] = row["beta_rel_error_mean"]
        n_heatmap[n_index, beta_index] = row["n_rel_error_mean"]

    vmin = min(float(np.min(beta_heatmap)), float(np.min(n_heatmap)))
    vmax = max(float(np.max(beta_heatmap)), float(np.max(n_heatmap)))
    norm = mcolors.Normalize(vmin = vmin, vmax = vmax)

    fig = plt.figure(figsize = (12, 5))
    grid = fig.add_gridspec(1, 3, width_ratios = [1.0, 0.06, 1.0], wspace = 0.30)
    left_ax = fig.add_subplot(grid[0, 0])
    cbar_ax = fig.add_subplot(grid[0, 1])
    right_ax = fig.add_subplot(grid[0, 2])

    beta_image = left_ax.imshow(beta_heatmap, origin = "lower", aspect = "auto", cmap = heatmap_cmap, norm = norm)
    left_ax.set_xticks(range(len(beta_guesses)), [str(value) for value in beta_guesses])
    left_ax.set_yticks(range(len(n_guesses)), [str(value) for value in n_guesses])
    left_ax.set_xlabel(r"Initial $\beta$")
    left_ax.set_ylabel(r"Initial $n$")
    left_ax.set_title(r"Relative Error on $\beta$")

    n_image = right_ax.imshow(n_heatmap, origin = "lower", aspect = "auto", cmap = heatmap_cmap, norm = norm)
    right_ax.set_xticks(range(len(beta_guesses)), [str(value) for value in beta_guesses])
    right_ax.set_yticks(range(len(n_guesses)), [str(value) for value in n_guesses])
    right_ax.set_xlabel(r"Initial $\beta$")
    right_ax.set_ylabel(r"Initial $n$")
    right_ax.set_title(r"Relative Error on $n$")
    shared_colorbar = fig.colorbar(n_image, cax = cbar_ax)
    shared_colorbar.set_label("Relative Error")

    finalize_figure(figure_path)


if __name__ == "__main__":
    main()
