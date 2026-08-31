"""
Experiment 4 — Initial parameter guesses.

Question: what is the price of starting the inverse-PINN optimiser away from the true
(β, n) — i.e. what does the error landscape around the truth actually look like?

Design:
  - True parameters: β = 5.0, n = 3.0 (canonical oscillatory case)
  - Fixed noise level: 0.05; all three repressors observed; stride = 1 (dense)
  - 9×9 UNIFORM grid of initial guesses, truth at the exact centre (index 4/8):
      β₀ = 1, 2, 3, 4, 5, 6, 7, 8, 9      (true β = 5.0 at index 4)
      n₀ = 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5  (true n  = 3.0 at index 4)
    No far-out/unphysical outliers this time (the original design put β₀=15 and
    n₀=10 far outside the sampled region, which produced huge, unmeasured facets in
    the surface plot); every point here is a real measurement, and the axes are
    plotted tight to this exact range -- no padding into unsampled territory.
  - train_iterations DELIBERATELY reduced from 10000 to 2000. At 10000 iterations the
    optimiser converges to within noise from nearly every guess in the old grid,
    leaving no gradient to show. At 2000 iterations, guesses far from the truth have
    not fully converged, so the landscape actually reflects the cost of a bad guess.
  - 1 seed per grid cell (81 total runs, 2000 iterations each)
  - The true parameters β = 5.0, n = 3.0 are those of the synthetic dataset used for training

Figure: 3D error-landscape surface over the (β₀, n₀) plane — z = log10(combined
relative parameter error), with a marker + drop-line at the true parameters.
  - Combined error: 0.5 × (|Δβ|/β + |Δn|/n) per grid cell
  - Because the grid is uniform and dense, plot_surface's bilinear shading between
    adjacent nodes is an honest reading of the surface (unlike the old sparse grid,
    where the same shading would have fabricated a cliff across huge unmeasured gaps).
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.utils import (
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
beta_guesses = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]  # true=5.0 at index 4 (center)
n_guesses = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]     # true=3.0 at index 4 (center)
seeds = [0]
train_iterations = 2000
results_dir = "results/exp4_initial_guess"
figure_path = "figures/exp4_initial_guess.png"

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


def _check_no_stale_results(manifest_path, raw_csv_path):
    """Refuse to resume into runs left over from a different grid/iteration design.

    The per-run resume logic below only keys on (beta_guess, n_guess, seed): it has
    no way to tell a run apart from an earlier design that used a different
    train_iterations. Several points in this grid overlap with previous versions of
    this experiment, so silently resuming would mix results trained for different
    iteration counts into one "landscape" without any indication in the data.
    """
    if not (os.path.exists(manifest_path) and os.path.exists(raw_csv_path)):
        return
    with open(manifest_path) as f:
        prior_manifest = json.load(f)
    prior_iterations = prior_manifest.get("train_iterations")
    if prior_iterations is not None and prior_iterations != train_iterations:
        raise RuntimeError(
            f"'{raw_csv_path}' holds runs from a previous exp4 design "
            f"(train_iterations={prior_iterations}), but this script now uses "
            f"train_iterations={train_iterations}. Archive or delete the existing "
            f"'{os.path.dirname(raw_csv_path)}' runs/ directory and raw CSV before "
            "re-running, otherwise stale runs would be silently reused for any "
            "overlapping (beta_guess, n_guess) grid points."
        )


def main():
    ensure_project_directories()
    expected_runs = len(beta_guesses) * len(n_guesses) * len(seeds)
    _check_no_stale_results(
        os.path.join(results_dir, "run_manifest.json"),
        os.path.join(results_dir, "exp4_initial_guess_raw.csv"),
    )
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

    # 3D error landscape over the (beta0, n0) plane, built from the dense uniform
    # 13x13 grid. Unlike the old sparse grid, every adjacent pair of nodes here is
    # close together, so plot_surface's bilinear shading between them is an honest
    # reading of the surface rather than an interpolation across unmeasured gaps.
    combined_heatmap = np.full((len(n_guesses), len(beta_guesses)), np.nan)
    for row in summary_rows:
        bi = beta_guesses.index(row["beta_guess"])
        ni = n_guesses.index(row["n_guess"])
        combined_heatmap[ni, bi] = row["parameter_rel_error_mean"]

    log_error = np.log10(np.clip(combined_heatmap, 1e-4, None))
    X, Y = np.meshgrid(beta_guesses, n_guesses)

    plt.rcParams['axes.formatter.useoffset'] = False

    fig = plt.figure(figsize=(14, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    # mplot3d's tight_layout() call in finalize_figure() is a no-op (it can't
    # size 3D axes), which is why this figure had so much blank margin by
    # default -- shrinking the margins by hand instead is what actually
    # enlarges the plot relative to the canvas.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)

    # Tight, exact bounds everywhere (no default ~5% mplot3d margin) so the axis
    # panes meet the data at the corners instead of floating with a visible gap.
    finite_log_error = log_error[np.isfinite(log_error)]
    z_min, z_max = float(finite_log_error.min()), float(finite_log_error.max())

    # The surface and the colorbar share this same norm, spanning whole decades
    # (10^floor(min) to 10^ceil(max)) rather than the exact data min/max, so the
    # colorbar reads as a fixed general error-magnitude reference scale and the
    # colors on the surface actually match what it shows -- not just a bar
    # showing round numbers next to a surface colored on a different scale.
    decade_lo, decade_hi = int(np.floor(z_min)), int(np.ceil(z_max))
    norm = plt.Normalize(vmin=decade_lo, vmax=decade_hi)

    surf = ax.plot_surface(
        X, Y, log_error, cmap="viridis", norm=norm,
        edgecolor="0.3", linewidth=0.3, antialiased=True, alpha=0.95,
    )
    # Real measured grid points on top of the surface, small/translucent since the
    # grid is dense enough that the surface shading alone already reads clearly.
    ax.scatter(X.ravel(), Y.ravel(), log_error.ravel(), color="black", s=6, alpha=0.5, depthshade=False)

    # No ax.set_title(): exp2/3/5/6 carry no in-plot title either, relying on the
    # external figure caption -- matching that convention here instead of the
    # one-off title this figure had before.
    # azim=120 (not the earlier 35) is what makes mplot3d pick matching panes for
    # beta0/n0 -- at azim=35 their "0" ticks were drawn on opposite corners of the
    # box, and the z-axis landed on the left, away from the colorbar.
    ax.view_init(elev=25, azim=120)

    # Tight to the actual sampled range (not padded out to 0): the plotted region
    # should only ever cover grid points that were really measured.
    ax.set_xlim3d(min(beta_guesses), max(beta_guesses))
    ax.set_ylim3d(min(n_guesses), max(n_guesses))
    ax.set_zlim3d(z_min, z_max)
    ax.invert_yaxis()  # n0 runs high-to-low from front to back instead of low-to-high

    # Tick marks sit exactly at the tested grid values. Both axes happen to
    # start at 1.0, and at the shared front corner mplot3d draws both "1"
    # labels right on top of each other -- blanking the n0 one (keeping its
    # tick mark) avoids the doubled-up "1".
    ax.set_xticks(beta_guesses)
    ax.set_yticks(n_guesses)
    n_tick_labels = [f"{v:g}" for v in n_guesses]
    n_tick_labels[n_guesses.index(1.0)] = ""
    ax.set_yticklabels(n_tick_labels)

    # Uniform z-tick label offset from the axis line: mplot3d's default pad is a
    # fixed data-space (not screen-space) gap, so under perspective it looks
    # inconsistent -- the low ticks nearest the viewer end up almost touching the
    # axis line. A larger explicit pad keeps every label visibly clear of it.
    ax.zaxis.set_tick_params(pad=10)

    # Ticks are evenly spaced within the exact [z_min, z_max] data range (not
    # rounded outward to whole decades), so the colorbar's mapped range and its
    # displayed ticks always coincide -- rounding outward previously left a blank
    # strip on the colorbar above the true max value. Max 3 decimals.
    z_ticks = np.linspace(z_min, z_max, 5)
    ax.set_zticks(z_ticks)
    ax.set_zticklabels([f"{10**t:.3f}" for t in z_ticks])

    # x/y titles stay horizontal and z stays vertical -- simple, predictable,
    # and matches the other figures in this project rather than chasing an
    # exact on-screen "parallel to the axis line" angle.
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel(r"Initial $\beta_0$", fontsize=11, labelpad=10, rotation=0)
    ax.set_ylabel(r"Initial $n_0$", fontsize=11, labelpad=20, rotation=0)
    ax.set_zlabel("Combined Parameter Error", fontsize=11, labelpad=20, rotation=90)

    # Drop-line to the axis floor (the true data minimum, since the z-axis is now
    # tight to the data), in a color distinct from the black grid-point dots.
    TRUTH_COLOR = "#C62828"
    true_z = log_error[n_guesses.index(true_n), beta_guesses.index(true_beta)]
    ax.plot(
        [true_beta, true_beta], [true_n, true_n], [z_min, true_z],
        color=TRUTH_COLOR, linestyle="--", linewidth=2.2, zorder=10,
    )
    ax.scatter(
        [true_beta], [true_n], [true_z], color="black", s=45,
        edgecolor=TRUTH_COLOR, linewidth=0.9, depthshade=False, zorder=11,
        label=rf"True parameters ($\beta$={true_beta}, $n$={true_n})",
    )

    # Standard vertical colorbar next to the (also vertical) z-axis. Its ticks
    # are a fixed reference scale -- whole decades (10^0, 10^-1, ...) that a
    # reader can associate a color with an error magnitude by, independent of
    # exactly where this particular run's data happens to fall -- not the same
    # thing as the z-axis, which reports this run's actual measured values. It
    # carries its own "(log scale)" label now: since it shows different numbers
    # than the z-axis (not just a duplicate), that label is no longer redundant.
    cbar_ticks = list(range(decade_lo, decade_hi + 1))
    cbar = fig.colorbar(surf, ax=ax, shrink=0.45, aspect=18, pad=0.12, ticks=cbar_ticks)
    cbar.set_label("Error Magnitude (log scale)")
    # set_yticklabels() alone doesn't reliably register as a FixedFormatter here,
    # so finalize_figure()'s "only touch non-fixed formatters" sweep silently
    # overwrites it with raw decimal values right before saving (same issue as
    # before with the z-axis). Setting the FixedFormatter explicitly avoids that.
    cbar.ax.yaxis.set_major_formatter(mticker.FixedFormatter([rf"$10^{{{t}}}$" for t in cbar_ticks]))
    cbar_pos = cbar.ax.get_position()
    cbar.ax.set_position([cbar_pos.x0 - 0.05, cbar_pos.y0 + 0.06, cbar_pos.width, cbar_pos.height])

    # handletextpad pulls the marker in the legend swatch right up against its
    # text, and the legend sits above the plot (not inline in the 3D scene).
    ax.legend(loc="upper left", frameon=False, fontsize=9, handletextpad=0.3)

    # bbox_inches="tight" crops the saved PNG to the actual rendered content
    # (plot + legend + colorbar) instead of the full fixed-size canvas --
    # tight_layout() can't do this for 3D axes, which is why the figure had a
    # lot of unused blank margin otherwise.
    finalize_figure(figure_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
