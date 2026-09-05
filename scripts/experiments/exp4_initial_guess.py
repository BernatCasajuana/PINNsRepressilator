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

Figure: 3D error-landscape surface over the (Δβ₀, Δn₀) plane, where Δβ₀ = β₀ − β_true
and Δn₀ = n₀ − n_true — z = log10(combined relative parameter error). Axes show the
OFFSET of the initial guess from the truth (not the raw guess value), so the true
parameters always sit at the grid centre (0, 0). No separate marker or legend is
needed for that point: it's already labelled by the axes and already plotted as one
of the ordinary grid-point dots.
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
from matplotlib.text import Text

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
    # 9x9 grid. Unlike the old sparse grid, every adjacent pair of nodes here is
    # close together, so plot_surface's bilinear shading between them is an honest
    # reading of the surface rather than an interpolation across unmeasured gaps.
    combined_heatmap = np.full((len(n_guesses), len(beta_guesses)), np.nan)
    for row in summary_rows:
        bi = beta_guesses.index(row["beta_guess"])
        ni = n_guesses.index(row["n_guess"])
        combined_heatmap[ni, bi] = row["parameter_rel_error_mean"]

    log_error = np.log10(np.clip(combined_heatmap, 1e-4, None))

    # Plot axes show the OFFSET of each initial guess from the truth, not the raw
    # guess value, so the true parameters land at (0, 0) -- the grid centre -- no
    # matter what beta_true/n_true happen to be. bi/ni indices above are unaffected:
    # they still key off the absolute beta_guesses/n_guesses used to run the PINNs.
    beta_deltas = [round(b - true_beta, 10) for b in beta_guesses]
    n_deltas = [round(n - true_n, 10) for n in n_guesses]
    X, Y = np.meshgrid(beta_deltas, n_deltas)

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
    ax.set_xlim3d(min(beta_deltas), max(beta_deltas))
    ax.set_ylim3d(min(n_deltas), max(n_deltas))
    ax.set_zlim3d(z_min, z_max)
    ax.invert_yaxis()  # n0 runs high-to-low from front to back instead of low-to-high

    # Tick marks sit exactly at the tested grid offsets. Unlike the old raw-value
    # axes (which both happened to start at 1.0 and drew a doubled-up "1" label at
    # the shared front corner), the min/max offsets differ between beta and n here,
    # so no corner tick needs to be blanked out.
    ax.set_xticks(beta_deltas)
    ax.set_yticks(n_deltas)
    ax.set_yticklabels([f"{v:g}" for v in n_deltas])

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
    # Delta notation (rather than spelling out "beta0 - beta_true") reads as
    # "distance/difference from truth" at a glance -- the true parameters sit at
    # offset (0, 0) by construction, so no separate marker is needed to point
    # that out (see below).
    ax.set_xlabel(r"$\Delta\beta_0 = \beta_0 - \beta_{\mathrm{true}}$", labelpad=10, rotation=0, fontsize=18)
    ax.set_ylabel(r"$\Delta n_0 = n_0 - n_{\mathrm{true}}$", labelpad=20, rotation=0, fontsize=18)
    ax.set_zlabel("Combined Parameter Error", labelpad=20, rotation=90, fontsize=15)
    # mplot3d hardcodes label ha/va to 'center' in a class-level _AXINFO table
    # shared across all three axes, and reapplies it on every draw (see
    # axis3d.py _draw_labels), which silently overrides any ha= passed to
    # set_xlabel/set_ylabel above. Swapping in a fresh 'label' dict on just
    # this axis (instead of mutating the shared one, which would also shift
    # the y/z labels) is what actually nudges the label sideways off its
    # computed anchor and survives the redraw.
    ax.xaxis._axinfo["label"] = {**ax.xaxis._axinfo["label"], "ha": "right"}
    ax.yaxis._axinfo["label"] = {**ax.yaxis._axinfo["label"], "ha": "left"}

    def _nudge_label_x(label, dx_points):
        """Pull a 3D axis label back toward center by a fixed screen-space
        amount, on top of whatever the ha='right'/'left' override above and
        mplot3d's own per-draw position computation already place it at.

        mplot3d recomputes and overwrites label.set_position(...) on every
        draw (axis3d.py _draw_labels) using data-space coordinates, so a
        one-off position/transform tweak made now would just be discarded at
        render time. Shadowing set_position with a wrapper that intercepts
        each of those calls and offsets them in display space (points) is
        what makes the nudge survive every redraw, including the extra ones
        bbox_inches='tight' triggers while computing the crop.
        """
        original_set_position = Text.set_position.__get__(label)
        dx_pixels = dx_points * fig.dpi / 72.0

        def wrapped(xy):
            trans = label.get_transform()
            dx_disp, dy_disp = trans.transform(xy)
            original_set_position(trans.inverted().transform((dx_disp + dx_pixels, dy_disp)))

        label.set_position = wrapped

    _nudge_label_x(ax.xaxis.label, dx_points=12)   # beta label: pull back right
    _nudge_label_x(ax.yaxis.label, dx_points=-28)  # n label: pull back left

    # No separate truth marker/legend: the axis labels already say (0, 0) is the
    # truth, and the (0, 0) point is already part of the black grid-point dots
    # plotted above -- adding a second, differently-colored marker there would
    # just be redundant.

    # Standard vertical colorbar next to the (also vertical) z-axis. Its ticks
    # are a fixed reference scale -- whole decades (10^0, 10^-1, ...) that a
    # reader can associate a color with an error magnitude by, independent of
    # exactly where this particular run's data happens to fall -- not the same
    # thing as the z-axis, which reports this run's actual measured values. It
    # carries its own "(log scale)" label now: since it shows different numbers
    # than the z-axis (not just a duplicate), that label is no longer redundant.
    cbar_ticks = list(range(decade_lo, decade_hi + 1))
    cbar = fig.colorbar(surf, ax=ax, shrink=0.45, aspect=18, pad=0.12, ticks=cbar_ticks)
    # Colorbar label reads left-to-right as "title, then bar, then tick numbers":
    # moved off the default right-hand label position (which stacks it past the
    # tick numbers, outside the bar) onto the left instead.
    cbar.ax.yaxis.set_label_position("left")
    cbar.set_label("Error Magnitude (log scale)", labelpad=10)
    # set_yticklabels() alone doesn't reliably register as a FixedFormatter here,
    # so finalize_figure()'s "only touch non-fixed formatters" sweep silently
    # overwrites it with raw decimal values right before saving (same issue as
    # before with the z-axis). Setting the FixedFormatter explicitly avoids that.
    cbar.ax.yaxis.set_major_formatter(mticker.FixedFormatter([rf"$10^{{{t}}}$" for t in cbar_ticks]))
    # Moved from its default right-hand slot (next to the z-axis) into the empty
    # upper-left region of the figure. fig.colorbar() doesn't support that
    # placement directly, so the axes position is overridden by hand afterward.
    cbar_pos = cbar.ax.get_position()
    cbar.ax.set_position([0.27, 0.38, cbar_pos.width, cbar_pos.height])

    # bbox_inches="tight" crops the saved PNG to the actual rendered content
    # (plot + colorbar) instead of the full fixed-size canvas -- tight_layout()
    # can't do this for 3D axes, which is why the figure had a lot of unused
    # blank margin otherwise. mplot3d's get_tightbbox() doesn't reliably include
    # the z-axis label in that automatic bbox, so it gets silently cropped off
    # unless it's passed explicitly via bbox_extra_artists.
    finalize_figure(
        figure_path, bbox_inches="tight",
        bbox_extra_artists=[ax.xaxis.label, ax.yaxis.label, ax.zaxis.label],
    )


if __name__ == "__main__":
    main()
