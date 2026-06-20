"""
Utilities shared by the experiment driver scripts.
"""

import csv
import json
import os
import random
import sys
from datetime import datetime, timezone

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1") # Ensure compatibility with DeepXDE's use of Keras
os.environ.setdefault("DDE_BACKEND", "tensorflow") # Ensure DeepXDE uses TensorFlow as the backend (compatible)

# Lowercase compatibility for local scripts that still read these keys.
os.environ.setdefault("tf_use_legacy_keras", os.environ["TF_USE_LEGACY_KERAS"])
os.environ.setdefault("dde_backend", os.environ["DDE_BACKEND"])

# Add the parent directory of "scripts" to the Python path to allow imports from "data" and "pinns"
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.stats
import tensorflow as tf

from datasets.data import protein_repressilator_rhs

# Match the paper's LaTeX typesetting (Computer Modern) without requiring a LaTeX install.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

# Default parameters for synthetic dataset generation
default_x0 = [1.0, 1.0, 1.2]
default_t_max = 20.0
default_n_points = 1000

# Function to ensure necessary directories exist
def ensure_project_directories():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

# Function to set global random seeds for reproducibility
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    dde.config.set_random_seed(seed)

# Function to simulate the system and generate clean data
def simulate_repressilator(beta, n, x0=None, t_max=default_t_max, n_points=default_n_points):
    if x0 is None:
        x0 = default_x0
    t = np.linspace(0, t_max, n_points)[:, None]
    y_clean = scipy.integrate.odeint(protein_repressilator_rhs, x0, t.flatten(), args=(beta, n))
    return t, y_clean

# Function to create a synthetic dataset with added noise
def make_synthetic_dataset(
    beta,
    n,
    noise_level,
    seed,
    x0=None,
    t_max=default_t_max,
    n_points=default_n_points,
):
    t, y_clean = simulate_repressilator(beta=beta, n=n, x0=x0, t_max=t_max, n_points=n_points)
    signal_amplitude = float(np.mean(np.ptp(y_clean, axis=0)))
    noise_sigma = noise_level * signal_amplitude
    rng = np.random.default_rng(seed)
    y_noisy = y_clean + rng.normal(0.0, noise_sigma, size=y_clean.shape)
    return {
        "name": f"beta{beta}_n{n}_noise{noise_level}_seed{seed}",
        "t": t,
        "y": y_noisy,
        "y_clean": y_clean,
        "beta": beta,
        "n": n,
        "noise": noise_sigma,
        "noise_level": noise_level,
        "signal_amplitude": signal_amplitude,
    }

# Function to compute observation indices
def evenly_spaced_observation_indices(total_points, observation_count):
    if observation_count <= 0:
        raise ValueError(f"observation_count must be a positive integer. Got {observation_count}.")
    if observation_count >= total_points:
        return list(range(total_points))
    return np.unique(np.linspace(0, total_points - 1, observation_count, dtype=int)).tolist()

# Function to store results in a structured CSV file
def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_run_manifest(path, manifest_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(manifest_data)
    payload["generated_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with open(path, "w", encoding="utf-8") as manifest_file:
        json.dump(payload, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")

# Function to aggregate metrics from multiple runs
def aggregate_metrics(rows, group_keys, metric_keys):
    groups = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for key, group_rows in groups.items():
        summary_row = {group_key: key[index] for index, group_key in enumerate(group_keys)}
        summary_row["num_runs"] = len(group_rows)
        for metric_key in metric_keys:
            values = np.array([float(row[metric_key]) for row in group_rows], dtype=float)
            summary_row[f"{metric_key}_mean"] = float(np.mean(values))
            summary_row[f"{metric_key}_std"] = float(np.std(values, ddof=0))
        summary_rows.append(summary_row)

    return summary_rows


def metric_values_by_group(rows, group_key, metric_key):
    grouped_values = {}
    for row in rows:
        group_value = row[group_key]
        grouped_values.setdefault(group_value, []).append(float(row[metric_key]))
    return grouped_values


def mann_whitney_p_value(sample_a, sample_b):
    values_a = np.asarray(sample_a, dtype=float)
    values_b = np.asarray(sample_b, dtype=float)
    if values_a.size < 2 or values_b.size < 2:
        return float("nan")
    result = scipy.stats.mannwhitneyu(values_a, values_b, alternative="two-sided", method="asymptotic")
    return float(result.pvalue)


def jonckheere_terpstra_p_value(groups_in_order):
    """Two-sided Jonckheere-Terpstra test for monotonic trend across ordered groups.

    groups_in_order: list of array-like, ordered from the group expected to have the
    smallest values to the group expected to have the largest values.
    Uses a normal approximation (appropriate for n >= 5 per group).
    Returns two-sided p-value.
    """
    groups = [np.asarray(g, dtype=float) for g in groups_in_order]
    n_i = [len(g) for g in groups]
    k = len(groups)
    N = sum(n_i)
    if k < 2 or any(ni < 1 for ni in n_i):
        return float("nan")
    jt_stat = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            result = scipy.stats.mannwhitneyu(groups[i], groups[j], alternative="less", method="asymptotic")
            jt_stat += float(result.statistic)
    e_j = (N ** 2 - sum(ni ** 2 for ni in n_i)) / 4.0
    var_j = (N ** 2 * (2 * N + 3) - sum(ni ** 2 * (2 * ni + 3) for ni in n_i)) / 72.0
    if var_j <= 0:
        return float("nan")
    z = (jt_stat - e_j) / np.sqrt(var_j)
    return 2.0 * float(scipy.stats.norm.sf(abs(z)))


def holm_bonferroni_adjust(p_values):
    adjusted_p_values = [float("nan")] * len(p_values)
    valid_indexed_p_values = [
        (index, float(p_value))
        for index, p_value in enumerate(p_values)
        if np.isfinite(p_value)
    ]
    if not valid_indexed_p_values:
        return adjusted_p_values

    valid_indexed_p_values.sort(key=lambda item: item[1])
    total_tests = len(valid_indexed_p_values)
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(valid_indexed_p_values):
        adjusted = (total_tests - rank) * p_value
        running_max = max(running_max, adjusted)
        adjusted_p_values[original_index] = min(1.0, running_max)

    return adjusted_p_values


def p_value_to_stars(p_value):
    if not np.isfinite(p_value):
        return ""
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return ""


def format_p_value_label(p_value):
    if not np.isfinite(p_value):
        return "p=n/a"
    if p_value < 1e-3:
        return f"p={p_value:.1e}"
    return f"p={p_value:.3f}"


def pairwise_significance(values_by_group, comparisons):
    raw_p_values = []
    for group_a, group_b in comparisons:
        raw_p_values.append(
            mann_whitney_p_value(
                values_by_group.get(group_a, []),
                values_by_group.get(group_b, []),
            )
        )

    adjusted_p_values = holm_bonferroni_adjust(raw_p_values)
    significance = {}
    for (group_a, group_b), raw_p_value, adjusted_p_value in zip(comparisons, raw_p_values, adjusted_p_values):
        significance[(group_a, group_b)] = {
            "raw_p_value": raw_p_value,
            "adjusted_p_value": adjusted_p_value,
            "stars": p_value_to_stars(adjusted_p_value),
        }
    return significance


def significance_label(significance_entry, use_adjusted_p_value=True):
    if not significance_entry:
        return "p=n/a"
    p_value_key = "adjusted_p_value" if use_adjusted_p_value else "raw_p_value"
    p_value = float(significance_entry.get(p_value_key, float("nan")))
    stars = p_value_to_stars(p_value)
    p_value_label = format_p_value_label(p_value)
    return f"{p_value_label} {stars}".strip()


def annotate_pairwise_comparisons(
    axis,
    x_positions,
    top_values,
    comparisons,
    significance,
    use_adjusted_p_value=True,
    only_significant=False,
):
    if not comparisons:
        return

    finite_top_values = np.asarray(
        [
            float(top_value)
            for top_value in top_values.values()
            if np.isfinite(top_value)
        ],
        dtype=float,
    )
    if finite_top_values.size == 0:
        return

    value_span = float(np.ptp(finite_top_values))
    if value_span == 0.0:
        value_span = max(float(np.max(np.abs(finite_top_values))), 1.0)

    tick_height = 0.03 * value_span
    stack_step = 0.12 * value_span
    label_offset = 0.02 * value_span

    p_value_key = "adjusted_p_value" if use_adjusted_p_value else "raw_p_value"

    valid_comparisons = []
    for group_a, group_b in comparisons:
        if (
            group_a in x_positions
            and group_b in x_positions
            and group_a in top_values
            and group_b in top_values
        ):
            if only_significant:
                entry = significance.get((group_a, group_b)) or significance.get((group_b, group_a))
                p_value = float(entry.get(p_value_key, float("nan"))) if entry else float("nan")
                if not p_value_to_stars(p_value):
                    continue
            valid_comparisons.append((group_a, group_b))
    if not valid_comparisons:
        return

    current_bottom, current_top = axis.get_ylim()
    max_top_value = float(np.max(finite_top_values))
    required_top = max_top_value + (len(valid_comparisons) + 2) * stack_step
    if required_top > current_top:
        axis.set_ylim(current_bottom, required_top)

    for comparison_index, (group_a, group_b) in enumerate(valid_comparisons, start=1):
        x_a = float(x_positions[group_a])
        x_b = float(x_positions[group_b])
        if x_a > x_b:
            x_a, x_b = x_b, x_a

        base_height = max(float(top_values[group_a]), float(top_values[group_b]))
        y_line = base_height + comparison_index * stack_step
        axis.plot(
            [x_a, x_a, x_b, x_b],
            [y_line - tick_height, y_line, y_line, y_line - tick_height],
            color="black",
            linewidth=0.9,
        )

        comparison_significance = significance.get((group_a, group_b))
        if comparison_significance is None:
            comparison_significance = significance.get((group_b, group_a))
        label = significance_label(
            comparison_significance,
            use_adjusted_p_value=use_adjusted_p_value,
        )
        axis.text(
            0.5 * (x_a + x_b),
            y_line + label_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def annotate_significance_markers(axis, positions, top_values, labels):
    if len(positions) == 0 or len(top_values) == 0 or len(labels) == 0:
        return

    finite_top_values = np.asarray(
        [
            float(top_value)
            for top_value in top_values
            if np.isfinite(top_value)
        ],
        dtype=float,
    )
    if finite_top_values.size == 0:
        return

    value_span = float(np.ptp(finite_top_values))
    if value_span == 0.0:
        value_span = max(float(np.max(np.abs(finite_top_values))), 1.0)

    text_offset = 0.06 * value_span
    current_bottom, current_top = axis.get_ylim()
    required_top = float(np.max(finite_top_values)) + 3.0 * text_offset
    if required_top > current_top:
        axis.set_ylim(current_bottom, required_top)

    for position, top_value, label in zip(positions, top_values, labels):
        if not label:
            continue
        axis.text(
            position,
            float(top_value) + text_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

# Function to finalize and save a figure
def finalize_figure(path):
    import matplotlib.ticker as mticker
    _fmt = mticker.FuncFormatter(lambda x, _: f'{x:.3g}')
    for ax in plt.gcf().get_axes():
        if ax.get_yscale() != 'log':
            if not isinstance(ax.yaxis.get_major_formatter(), mticker.FixedFormatter):
                ax.yaxis.set_major_formatter(_fmt)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
