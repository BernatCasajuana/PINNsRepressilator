"""Utilities shared by the PINN experiment driver scripts."""

import csv
import os
import random

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["DDE_BACKEND"] = "tensorflow"

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import tensorflow as tf

from generate_data import protein_repressilator_rhs


DEFAULT_X0 = [1.0, 1.0, 1.2]
DEFAULT_T_MAX = 20.0
DEFAULT_N_POINTS = 1000


def ensure_project_directories():
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    dde.config.set_random_seed(seed)


def simulate_repressilator(beta, n, x0=None, t_max=DEFAULT_T_MAX, n_points=DEFAULT_N_POINTS):
    if x0 is None:
        x0 = DEFAULT_X0
    t = np.linspace(0, t_max, n_points)[:, None]
    y_clean = scipy.integrate.odeint(protein_repressilator_rhs, x0, t.flatten(), args=(beta, n))
    return t, y_clean


def make_synthetic_dataset(
    beta,
    n,
    noise_level,
    seed,
    x0=None,
    t_max=DEFAULT_T_MAX,
    n_points=DEFAULT_N_POINTS,
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


def evenly_spaced_observation_indices(total_points, observation_count):
    if observation_count >= total_points:
        return list(range(total_points))
    return np.unique(np.linspace(0, total_points - 1, observation_count, dtype=int)).tolist()


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def finalize_figure(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
