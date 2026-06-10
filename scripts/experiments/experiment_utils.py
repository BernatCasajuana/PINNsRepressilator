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
import tensorflow as tf

from data.generate_data import protein_repressilator_rhs # Import the ODE function

# Default parameters for synthetic dataset generation
default_x0 = [1.0, 1.0, 1.2]
default_t_max = 20.0
default_n_points = 1000

# Function to ensure necessary directories exist
def ensure_project_directories():
    os.makedirs("datasets", exist_ok=True)
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

# Function to finalize and save a figure
def finalize_figure(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
