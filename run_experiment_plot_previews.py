#!/usr/bin/env python3
"""
Generate lightweight preview figures for all experiments.

This script is intended for visual checks (palette/style/layout) before long runs.
It overrides a small subset of experiment globals at runtime:
- seeds -> one seed
- train_iterations -> reduced iterations
- results_dir / figure_path -> preview-specific outputs
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def configure_runtime_environment() -> None:
    # Use canonical environment variable names expected by TensorFlow/DeepXDE.
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("DDE_BACKEND", "tensorflow")

    # Keep lowercase compatibility because some local scripts still use these keys.
    os.environ.setdefault("tf_use_legacy_keras", os.environ["TF_USE_LEGACY_KERAS"])
    os.environ.setdefault("dde_backend", os.environ["DDE_BACKEND"])


@dataclass(frozen=True)
class PreviewSpec:
    name: str
    module_path: str
    figure_filename: str
    results_subdir: str


SPECS = [
    PreviewSpec(
        name="exp_noise_sweep",
        module_path="scripts.experiments.exp_noise_sweep",
        figure_filename="preview_exp_noise_sweep.png",
        results_subdir="exp_noise_sweep",
    ),
    PreviewSpec(
        name="exp_partial_observation",
        module_path="scripts.experiments.exp_partial_observation",
        figure_filename="preview_exp_partial_observation.png",
        results_subdir="exp_partial_observation",
    ),
    PreviewSpec(
        name="exp_sampling_density",
        module_path="scripts.experiments.exp_sampling_density",
        figure_filename="preview_exp_sampling_density.png",
        results_subdir="exp_sampling_density",
    ),
    PreviewSpec(
        name="exp_initial_guess",
        module_path="scripts.experiments.exp_initial_guess",
        figure_filename="preview_exp_initial_guess.png",
        results_subdir="exp_initial_guess",
    ),
    PreviewSpec(
        name="exp_regime_comparison",
        module_path="scripts.experiments.exp_regime_comparison",
        figure_filename="preview_exp_regime_comparison.png",
        results_subdir="exp_regime_comparison",
    ),
]


def apply_overrides(module: Any, overrides: Dict[str, Any]) -> None:
    for attribute, value in overrides.items():
        if not hasattr(module, attribute):
            raise AttributeError(
                f"Module {module.__name__} has no attribute '{attribute}'. "
                "Update this preview script to match the experiment module."
            )
        setattr(module, attribute, value)


def run_preview(spec: PreviewSpec, train_iterations: int, seed: int) -> None:
    print(f"\nPreparing module import for {spec.name} ({spec.module_path})...")
    print("TensorFlow/DeepXDE backend initialization may take around 20-60 seconds on first import.")
    module = importlib.import_module(spec.module_path)

    preview_results_dir = os.path.join(ROOT, "results", "preview_figures", spec.results_subdir)
    preview_figure_path = os.path.join(ROOT, "figures", spec.figure_filename)

    overrides = {
        "seeds": [seed],
        "train_iterations": train_iterations,
        "results_dir": preview_results_dir,
        "figure_path": preview_figure_path,
    }

    apply_overrides(module, overrides)

    print(f"\\n=== Running {spec.name} preview ===")
    print(f"seeds={overrides['seeds']}, train_iterations={train_iterations}")
    print(f"results_dir={preview_results_dir}")
    print(f"figure_path={preview_figure_path}")

    module.main()


def main() -> None:
    configure_runtime_environment()

    parser = argparse.ArgumentParser(description="Run lightweight preview figures for all experiments.")
    parser.add_argument(
        "--train-iterations",
        type=int,
        default=200,
        help="Training iterations per inverse run for preview generation (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Single random seed used for preview runs (default: 0).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[spec.name for spec in SPECS],
        help="Run only specific preview experiment names.",
    )
    args = parser.parse_args()

    selected_specs = [spec for spec in SPECS if args.only is None or spec.name in args.only]

    print("Generating preview figures for selected experiments...")
    print(f"Selected: {[spec.name for spec in selected_specs]}")
    for spec in selected_specs:
        run_preview(spec, train_iterations=args.train_iterations, seed=args.seed)

    print("\\nAll preview experiments completed.")
    print("Preview figures saved under figures/preview_exp_*.png")


if __name__ == "__main__":
    main()
