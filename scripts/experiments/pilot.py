#!/usr/bin/env python3
"""
Generate lightweight pilot figures for all experiments.

This script is intended for visual checks (palette/style/layout) before long runs.
It overrides a small subset of experiment globals at runtime:
  - train_iterations → reduced (default 100)
  - seeds            → reduced (default: experiment's own seed list)
  - results_dir    → results/pilot/<exp_name>
  - figure_path    → figures/pilot_<exp_name>.png

Usage:
  python scripts/experiments/pilot.py                          # all experiments, 100 iters, full seed set
  python scripts/experiments/pilot.py --train-iterations 500   # slightly more converged check
  python scripts/experiments/pilot.py --seeds 1                # 1 seed per experiment
  python scripts/experiments/pilot.py --only exp1_noise_sweep exp4_initial_guess
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict


# Project root, i.e. two levels above this file (scripts/experiments/pilot.py)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def configure_runtime_environment() -> None:
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("DDE_BACKEND", "tensorflow")
    os.environ.setdefault("tf_use_legacy_keras", os.environ["TF_USE_LEGACY_KERAS"])
    os.environ.setdefault("dde_backend", os.environ["DDE_BACKEND"])


@dataclass(frozen=True)
class PreviewSpec:
    name: str
    module_path: str


# Ordered by scientific priority (1 = most central to paper thesis)
SPECS = [
    PreviewSpec(name="exp1_noise_sweep",         module_path="scripts.experiments.exp1_noise_sweep"),
    PreviewSpec(name="exp2_partial_observation", module_path="scripts.experiments.exp2_partial_observation"),
    PreviewSpec(name="exp3_sampling_density",    module_path="scripts.experiments.exp3_sampling_density"),
    PreviewSpec(name="exp4_initial_guess",       module_path="scripts.experiments.exp4_initial_guess"),
    PreviewSpec(name="exp5_regime_comparison",   module_path="scripts.experiments.exp5_regime_comparison"),
    PreviewSpec(name="exp6_loss_weights",        module_path="scripts.experiments.exp6_loss_weights"),
    PreviewSpec(name="exp7_convergence",         module_path="scripts.experiments.exp7_convergence"),
]


def apply_overrides(module: Any, overrides: Dict[str, Any]) -> None:
    for attribute, value in overrides.items():
        if not hasattr(module, attribute):
            raise AttributeError(
                f"Module {module.__name__} has no attribute '{attribute}'. "
                "Update this pilot script to match the experiment module."
            )
        setattr(module, attribute, value)


def run_preview(spec: PreviewSpec, train_iterations: int, n_seeds: int | None) -> None:
    print(f"\nLoading {spec.name} ({spec.module_path})...")
    module = importlib.import_module(spec.module_path)

    pilot_results_dir = os.path.join(ROOT, "results", "pilot", spec.name)
    pilot_figure_path = os.path.join(ROOT, "figures", f"pilot_{spec.name}.png")

    overrides: Dict[str, Any] = {
        "train_iterations": train_iterations,
        "results_dir": pilot_results_dir,
    }
    if hasattr(module, "figure_path"):
        overrides["figure_path"] = pilot_figure_path
    if hasattr(module, "table_path"):
        overrides["table_path"] = os.path.join(ROOT, "tables", f"pilot_{spec.name}.tex")
    if hasattr(module, "observability_table_path"):
        overrides["observability_table_path"] = os.path.join(ROOT, "tables", "pilot_observability.tex")
    if hasattr(module, "exp2_results_dir"):
        overrides["exp2_results_dir"] = os.path.join(ROOT, "results", "pilot", "exp2_partial_observation")
    if n_seeds is not None:
        original_seeds = getattr(module, "seeds", [0])
        overrides["seeds"] = original_seeds[:n_seeds]
    apply_overrides(module, overrides)

    seeds_used = getattr(module, "seeds", "?")
    print(f"=== Running {spec.name} pilot ===")
    print(f"  seeds={seeds_used}, train_iterations={train_iterations}")
    print(f"  results_dir={pilot_results_dir}")
    print(f"  figure_path={pilot_figure_path}")

    module.main()


def main() -> None:
    configure_runtime_environment()

    parser = argparse.ArgumentParser(description="Run lightweight pilot figures for all experiments.")
    parser.add_argument(
        "--train-iterations", type=int, default=100,
        help="Training iterations per run (default: 100).",
    )
    parser.add_argument(
        "--only", nargs="+",
        choices=[spec.name for spec in SPECS],
        help="Run only specific pilot experiment names.",
    )
    parser.add_argument(
        "--seeds", type=int, default=None, metavar="N",
        help="Use only the first N seeds from each experiment (default: keep all).",
    )
    args = parser.parse_args()

    selected = [spec for spec in SPECS if args.only is None or spec.name in args.only]

    print(f"Generating pilot figures for: {[spec.name for spec in selected]}")
    for spec in selected:
        run_preview(spec, train_iterations=args.train_iterations, n_seeds=args.seeds)

    print("\nAll pilot experiments completed.")
    print("Pilot figures saved under figures/pilot_*.png")


if __name__ == "__main__":
    main()
