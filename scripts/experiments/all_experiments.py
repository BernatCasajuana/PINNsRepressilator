"""
Runs all experiment drivers in the experiments package sequentially.
Each experiment is responsible for generating its own results and figures.

Execution order reflects scientific priority (1 = most central to the paper's thesis):
  1. forward_vs_inverse   — core finding: RMSE ≠ parameter accuracy
  2. noise_sweep          — primary robustness sweep
  3. partial_observation  — measurement design constraints
  4. sampling_density     — data density requirements
  5. initial_guess        — non-convex optimisation landscape
  6. regime_comparison    — stable vs oscillatory identifiability
  7. loss_weights         — physics loss weight sensitivity
  8. convergence          — training dynamics diagnostic
"""

import os
import sys

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.exp1_forward_vs_inverse import main as run_exp1
from experiments.exp2_noise_sweep import main as run_exp2
from experiments.exp3_partial_observation import main as run_exp3
from experiments.exp4_sampling_density import main as run_exp4
from experiments.exp5_initial_guess import main as run_exp5
from experiments.exp6_regime_comparison import main as run_exp6
from experiments.exp7_loss_weights import main as run_exp7
from experiments.exp8_convergence import main as run_exp8


def main():
    experiments = [
        ("exp1_forward_vs_inverse",  run_exp1),
        ("exp2_noise_sweep",         run_exp2),
        ("exp3_partial_observation", run_exp3),
        ("exp4_sampling_density",    run_exp4),
        ("exp5_initial_guess",       run_exp5),
        ("exp6_regime_comparison",   run_exp6),
        ("exp7_loss_weights",        run_exp7),
        ("exp8_convergence",         run_exp8),
    ]

    for experiment_name, experiment_main in experiments:
        print(f"\n=== Running {experiment_name} ===")
        experiment_main()


if __name__ == "__main__":
    main()
