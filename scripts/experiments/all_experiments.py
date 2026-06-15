"""
Runs all experiment drivers in the experiments package sequentially.
Each experiment is responsible for generating its own results and figures.

Execution order reflects scientific priority (1 = most central to the paper's thesis):
  1. noise_sweep          — primary robustness sweep + PINN vs L-BFGS-B baseline
  2. partial_observation  — measurement design constraints
  3. sampling_density     — data density requirements
  4. initial_guess        — non-convex optimisation landscape
  5. regime_comparison    — stable vs oscillatory identifiability
  6. loss_weights         — physics loss weight sensitivity
  7. convergence          — training dynamics diagnostic
"""

import os
import sys

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.exp1_noise_sweep import main as run_exp1
from experiments.exp2_partial_observation import main as run_exp2
from experiments.exp3_sampling_density import main as run_exp3
from experiments.exp4_initial_guess import main as run_exp4
from experiments.exp5_regime_comparison import main as run_exp5
from experiments.exp6_loss_weights import main as run_exp6
from experiments.exp7_convergence import main as run_exp7


def main():
    experiments = [
        ("exp1_noise_sweep",         run_exp1),
        ("exp2_partial_observation", run_exp2),
        ("exp3_sampling_density",    run_exp3),
        ("exp4_initial_guess",       run_exp4),
        ("exp5_regime_comparison",   run_exp5),
        ("exp6_loss_weights",        run_exp6),
        ("exp7_convergence",         run_exp7),
    ]

    for experiment_name, experiment_main in experiments:
        print(f"\n=== Running {experiment_name} ===")
        experiment_main()


if __name__ == "__main__":
    main()
