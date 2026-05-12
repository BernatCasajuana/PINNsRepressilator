"""
Runs all experiment drivers in the experiments package sequentially.
Each experiment is responsible for generating its own results and figures.
"""

import os
import sys

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.exp_initial_guess import scripts_dir, main as run_exp_initial_guess
from experiments.exp_noise_sweep import main as run_exp_noise_sweep
from experiments.exp_partial_observation import main as run_exp_partial_observation
from experiments.exp_regime_comparison import main as run_exp_regime_comparison
from experiments.exp_sampling_density import main as run_exp_sampling_density


def main():
    experiments = [
        ("exp_noise_sweep", run_exp_noise_sweep),
        ("exp_partial_observation", run_exp_partial_observation),
        ("exp_sampling_density", run_exp_sampling_density),
        ("exp_initial_guess", run_exp_initial_guess),
        ("exp_regime_comparison", run_exp_regime_comparison),
    ]

    for experiment_name, experiment_main in experiments:
        print(f"\n=== Running {experiment_name} ===")
        experiment_main()


if __name__ == "__main__":
    main()