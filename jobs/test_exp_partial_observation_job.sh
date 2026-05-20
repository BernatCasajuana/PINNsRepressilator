#!/bin/bash
#SBATCH --job-name=test_exp_partial_obs       # Nom del job
#SBATCH --output=test_exp_partial_obs_output.txt  # Fitxer de sortida
#SBATCH --error=test_exp_partial_obs_error.txt    # Fitxer d'errors
#SBATCH --time=01:00:00                       # Temps maxim (hh:mm:ss)
#SBATCH --cpus-per-task=4                     # Nombre de CPUs per tasca
#SBATCH --mem=8GB                             # Memoria assignada

# Load conda module and activate environment
module load conda
conda activate pinnsvenv

# Working directory
cd ~/projects/pinns-repressilator

# Execute a lightweight test run for Experiment 2
python -u <<'PY'
from scripts.experiments import exp_partial_observation as exp

exp.seeds = [0]
exp.observation_designs = [
    ("x1,x2,x3", [0, 1, 2]),
    ("x1", [0]),
]
exp.train_iterations = 100
exp.results_dir = "results/test_jobs/exp_partial_observation"
exp.figure_path = "figures/test_jobs/exp_partial_observation.png"

print("=== Running quick test: exp_partial_observation ===")
print(f"designs={exp.observation_designs}, seeds={exp.seeds}, train_iterations={exp.train_iterations}")
exp.main()
PY
