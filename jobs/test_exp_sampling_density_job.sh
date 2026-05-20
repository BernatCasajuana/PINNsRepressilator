#!/bin/bash
#SBATCH --job-name=test_exp_sampling          # Nom del job
#SBATCH --output=test_exp_sampling_output.txt # Fitxer de sortida
#SBATCH --error=test_exp_sampling_error.txt   # Fitxer d'errors
#SBATCH --time=01:00:00                       # Temps maxim (hh:mm:ss)
#SBATCH --cpus-per-task=4                     # Nombre de CPUs per tasca
#SBATCH --mem=8GB                             # Memoria assignada

# Load conda module and activate environment
module load conda
conda activate pinnsvenv

# Working directory
cd ~/projects/pinns-repressilator

# Execute a lightweight test run for Experiment 3
python -u <<'PY'
from scripts.experiments import exp_sampling_density as exp

exp.observation_counts = [10]
exp.seeds = [0]
exp.train_iterations = 100
exp.results_dir = "results/test_jobs/exp_sampling_density"
exp.figure_path = "figures/test_jobs/exp_sampling_density.png"

print("=== Running quick test: exp_sampling_density ===")
print(f"observation_counts={exp.observation_counts}, seeds={exp.seeds}, train_iterations={exp.train_iterations}")
exp.main()
PY
