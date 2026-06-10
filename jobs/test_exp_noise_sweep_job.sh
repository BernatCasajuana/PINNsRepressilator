#!/bin/bash
#SBATCH --job-name=test_exp_noise_sweep       # Nom del job
#SBATCH --output=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/test_exp_noise_sweep_output.txt # Fitxer de sortida
#SBATCH --error=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/test_exp_noise_sweep_error.txt # Fitxer d'error
#SBATCH --time=01:00:00                       # Temps maxim (hh:mm:ss)
#SBATCH --cpus-per-task=4                     # Nombre de CPUs per tasca
#SBATCH --mem=8GB                             # Memoria assignada

# Load conda module and activate environment
source /opt/software/miniconda3/bin/activate pinns-repressilator-venv

# Working directory
cd /home/10040984@uvic.local/projects/pinns-repressilator

# Set PYTHONPATH to include the current directory
export PYTHONPATH=/home/10040984@uvic.local/projects/pinns-repressilator:$PYTHONPATH

# Execute a lightweight test run for Experiment 1
python -u <<'PY'
from scripts.experiments import exp_noise_sweep as exp

exp.train_iterations = 200

ROOT = "/home/10040984@uvic.local/projects/pinns-repressilator"

exp.results_dir = ROOT + "/results/test_exp_noise_sweep"
exp.figure_path = ROOT + "/figures/test_exp_noise_sweep.png"

print("=== Running quick test: exp_noise_sweep ===")
print(f"train_iterations={exp.train_iterations} (default experiment setup)")
exp.main()
PY
