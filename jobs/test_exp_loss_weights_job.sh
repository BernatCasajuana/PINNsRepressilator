#!/bin/bash
#SBATCH --job-name=test_exp_loss_weights   # Nom del job
#SBATCH --output=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/test_exp_loss_weights_output.txt # Fitxer de sortida
#SBATCH --error=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/test_exp_loss_weights_error.txt # Fitxer d'error
#SBATCH --time=02:00:00                    # Temps maxim (hh:mm:ss)
#SBATCH --cpus-per-task=1                  # Nombre de CPUs per tasca
#SBATCH --mem=4GB                          # Memoria assignada

# Load conda module and activate environment
source /opt/software/miniconda3/bin/activate pinns-repressilator-venv

# Working directory
cd /home/10040984@uvic.local/projects/pinns-repressilator

# Set PYTHONPATH to include the current directory
export PYTHONPATH=/home/10040984@uvic.local/projects/pinns-repressilator:$PYTHONPATH

# Execute a lightweight test run for Experiment 6
python -u <<'PY'
from scripts.experiments import exp6_loss_weights as exp

exp.train_iterations = 1000

ROOT = "/home/10040984@uvic.local/projects/pinns-repressilator"

exp.results_dir = ROOT + "/results/test_exp6_loss_weights"
exp.table_path  = ROOT + "/tables/test_exp6_loss_weights.tex"

print("=== Running quick test: exp6_loss_weights ===")
print(f"train_iterations={exp.train_iterations}")
exp.main()
PY
