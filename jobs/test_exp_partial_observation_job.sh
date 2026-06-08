#!/bin/bash
#SBATCH --job-name=test_exp_partial_obs       # Nom del job
#SBATCH --output=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/test_exp_partial_obs_output.txt # Fitxer de sortida
#SBATCH --error=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/test_exp_partial_obs_error.txt # Fitxer d'error
#SBATCH --time=01:00:00                       # Temps maxim (hh:mm:ss)
#SBATCH --cpus-per-task=4                     # Nombre de CPUs per tasca
#SBATCH --mem=8GB                             # Memoria assignada

# Load conda module and activate environment
source /opt/software/miniconda3/bin/activate pinns-repressilator-venv

# Working directory
cd /home/10040984@uvic.local/projects/pinns-repressilator

# Set PYTHONPATH to include the current directory
export PYTHONPATH=/home/10040984@uvic.local/projects/pinns-repressilator:$PYTHONPATH

# Execute a lightweight test run for Experiment 2
python -u <<'PY'
from scripts.experiments import exp_partial_observation as exp

exp.seeds = [0]
exp.observation_designs = [
    ("x1,x2,x3", [0, 1, 2]),
    ("x1", [0]),
]
exp.train_iterations = 100

ROOT = "/home/10040984@uvic.local/projects/pinns-repressilator"

exp.results_dir = ROOT + "/results/test_exp_partial_observation"
exp.figure_path = ROOT + "/figures/test_exp_partial_observation.png"

print("=== Running quick test: exp_partial_observation ===")
print(f"designs={exp.observation_designs}, seeds={exp.seeds}, train_iterations={exp.train_iterations}")
exp.main()
PY
