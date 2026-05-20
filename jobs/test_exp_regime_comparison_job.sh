#!/bin/bash
#SBATCH --job-name=test_exp_regime            # Nom del job
#SBATCH --output=test_exp_regime_output.txt   # Fitxer de sortida
#SBATCH --error=test_exp_regime_error.txt     # Fitxer d'errors
#SBATCH --time=01:00:00                       # Temps maxim (hh:mm:ss)
#SBATCH --cpus-per-task=4                     # Nombre de CPUs per tasca
#SBATCH --mem=8GB                             # Memoria assignada

# Load conda module and activate environment
module load conda
conda activate pinnsvenv

# Working directory
cd ~/projects/pinns-repressilator

# Execute a lightweight test run for Experiment 5
python -u <<'PY'
from scripts.experiments import exp_regime_comparison as exp

exp.noise_levels = [0.05]
exp.seeds = [0]
exp.train_iterations = 100
exp.results_dir = "results/test_jobs/exp_regime_comparison"
exp.figure_path = "figures/test_jobs/exp_regime_comparison.png"

print("=== Running quick test: exp_regime_comparison ===")
print(f"regimes={exp.regimes}, noise_levels={exp.noise_levels}, seeds={exp.seeds}, train_iterations={exp.train_iterations}")
exp.main()
PY
