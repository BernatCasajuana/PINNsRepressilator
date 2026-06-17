#!/bin/bash
#SBATCH --job-name=pilot_all_exps             # Nom del job
#SBATCH --output=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/pilot_output.txt
#SBATCH --error=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/pilot_error.txt
#SBATCH --time=01:00:00                       # Temps maxim (hh:mm:ss)
#SBATCH --cpus-per-task=1                     # Nombre de CPUs per tasca
#SBATCH --mem=8GB                             # Memoria assignada

source /opt/software/miniconda3/bin/activate pinns-repressilator-venv

cd /home/10040984@uvic.local/projects/pinns-repressilator
export PYTHONPATH=/home/10040984@uvic.local/projects/pinns-repressilator:$PYTHONPATH

ROOT="/home/10040984@uvic.local/projects/pinns-repressilator"

echo "=== PILOT: 10 iterations, all seeds ==="

python -u <<PY
from scripts.experiments import exp1_noise_sweep as exp
exp.train_iterations = 10
exp.results_dir = "$ROOT/results/pilot_exp1_noise_sweep"
exp.figure_path  = "$ROOT/figures/pilot_exp1_noise_sweep.png"
print("--- exp1_noise_sweep ---")
exp.main()
PY

python -u <<PY
from scripts.experiments import exp2_partial_observation as exp
exp.train_iterations = 10
exp.results_dir = "$ROOT/results/pilot_exp2_partial_observation"
exp.table_path  = "$ROOT/tables/pilot_exp2_partial_observation.tex"
print("--- exp2_partial_observation ---")
exp.main()
PY

python -u <<PY
from scripts.experiments import exp3_sampling_density as exp
exp.train_iterations         = 10
exp.results_dir              = "$ROOT/results/pilot_exp3_sampling_density"
exp.table_path               = "$ROOT/tables/pilot_exp3_sampling_density.tex"
exp.observability_table_path = "$ROOT/tables/pilot_observability.tex"
exp.exp2_results_dir         = "$ROOT/results/pilot_exp2_partial_observation"
print("--- exp3_sampling_density ---")
exp.main()
PY

python -u <<PY
from scripts.experiments import exp4_initial_guess as exp
exp.train_iterations = 10
exp.results_dir = "$ROOT/results/pilot_exp4_initial_guess"
exp.figure_path  = "$ROOT/figures/pilot_exp4_initial_guess.png"
print("--- exp4_initial_guess ---")
exp.main()
PY

python -u <<PY
from scripts.experiments import exp5_regime_comparison as exp
exp.train_iterations = 10
exp.results_dir = "$ROOT/results/pilot_exp5_regime_comparison"
exp.table_path  = "$ROOT/tables/pilot_exp5_regime_comparison.tex"
print("--- exp5_regime_comparison ---")
exp.main()
PY

python -u <<PY
from scripts.experiments import exp6_loss_weights as exp
exp.train_iterations = 10
exp.results_dir = "$ROOT/results/pilot_exp6_loss_weights"
exp.table_path  = "$ROOT/tables/pilot_exp6_loss_weights.tex"
print("--- exp6_loss_weights ---")
exp.main()
PY

python -u <<PY
from scripts.experiments import exp7_convergence as exp
exp.train_iterations = 10
exp.results_dir = "$ROOT/results/pilot_exp7_convergence"
exp.figure_path  = "$ROOT/figures/pilot_exp7_convergence.png"
print("--- exp7_convergence ---")
exp.main()
PY

echo "=== PILOT COMPLETE ==="
