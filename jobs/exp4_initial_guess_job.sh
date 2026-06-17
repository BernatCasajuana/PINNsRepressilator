#!/bin/bash
#SBATCH --job-name=exp4_initial_guess          # Nom del job
#SBATCH --output=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/exp4_initial_guess_output.txt
#SBATCH --error=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/exp4_initial_guess_error.txt
#SBATCH --time=24:00:00                       # Temps maxim (hh:mm:ss)
#SBATCH --cpus-per-task=1                     # Nombre de CPUs per tasca
#SBATCH --mem=8GB                             # Memoria assignada

source /opt/software/miniconda3/bin/activate pinns-repressilator-venv

cd /home/10040984@uvic.local/projects/pinns-repressilator
export PYTHONPATH=/home/10040984@uvic.local/projects/pinns-repressilator:$PYTHONPATH

python -u <<'PY'
from scripts.experiments import exp4_initial_guess as exp
print("=== Running exp4_initial_guess ===")
exp.main()
PY
