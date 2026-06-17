#!/bin/bash
#SBATCH --job-name=exp1_noise_sweep            # Nom del job
#SBATCH --output=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/exp1_noise_sweep_output.txt
#SBATCH --error=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/exp1_noise_sweep_error.txt
#SBATCH --time=12:00:00                       # Temps maxim (hh:mm:ss)
#SBATCH --cpus-per-task=1                     # Nombre de CPUs per tasca
#SBATCH --mem=8GB                             # Memoria assignada

source /opt/software/miniconda3/bin/activate pinns-repressilator-venv

cd /home/10040984@uvic.local/projects/pinns-repressilator
export PYTHONPATH=/home/10040984@uvic.local/projects/pinns-repressilator:$PYTHONPATH

python -u <<'PY'
from scripts.experiments import exp1_noise_sweep as exp
print("=== Running exp1_noise_sweep ===")
exp.main()
PY
