#!/bin/bash
#SBATCH --job-name=exp5_regime_comparison      # Nom del job
#SBATCH --output=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/exp5_regime_comparison_output.txt
#SBATCH --error=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/exp5_regime_comparison_error.txt
#SBATCH --time=12:00:00                       # Temps maxim (hh:mm:ss)
#SBATCH --cpus-per-task=1                     # Nombre de CPUs per tasca
#SBATCH --mem=8GB                             # Memoria assignada

source /opt/software/miniconda3/bin/activate pinns-repressilator-venv

cd /home/10040984@uvic.local/projects/pinns-repressilator
export PYTHONPATH=/home/10040984@uvic.local/projects/pinns-repressilator:$PYTHONPATH

python -u <<'PY'
from scripts.experiments import exp5_regime_comparison as exp
print("=== Running exp5_regime_comparison ===")
exp.main()
PY
