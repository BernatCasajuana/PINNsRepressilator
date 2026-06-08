#!/bin/bash
#SBATCH --job-name=experiments           # Nom del job
#SBATCH --output=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/experiments_output.txt # Fitxer de sortida
#SBATCH --error=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/experiments_error.txt # Fitxer d'error
#SBATCH --time=04:00:00                  # Temps màxim (hh:mm:ss)
#SBATCH --cpus-per-task=12               # Nombre de CPUs per tasca
#SBATCH --mem=8GB                        # Memòria assignada

# Load conda module and activate environment
source /opt/software/miniconda3/bin/activate pinns-repressilator-venv

# Working directory
cd /home/10040984@uvic.local/projects/pinns-repressilator

# Set PYTHONPATH to include the current directory
export PYTHONPATH=/home/10040984@uvic.local/projects/pinns-repressilator:$PYTHONPATH

# Execute the Python script
python -m scripts.experiments.all_experiments