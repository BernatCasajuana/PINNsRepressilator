#!/bin/bash
#SBATCH --job-name=inverse              # Nom del job
#SBATCH --output=inverse_output.txt     # Fitxer de sortida
#SBATCH --error=inverse_error.txt       # Fitxer d’errors
#SBATCH --time=04:00:00                 # Temps màxim (hh:mm:ss)
#SBATCH --cpus-per-task=4               # Nombre de CPUs per tasca
#SBATCH --mem=8GB                       # Memòria assignada

# Load conda module and activate environment
module load conda
conda activate pinnsvenv

# Working directory
cd ~/projects/pinns-repressilator

# Execute the Python script
python scripts/pinns/all_inverse.py