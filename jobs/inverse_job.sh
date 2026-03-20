#!/bin/bash
#SBATCH --job-name=inverse_PINN         # Nom del job
#SBATCH --output=inverse_output.txt     # Fitxer de sortida
#SBATCH --error=inverse_error.txt       # Fitxer d’errors
#SBATCH --time=04:00:00                 # Temps màxim (hh:mm:ss)
#SBATCH --cpus-per-task=4               # Nombre de CPUs per tasca
#SBATCH --mem=8GB                       # Memòria assignada

# Working directory
cd $HOME/PINNsRepressilator

# Execute the Python script
python scripts/run_all_inverse.py