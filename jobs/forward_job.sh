#!/bin/bash
#SBATCH --job-name=forward_PINN         # Nom del job
#SBATCH --output=forward_output.txt     # Fitxer de sortida
#SBATCH --error=forward_error.txt       # Fitxer d’errors
#SBATCH --time=04:00:00                 # Temps màxim (hh:mm:ss)
#SBATCH --cpus-per-task=4               # Nombre de CPUs per tasca
#SBATCH --mem=8GB                       # Memòria assignada

# Working directory
cd $HOME/PINNsRepressilator

# Execute the Python script
python scripts/run_all_forward.py