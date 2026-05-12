#!/bin/bash
#SBATCH --job-name=experiments_PINN      # Nom del job
#SBATCH --output=experiments_output.txt  # Fitxer de sortida
#SBATCH --error=experiments_error.txt    # Fitxer d'errors
#SBATCH --time=04:00:00                  # Temps maxim (hh:mm:ss)
#SBATCH --cpus-per-task=4                # Nombre de CPUs per tasca
#SBATCH --mem=8GB                        # Memoria assignada

# Working directory
cd $HOME/projects/pinns-repressilator

# Execute the Python script
python scripts/experiments/all_experiments.py