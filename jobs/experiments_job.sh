!/bin/bash
--job-name = experiments.          # Nom del job
--output = experiments_output.txt  # Fitxer de sortida
--error = experiments_error.txt    # Fitxer d'errors
--time = 04:00:00                  # Temps màxim (hh:mm:ss)
--cpus-per-task = 4                # Nombre de CPUs per tasca
--mem = 8GB                        # Memòria assignada

# Working directory
cd $HOME/projects/pinns-repressilator

# Execute the Python script
python scripts/experiments/all_experiments.py