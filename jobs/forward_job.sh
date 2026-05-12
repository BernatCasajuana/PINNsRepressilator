!/bin/bash
--job-name = forward              # Nom del job
--output = forward_output.txt     # Fitxer de sortida
--error = forward_error.txt       # Fitxer d’errors
--time = 04:00:00                 # Temps màxim (hh:mm:ss)
--cpus-per-task = 4               # Nombre de CPUs per tasca
--mem = 8GB                       # Memòria assignada

# Working directory
cd $HOME/projects/pinns-repressilator

# Execute the Python script
python scripts/pinns/all_forward.py