#!/bin/bash
#SBATCH --job-name=test                 # Nom del job
#SBATCH --output=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/test_output.txt # Fitxer de sortida
#SBATCH --error=/home/10040984@uvic.local/projects/pinns-repressilator/jobs/test_error.txt # Fitxer d'error
#SBATCH --time=01:00:00                 # Temps maxim (hh:mm:ss)
#SBATCH --cpus-per-task=4               # Nombre de CPUs per tasca
#SBATCH --mem=8GB                       # Memoria assignada

# Load conda module and activate environment
source /opt/software/miniconda3/bin/activate pinns-repressilator-venv

# Working directory
cd /home/10040984@uvic.local/projects/pinns-repressilator

# Set PYTHONPATH to include the current directory
export PYTHONPATH=/home/10040984@uvic.local/projects/pinns-repressilator:$PYTHONPATH

# Run one lightweight forward and one lightweight inverse job on a single dataset
python -u <<'PY'
from scripts.pinns.forward import run_forward
from scripts.pinns.inverse import run_inverse

dataset_path = "datasets/beta5.0_n3.0_noise0.05.npz"

print(f"=== Running forward test on {dataset_path} ===")
forward_result = run_forward(
    dataset_path,
    outdir_base = "results/test_cluster/forward",
    observation_stride = 10,
    adam_epochs = 500,
    run_lbfgs = False,
)
print(f"Forward results saved in {forward_result['outdir']}")

print(f"=== Running inverse test on {dataset_path} ===")
inverse_result = run_inverse(
    dataset_path,
    outdir_base = "results/test_cluster/inverse",
    C1_guess = 4.0,
    C2_guess = 2.5,
    observation_stride = 10,
    observed_components = [0, 1, 2],
    train_iterations = 1000,
    random_seed = 0,
    save_checkpoint = False,
)
print(f"Inverse results saved in {inverse_result['outdir']}")
PY