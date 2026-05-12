!/bin/bash
--job-name = test                 # Nom del job
--output = test_output.txt        # Fitxer de sortida
--error = test_error.txt          # Fitxer d'errors
--time = 01:00:00                 # Temps maxim (hh:mm:ss)
--cpus-per-task = 4               # Nombre de CPUs per tasca
--mem = 8GB                       # Memoria assignada

# Working directory
cd $HOME/projects/pinns-repressilator

# Optional virtual environment
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

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