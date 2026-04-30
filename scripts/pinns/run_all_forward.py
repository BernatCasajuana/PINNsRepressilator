"""
run_all_forward.py

Automates the training of the PINN for all datasets stored in the 'datasets' folder.
Iterates over every .npz file, calling the `run_forward` function for each dataset and saving the corresponding results.
Enables running a full set of forward experiments for multiple parameter and noise combinations in one go.
"""

# %% Import necessary libraries
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# %% Import the modular 'run_forward' function from run_forward.py
from pinns.run_forward import run_forward

# %% Define dataset folder and results folder
dataset_folder = "datasets"
outdir_base = "results"

# %% Run for each file in the datasets folder
def main():
    for file in os.listdir(dataset_folder):
        if file.endswith(".npz"):
            dataset_path = os.path.join(dataset_folder, file)
            print(f"\n=== Running PINN for {dataset_path} ===")
            run_forward(dataset_path, loss_weights=None, outdir_base=outdir_base)


if __name__ == "__main__":
    main()