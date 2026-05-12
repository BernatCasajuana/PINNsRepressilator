"""
Automates the training of the PINN for all datasets stored in the 'datasets' folder.
Iterates over every .npz file, calling the `run_forward` function for each dataset and saving the corresponding results.
Enables running a full set of forward experiments for multiple parameter and noise combinations in one go.
"""

# %% Import necessary libraries
import os
import sys

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# %% Import the modular 'run_forward' function from run_forward.py
from scripts.pinns.forward import run_forward

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