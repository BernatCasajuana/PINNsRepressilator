"""
run_all_inverse.py

Automates the training of the PINN for all datasets stored in the 'datasets' folder.
Iterates over every .npz file, calling the `run_inverse` function for each dataset and saving the corresponding results.
Enables running a full set of inverse experiments for multiple parameter initial guesses and noise combinations in one go.
"""

# %% Import necessary libraries
import os

# %% Import the modular 'run_inverse' function from run_inverse.py
from run_inverse import run_inverse

# %% Define dataset folder and results folder
dataset_folder = "datasets"
outdir_base = "results"

# %% Define initial guesses per dataset (optional)
# Format: "filename.npz": (beta_guess, n_guess)
guesses = {
    "dataset1.npz": (5.0, 2.0),
    "dataset2.npz": (4.5, 2.1),
    "dataset3.npz": (6.0, 2.5),
    # Add more
}

# %% Run inverse PINN for each file in the datasets folder
for file in os.listdir(dataset_folder):
    if file.endswith(".npz"):
        dataset_path = os.path.join(dataset_folder, file)
        C1_guess, C2_guess = guesses.get(file, (5.0, 2.0))  # default guesses
        print(f"\n=== Running Inverse PINN for {dataset_path} with guesses C1={C1_guess}, C2={C2_guess} ===")
        run_inverse(dataset_path, outdir_base=outdir_base, C1_guess=C1_guess, C2_guess=C2_guess)