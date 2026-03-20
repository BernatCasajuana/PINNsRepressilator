"""
generate_all_data.py

Automates the generation of multiple Repressilator datasets for different combinations of parameters and noise levels.
Imports the `generate_dataset` function from generate_data.py and iterates over all scenarios to create .npz files.
All datasets are saved in the 'datasets' folder for later use in the PINN training.
"""

# %% Import the modular 'generate_dataset' function from generate_data.py
from generate_data import generate_dataset

# %% Define parameters and noise levels
betas = [1.0, 5.0, 10.0, 20.0]
ns = [2.0, 2.5, 3.0, 3.5]
noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
x0 = [1, 1, 1.2]
t_max = 20
n_points = 1000

# %% Generate datasets for all combinations
for beta in betas:
    for n in ns:
        for noise_sigma in noise_levels:
            print(f"Generating dataset: beta={beta}, n={n}, noise={noise_sigma}")
            generate_dataset(beta=beta, n=n, x0=x0, t_max=t_max, n_points=n_points, noise_sigma=noise_sigma)