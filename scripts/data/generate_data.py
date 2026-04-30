"""
generate_data.py

Defines the Repressilator equations and generates datasets with simulated protein concentrations over time.
Noise can be added to emulate experimental conditions.
The main function, `generate_dataset`, solves the ODE system and saves the results as a .npz file.
An example block at the end shows how to manually generate a particulat dataset.
"""

# %% Import necessary libraries
import numpy as np
import scipy.integrate
import os

# %% Define ODE system
def protein_repressilator_rhs(x, t, beta, n):
    x1, x2, x3 = x
    return [
        beta / (1 + x3 ** n) - x1,
        beta / (1 + x1 ** n) - x2,
        beta / (1 + x2 ** n) - x3,
    ]

# %% Add noise to data
def add_noise(y, sigma):
    """Add Gaussian noise with sigma standard deviation."""
    return y + np.random.normal(0, sigma, y.shape)

# %% Generate dataset
def generate_dataset(beta, n, x0, t_max, n_points, noise_sigma=0.0, outdir="datasets"):
    os.makedirs(outdir, exist_ok=True)
# Time vector
    t = np.linspace(0, t_max, n_points)[:, None]
# Solve ODE
    y_clean = scipy.integrate.odeint(protein_repressilator_rhs, x0, t.flatten(), args=(beta, n))
# Add noise if specified
    if noise_sigma > 0:
        y_noisy = add_noise(y_clean, noise_sigma)
    else:
        y_noisy = y_clean
# Generate filename
    fname = f"beta{beta}_n{n}_noise{noise_sigma}.npz"
    fpath = os.path.join(outdir, fname)
# Save dataset as .npz file
    np.savez(fpath, t=t, y=y_noisy, y_clean=y_clean, beta=beta, n=n, noise=noise_sigma)
# Path to saved file
    print(f"Saved dataset: {fpath}")

# %% Custom combination, only executed when running this script directly
if __name__ == "__main__":
    # Example usage
    generate_dataset(beta=10, n=3.0, x0=[1, 1, 1.2], t_max=20, n_points=1000, noise_sigma=0.0)