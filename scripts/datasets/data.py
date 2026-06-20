"""
Repressilator ODE definition and dataset generation.

Defines the three-equation repressilator ODE (`protein_repressilator_rhs`) and
`generate_dataset`, which solves the system and saves results as a .npz file.
Run directly to regenerate the full 100-dataset grid (β × n × noise).
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

# %% Generate all datasets for the full parameter grid when run directly
if __name__ == "__main__":
    betas = [1.0, 5.0, 10.0, 20.0]
    ns = [1.5, 2.0, 2.5, 3.0, 3.5]
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    for beta in betas:
        for n in ns:
            for noise_sigma in noise_levels:
                generate_dataset(beta=beta, n=n, x0=[1, 1, 1.2], t_max=20, n_points=1000, noise_sigma=noise_sigma)