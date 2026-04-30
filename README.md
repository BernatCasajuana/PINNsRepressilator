# Empirical Characterization of Physics-Informed Neural Networks for Parameter Estimation on the Represilator

This project studies the reverse engineering performance of Physics-Informed Neural Networks (PINNs) in ODE-based models, using the repressilator, a synthetic gene regulatory network that exhibits oscillatory behavior, as a toy example.

The goal is to characterize how reliably PINNs can recover the system parameters and reconstruct the trajectories when the observation conditions become harder. 

Five experimental factors are considered:

1. observation noise,
2. partial observation of the repressors,
3. sampling density over time,
4. sensitivity to the initial parameter guesses,
5. stable versus oscillatory dynamical regimes.

For all five experiments, the main outputs are parameter recovery and state reconstruction errors.

## Repository Organization

- `datasets/`: synthetic datasets of repressors over time.
- `scripts/`: main Python scripts for data generation, PINN training, and experiment drivers.
- `results/`: experiment outputs, CSV summaries and trained model checkpoints.
- `figures/`: generated plots.
- `jobs/`: SLURM launch scripts for running the experiments on a cluster.

Inside `scripts/`, the code is organized into three main folders:

- `scripts/data/`: dataset generation scripts.
- `scripts/experiments/`: experiment setups and shared utilities.
- `scripts/pinns/`: PINN definitions and training for forward and inverse problems.

## Datasets

The datasets are stored as `.npz` files with names such as:

- `beta5.0_n3.0_noise0.1.npz`

Each dataset stores:

- `t`: time grid,
- `y`: noisy observations,
- `y_clean`: clean simulated trajectory,
- `beta`: true value of $\beta$,
- `n`: true value of the Hill coefficient,
- `noise`: observation noise level.

## Experiment Drivers

Each experiment script runs one study, sweeps the relevant condition across seeds, writes CSV summaries under `results/`, and saves the generated plot under `figures/`.

- `scripts/experiments/exp_noise_sweep.py`: Experiment 1, sensitivity to observation noise.
- `scripts/experiments/exp_partial_observation.py`: Experiment 2, sensitivity to partial repressor measurements.
- `scripts/experiments/exp_sampling_density.py`: Experiment 3, sensitivity to varying sampling density over time.
- `scripts/experiments/exp_initial_guess.py`: Experiment 4, sensitivity to initial parameter guesses.
- `scripts/experiments/exp_regime_comparison.py`: Experiment 5, comparison between stable and oscillatory regimes.

All experiment drivers use repeated seeds per configuration and report:

- relative error on $\beta$,
- relative error on $n$,
- an aggregate parameter recovery error,
- RMSE on the reconstructed trajectory.

## Dependencies

Dependencies are listed in `requirements.txt`. The main libraries are DeepXDE, TensorFlow, NumPy, SciPy, and Matplotlib.

Typical setup:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Notes on Execution

Each script defines its configuration near the top of the file and can be run directly as a Python script.

The reusable training code lives in `scripts/pinns/run_forward.py` and `scripts/pinns/run_inverse.py`, while the experiment drivers in `scripts/experiments/` call those functions and organize outputs under `results/` and `figures/`.

