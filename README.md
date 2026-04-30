# PINNs for the Repressilator Under Varying Experimental Conditions

This repository studies the performance of Physics-Informed Neural Networks (PINNs) on the repressilator inverse problem. The paper focus is empirical: how reliably can PINNs recover the repressilator parameters and reconstruct the trajectories when the observation conditions become harder?

The project evaluates inverse-PINN performance under five experimental factors:

1. observation noise,
2. partial observation of the repressors,
3. sampling density over time,
4. sensitivity to the initial parameter guesses,
5. stable versus oscillatory dynamical regimes.

For all experiments, the main outputs are parameter recovery error and state reconstruction error. Where useful, the per-run training-loss curves are also saved in the results folders.

## Repository Organization

- `datasets/`: synthetic repressilator datasets as `.npz` files.
- `scripts/`: data generation, PINN training, and experiment driver scripts.
- `results/`: experiment outputs, including CSV summaries and trained model checkpoints.
- `figures/`: generated plots for the paper.
- `jobs/`: SLURM launch scripts.

## Core Scripts

- `scripts/generate_data.py`: generate a single synthetic repressilator dataset.
- `scripts/generate_all_data.py`: generate the dataset grid across parameter regimes and noise levels.
- `scripts/run_forward.py`: forward PINN training for state reconstruction with known parameters.
- `scripts/run_inverse.py`: inverse PINN training for estimation of $\beta$ and $n$.
- `scripts/run_all_forward.py`: batch execution of forward runs across the dataset folder.
- `scripts/run_all_inverse.py`: batch execution of inverse runs across the dataset folder.
- `scripts/check_formulation.py`: compare the ODE right-hand side with the TensorFlow formulation used by the PINN.

## Experiment Drivers

Each experiment driver runs one study, sweeps the relevant condition across seeds, writes CSV summaries under `results/`, and saves a paper figure under `figures/`.

- `scripts/exp_noise_sweep.py`: Experiment 1, sensitivity to observation noise.
- `scripts/exp_partial_observation.py`: Experiment 2, sensitivity to measuring fewer repressors.
- `scripts/exp_sampling_density.py`: Experiment 3, sensitivity to the number of observation points.
- `scripts/exp_initial_guess.py`: Experiment 4, sensitivity to the initial guesses for $\beta$ and $n$.
- `scripts/exp_regime_comparison.py`: Experiment 5, comparison between stable and oscillatory regimes.

All experiment drivers use repeated seeds per configuration and report:

- relative error on $\beta$,
- relative error on $n$,
- an aggregate parameter recovery error,
- RMSE on the reconstructed trajectory.

## Datasets

The datasets are stored as `.npz` files with names such as:

- `beta5.0_n3.0_noise0.1.npz`

Each dataset stores at least:

- `t`: time grid,
- `y`: noisy observations,
- `y_clean`: clean simulated trajectory,
- `beta`: true value of $\beta$,
- `n`: true value of the Hill coefficient,
- `noise`: observation noise level.

## Experimental Design

### Experiment 1: Noise sensitivity

Question: how does inverse-PINN recovery degrade as observation noise increases?

Design: all three repressors are observed, dense sampling is used, and the relative noise level is swept over `0.01, 0.05, 0.10, 0.20, 0.30`.

Output: a two-panel figure with parameter recovery error and state reconstruction error versus noise.

### Experiment 2: Partial observation

Question: how much performance is lost when fewer repressors are measured?

Design: noise is fixed and four observation designs are compared: all three repressors, `x1,x2`, `x1,x3`, and `x1` only.

Output: a grouped comparison of parameter and state errors across observation designs.

### Experiment 3: Sampling density

Question: how sparse can the measurements be before recovery fails?

Design: noise is fixed, all three repressors are observed, and the number of observation points is varied over `10, 25, 50, 100, 200`.

Output: parameter and state errors versus the number of observation points.

### Experiment 4: Initial guess sensitivity

Question: how sensitive is inverse-PINN training to the initial guesses for $\beta$ and $n$?

Design: the inverse problem is run over a grid of initial guesses for $\beta_0$ and $n_0`.

Output: heatmaps of the relative recovery error on $\beta$ and $n$ over the initial-guess grid.

### Experiment 5: Stable versus oscillatory regime

Question: does the dynamical regime change the difficulty of inverse-PINN recovery?

Design: a stable regime and an oscillatory regime are compared across multiple noise levels.

Output: a regime comparison figure for parameter recovery error and state reconstruction error.

## Dependencies

Dependencies are listed in `requirements.txt`. The main libraries are DeepXDE, TensorFlow, NumPy, SciPy, and Matplotlib.

Typical setup:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Notes on Execution

The repository does not use command-line argument parsing for the experiment drivers. Each script defines its configuration near the top of the file and can be run directly as a Python script.

The reusable training code lives in `run_forward.py` and `run_inverse.py`, while the experiment drivers call those functions and organize outputs under `results/` and `figures/`.

