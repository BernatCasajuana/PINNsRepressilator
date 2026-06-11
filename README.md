# Empirical Characterization of Physics-Informed Neural Networks for Parameter Estimation on the Repressilator

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

Inside `jobs/`, the repository currently includes both full runs and quick validation runs:

- `jobs/experiments_job.sh`: runs all experiment drivers sequentially.
- `jobs/test_exp_*.sh`: lightweight smoke tests for each individual experiment.

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

All non-initial-guess experiment drivers use repeated seeds per configuration and report:

- relative error on $\beta$,
- relative error on $n$,
- an aggregate parameter recovery error,
- RMSE on the reconstructed trajectory.

The initial-guess experiment uses one seed by default and reports the same metrics on an initial-guess grid.

A dedicated script `scripts/experiments/all_experiments.py` runs all five experiments sequentially.

## Statistical Significance in Plots

For the non-initial-guess experiments, statistical comparisons in plots are computed and visualized as pairwise brackets with `p` value plus significance stars.

1. Test family: two-sided Mann-Whitney U test on per-seed metric samples.
2. Multiple testing: Holm-Bonferroni correction is applied within each panel.
3. Reported `p` in the figure labels: adjusted `p` values (not raw `p` values).
4. Star thresholds on adjusted `p`: `*` for `< 0.05`, `**` for `< 0.01`, `***` for `< 0.001`.
5. Baselines by experiment: noise sweep compares each noise level against `0.00`; sampling density compares each sparse setting against `100`; partial observation compares `x1,x2` and `x1` against `x1,x2,x3`; regime comparison pairs oscillatory vs stable at the same `beta`.

## Current Default Experimental Setup

The default configuration in the experiment scripts is currently:

- Experiment 1 (`exp_noise_sweep.py`): noise levels `0.00, 0.01, 0.05, 0.10, 0.20`; seeds `0, 1, 2, 3, 4`; `10000` training iterations per run.
- Experiment 2 (`exp_partial_observation.py`): observation designs `x1,x2,x3`, `x1,x2`, and `x1`; seeds `0, 1, 2, 3, 4`; `10000` training iterations per run.
- Experiment 3 (`exp_sampling_density.py`): observation counts `10, 25, 50, 100`; seeds `0, 1, 2, 3, 4`; `10000` training iterations per run.
- Experiment 4 (`exp_initial_guess.py`): initial-guess grid `beta0 in {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}` and `n0 in {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5}`; seed `0`; `10000` training iterations per run.
- Experiment 5 (`exp_regime_comparison.py`): four regime cases (`stable/oscillatory` crossed with `beta=5.0/8.0`, fixed `n` per regime), noise `0.05`, seeds `0, 1, 2, 3, 4`; `10000` training iterations per run.

## Current Default Compute Load

Using the defaults above, the total training-iteration budget is:

- Experiment 1: `5 x 5 x 10000 = 250000`
- Experiment 2: `3 x 5 x 10000 = 150000`
- Experiment 3: `4 x 5 x 10000 = 200000`
- Experiment 4: `7 x 7 x 1 x 10000 = 490000`
- Experiment 5: `4 x 5 x 10000 = 200000`

Total default budget across all experiments: `1290000` training iterations.

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

The reusable training code lives in `scripts/pinns/forward.py` and `scripts/pinns/inverse.py`, while the experiment drivers in `scripts/experiments/` call those functions and organize outputs under `results/` and `figures/`.
