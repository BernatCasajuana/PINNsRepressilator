# Physics-Informed Neural Networks for Parameter Recovery in the Repressilator

Companion code for the paper:

> **Physics-Informed Neural Networks for Parameter Recovery in the Repressilator Oscillatory Model**  
> B. Casajuana, R. Casals-Franch, A. López García de Lomana, P. Martí-Puig, J. Villà-Freixa  
> Universitat de Vic – Universitat Central de Catalunya (UVic-UCC)

---

## What this project does

The repressilator is a synthetic gene oscillator built from three cyclically repressing proteins. Its dynamics are governed by a three-equation ODE system with two key parameters: the maximal production rate β and the Hill coefficient n (Eq. 1 of the paper). Inferring these parameters from noisy, sparse, or partially observed time-series is a difficult inverse problem because the objective landscape is non-convex and the system is oscillatory.

This repository trains **inverse Physics-Informed Neural Networks (PINNs)** to recover β and n from synthetic protein-concentration time-series. Instead of calling an ODE solver repeatedly, a neural network represents the full trajectory and the ODE residual enters directly into the training loss:

```
L = λ_f · L_eq  +  λ_0 · L_IC  +  λ_y · L_obs
```

where `L_eq` penalises violations of the repressilator equations, `L_IC` enforces initial conditions, and `L_obs` fits sparse, possibly noisy observations. β and n are optimised jointly with the network weights.

Eight experiments characterise when and how this approach succeeds or fails, probing noise sensitivity, measurement design, optimisation landscape geometry, dynamical regime, loss weighting, and training dynamics.

---

## Repository layout

```
pinns-repressilator/
├── datasets/               # 100 pre-generated .npz datasets (β × n × noise grid)
├── scripts/
│   ├── data/               # ODE definition and dataset generation
│   │   ├── generate_data.py
│   │   └── generate_all_data.py
│   ├── pinns/              # PINN training modules (reusable)
│   │   ├── forward.py      # Forward problem: known β, n → predict trajectory
│   │   ├── inverse.py      # Inverse problem: estimate β, n from observations
│   │   ├── validate_ode.py # Sanity-check ODE formulation against scipy
│   │   ├── batch_forward.py
│   │   └── batch_inverse.py
│   ├── experiments/        # Experiment drivers (one file per experiment)
│   │   ├── experiment_utils.py         # Shared: seeding, noise model, statistics, plotting
│   │   ├── exp1_forward_vs_inverse.py  # Forward–inverse PINN performance gap
│   │   ├── exp2_noise_sweep.py         # Noise sensitivity (σ sweep)
│   │   ├── exp3_partial_observation.py # Partial observation (1/2/3 repressors)
│   │   ├── exp4_sampling_density.py    # Sampling density (10–100 points)
│   │   ├── exp5_initial_guess.py       # Initial guess sensitivity (7×7 grid)
│   │   ├── exp6_regime_comparison.py   # Stable vs oscillatory regime
│   │   ├── exp7_loss_weights.py        # Physics loss weight λ_f sensitivity
│   │   ├── exp8_convergence.py         # Convergence curves (β̂, n̂, losses)
│   │   └── all_experiments.py          # Run all eight sequentially
│   └── plots/              # Standalone visualisation scripts
│       └── plot_limit_cycle_3d.py  # 3D phase-space limit cycle
├── results/                # Output CSVs and per-run plots (generated at runtime)
├── figures/                # Summary figures (generated at runtime)
├── jobs/                   # SLURM scripts for cluster execution
│   ├── experiments_job.sh  # Full suite job
│   ├── forward_job.sh      # Single forward PINN job
│   ├── inverse_job.sh      # Single inverse PINN job
│   ├── test_job.sh         # General smoke test
│   └── test_exp_*.sh       # Per-experiment test jobs (one per experiment)
└── run_pilot.py            # Quick preview run with reduced iterations
```

---

## Quickstart

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Run a quick pilot to validate the pipeline (~5 min on CPU):**
```bash
python run_pilot.py
```

The pilot script accepts optional arguments to control the preview:
```bash
python run_pilot.py --train-iterations 500         # more converged preview
python run_pilot.py --only exp1_forward_vs_inverse exp5_initial_guess  # subset of experiments
```

**Run one full experiment:**
```bash
python scripts/experiments/exp2_noise_sweep.py
```

**Run all eight experiments (full budget, ~30–60 h on CPU):**
```bash
python scripts/experiments/all_experiments.py
```

**Standalone plot (no training needed):**
```bash
python scripts/plots/plot_limit_cycle_3d.py   # 3D limit cycle, β=8, n=3
```

---

## Datasets

Stored as `.npz` files under `datasets/`. File names encode the parameters, e.g. `beta5.0_n3.0_noise0.05.npz`.

The full grid: β ∈ {1.0, 5.0, 10.0, 20.0} × n ∈ {1.5, 2.0, 2.5, 3.0, 3.5} × noise ∈ {0.0, 0.01, 0.05, 0.1, 0.2} = 100 datasets.

Each file contains:

| Key | Description |
|---|---|
| `t` | Time grid, shape (1000, 1) |
| `y` | Noisy observations, shape (1000, 3) |
| `y_clean` | Clean ODE trajectory, shape (1000, 3) |
| `beta`, `n` | True parameter values |
| `noise` | Noise level (fraction of peak-to-peak amplitude) |

The experiment drivers generate datasets **in memory** using the same code path, so files in `datasets/` are not required to run the experiments.

---

## ODE model

```
dx₁/dt = β / (1 + x₃ⁿ) − x₁
dx₂/dt = β / (1 + x₁ⁿ) − x₂
dx₃/dt = β / (1 + x₂ⁿ) − x₃
```

Initial condition: (1.0, 1.0, 1.2). Integration span: t ∈ [0, 20], 1000 points.  
Main oscillatory setting: β = 5.0, n = 3.0. Stable comparison: β = 5.0, n = 1.5.

Noise is additive Gaussian: σ = noise_level × mean peak-to-peak signal amplitude.

---

## PINN architecture

All inverse PINNs use the same architecture (Section 2.3 of the paper):

- Input: scalar time t
- 5 hidden layers × 100 neurons, sinusoidal activation
- Output: (x̂₁, x̂₂, x̂₃) with softplus transform to enforce positivity
- Trainable scalars: β̂, n̂ (initial guesses configurable per experiment; defaults β₀ = 4.0, n₀ = 2.5)
- Optimizer: Adam, lr = 10⁻³

Most experiments use **10 000 iterations**. Experiments 1 and 7 use **5 000 iterations** (see Computational budget below).

---

## Experiments

Each experiment targets a specific axis of the inverse identification problem. All share the same canonical oscillatory setting (β = 5.0, n = 3.0, noise = 0.05) unless explicitly varied.

### Experiment 1 — Forward vs inverse PINN gap

The most fundamental comparison in the paper: what is the cost of not knowing the parameters? A *forward* PINN receives the true β and n and fits only the trajectory, while an *inverse* PINN must estimate them jointly from observations. By sweeping four noise levels (0%, 1%, 5%, 10%) and running both modes on the same data realisations, this experiment isolates the extra optimisation difficulty imposed by the parameter identification task. The figure shows state RMSE for both modes across noise levels as a single-panel line plot, making the identifiability cost directly visible.

### Experiment 2 — Noise sensitivity

How gracefully does the inverse PINN degrade as measurement noise increases? Noise levels from 0% to 20% of the peak-to-peak signal amplitude are tested under otherwise ideal conditions — all three repressors observed, dense sampling (1000 points). The four-panel (2×2) figure tracks β relative error, n relative error, combined parameter error, and state RMSE separately, with significance brackets relative to the noise-free baseline. The ODE residual provides a regularisation channel that is absent in purely data-driven fits, and this experiment tests whether it is effective enough to stabilise parameter accuracy under realistic noise.

### Experiment 3 — Partial observation

In many real experiments only a subset of molecular species can be measured. This experiment tests three measurement designs: all three repressors (3/3 — fully constrained), two repressors (2/3 — x₁ and x₂, with x₃ inferred from the ODE), and a single repressor (1/3 — x₁ alone, with x₂ and x₃ inferred). At fixed noise (5%) and dense sampling, the question is how much the physics constraint can compensate for missing measurements. The two-panel figure shows combined parameter error and state RMSE for each design. Results also feed into the joint observability figure produced by experiment 4.

### Experiment 4 — Sampling density

The repressilator is oscillatory, so the information content of a time-series is unevenly distributed — some phases are more constraining for n recovery than others. This experiment sweeps observation counts of 10, 25, 50, and 100 points (1%–10% of the full 1000-point grid), evenly spaced, to find where sparsity begins to degrade parameter recovery. In addition to its own two-panel figure, experiment 4 also produces a combined 2×2 observability figure (`exp3_4_observability.png`) that places the partial-observation (exp 3) and sampling-density results side by side, enabling a direct comparison of the two main axes of measurement design limitation.

### Experiment 5 — Initial guess sensitivity

Because the joint loss landscape over (β, n, network weights) is non-convex, the initial guesses for β and n can steer the optimiser into different attraction basins. This experiment runs a 7×7 grid of initial guesses — β₀ ∈ {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, n₀ ∈ {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5} — with one seed per cell (49 total runs at 10 000 iterations each). The true parameter location (β = 5.0, n = 3.0) is included in the grid. The figure is a heatmap of combined parameter error over the initial-guess plane, with a star at the true location and a dashed contour at the 10% error threshold. The shape of the reliable-recovery basin directly informs practical guidance on how to initialise inverse PINNs for the repressilator.

### Experiment 6 — Dynamical regime comparison

The Hill coefficient n controls whether the repressilator settles to a steady state (n ≈ 1.5) or sustains oscillations (n ≈ 3.0). These two regimes produce qualitatively different trajectory shapes, and it is not obvious a priori which is easier to identify from inverse PINN training. This experiment runs four cases — β ∈ {5.0, 8.0} crossed with regime ∈ {stable, oscillatory} — and compares both combined parameter error and state RMSE. The two-panel figure uses bars with individual seed overlays (strip plot style); each panel includes a small inset trajectory to give visual context for the dynamical behaviour. Significance brackets compare oscillatory vs stable at each β value.

### Experiment 7 — Physics loss weight sensitivity

The balance between the ODE residual term (λ_f · L_eq) and the observation term (λ_y · L_obs) is a central hyperparameter of the PINN formulation. Too small a λ_f and the physics constraint is effectively disabled; too large and the network ignores the observations and converges to any trajectory satisfying the ODE — not necessarily the one with the correct parameters. This experiment sweeps λ_f over five decades — {0.01, 0.1, 1.0, 10.0, 100.0} — while holding λ_0 = λ_y = 1.0 fixed. The three-panel (1×3) figure shows β and n error (overlaid), combined parameter error, and state RMSE, each with significance brackets relative to the λ_f = 1.0 baseline.

### Experiment 8 — Training convergence

This diagnostic experiment examines how the loss components and the parameter estimates evolve during training on the canonical condition (β = 5.0, n = 3.0, noise = 0.05). Three seeds are run for 10 000 iterations each, with the full loss history and parameter evolution logged every 100 steps. The 2×2 figure shows total loss (semilog), individual loss components L_eq, L_IC, and L_obs (semilog), β̂ convergence over iterations, and n̂ convergence over iterations — all seeds overlaid as transparent lines. This experiment serves as a diagnostic for understanding training dynamics and confirming that the standard 10 000-iteration budget is sufficient.

---

## Computational budget

| Experiment | Conditions | Seeds | Iterations | Total |
|---|---|---|---|---|
| 1 Forward vs inverse | 4 noise × fwd + inv | 3 | 5 000 | 120 000 |
| 2 Noise sweep | 5 noise levels | 5 | 10 000 | 250 000 |
| 3 Partial observation | 3 designs | 5 | 10 000 | 150 000 |
| 4 Sampling density | 4 counts | 5 | 10 000 | 200 000 |
| 5 Initial guesses | 7×7 grid | 1 | 10 000 | 490 000 |
| 6 Regime comparison | 4 cases | 5 | 10 000 | 200 000 |
| 7 Loss weights | 5 λ_f values | 5 | 5 000 | 125 000 |
| 8 Convergence | 1 condition | 3 | 10 000 | 30 000 |
| **Total** | | | | **1 565 000** |

Approximate wall-clock time: 30–60 h on CPU, 3–6 h on GPU.

---

## Experiment drivers

Each driver in `scripts/experiments/` follows the same pattern:

1. Define sweep configuration at the top of the file
2. For each condition × seed: call `run_inverse()` (or `run_forward()` for exp 1) with an in-memory dataset
3. Write per-run metrics to `results/<exp_name>/runs/`
4. Aggregate into `results/<exp_name>/<exp_name>_raw.csv` and `<exp_name>_summary.csv`
5. Save summary figure to `figures/<exp_name>.png`
6. Write `results/<exp_name>/run_manifest.json` (parameters + UTC timestamp)

### Statistical testing in figures

Pairwise comparisons use the two-sided Mann–Whitney U test (non-parametric, appropriate for small n = 5 seeds). Multiple testing is corrected with Holm–Bonferroni within each panel. Brackets show adjusted p-values; stars indicate `*` p < 0.05, `**` p < 0.01, `***` p < 0.001. Baseline comparisons:

- Noise sweep: each level vs σ = 0
- Sampling density: each count vs the densest condition
- Partial observation: each design vs full (3/3) observation
- Regime comparison: oscillatory vs stable at matching β
- Loss weight: each λ_f vs λ_f = 1 (baseline)

---

## Outputs

```
results/
└── <exp_name>/
    ├── run_manifest.json
    ├── <exp_name>_raw.csv       # one row per (condition, seed)
    ├── <exp_name>_summary.csv   # mean ± std per condition
    └── runs/
        └── <config>/
            └── seed-N/
                ├── inverse_metrics.csv
                ├── inverse_estimated_parameters.csv
                ├── inverse_loss.png
                └── inverse_prediction.png

figures/
├── exp1_forward_vs_inverse.png
├── exp2_noise_sweep.png
├── exp3_partial_observation.png
├── exp4_sampling_density.png
├── exp3_4_observability.png     # joint observability figure (produced by exp4)
├── exp5_initial_guess.png
├── exp6_regime_comparison.png
├── exp7_loss_weights.png
└── exp8_convergence.png
```

---

## Cluster execution

The `jobs/` directory contains SLURM scripts for running the experiments on a computing cluster. `experiments_job.sh` submits the full eight-experiment suite as a single job. `test_job.sh` runs a lightweight general smoke test, and `test_exp_<name>_job.sh` provides a dedicated test job for each individual experiment — useful for debugging or re-running a single experiment independently. `forward_job.sh` and `inverse_job.sh` target the underlying PINN modules directly for isolated testing.

All SLURM scripts set `PYTHONPATH` to the project root and activate the `pinns-repressilator-venv` conda environment.

---

## Dependencies

Key libraries: `deepxde==1.15.0`, `tensorflow==2.16.2`, `numpy`, `scipy`, `matplotlib`.  
Full list: `requirements.txt`. For cluster use: `requirements_cluster.txt`.

---

## Reproducibility

All random seeds are fixed per run; results are deterministic given the same hardware and library versions. The pilot script (`run_pilot.py`) is intended for pipeline validation and figure layout checks only — it uses a reduced iteration budget and results should not be interpreted as final. The full-budget configuration defined in each experiment driver is required to reproduce the paper figures and numbers.
