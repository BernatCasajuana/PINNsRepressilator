# Physics-Informed Neural Networks for Parameter Recovery in the Repressilator

Companion code for the paper:

> **Physics-Informed Neural Networks for Parameter Recovery in the Repressilator Oscillatory Model**  
> B. Casajuana, R. Casals-Franch, A. López García de Lomana, P. Martí-Puig, J. Villà-Freixa  
> Universitat de Vic – Universitat Central de Catalunya (UVic-UCC)

---

## What this project does

The repressilator is a synthetic gene oscillator built from three cyclically repressing proteins. Its dynamics are governed by a three-equation ODE system with two parameters: the maximal production rate β and the Hill coefficient n (Eq. 1 of the paper). Inferring these parameters from noisy, sparse, or partially observed time-series is a difficult inverse problem because the objective landscape is non-convex and the system is oscillatory.

This repository trains **inverse Physics-Informed Neural Networks (PINNs)** to recover β and n from synthetic protein-concentration time-series. Instead of calling an ODE solver repeatedly, a neural network represents the full trajectory and the ODE residual enters directly into the training loss:

```
L = λ_f · L_eq  +  λ_0 · L_IC  +  λ_y · L_obs
```

where `L_eq` penalises violations of the repressilator equations, `L_IC` enforces initial conditions, and `L_obs` fits sparse, possibly noisy observations. β and n are optimised jointly with the network weights.

Eight experiments characterise when and how this approach succeeds or fails:

| # | Experiment | Factor varied | Figure layout | Key finding |
|---|---|---|---|---|
| 1 | Forward vs inverse | RMSE with/without parameter estimation | 1-panel: state RMSE line plot | Low RMSE does not guarantee accurate parameter recovery |
| 2 | Noise sweep | Relative observation noise (0%–20%) | 2×2: β, n, combined, state RMSE | Parameter error is stable; trajectory RMSE degrades faster |
| 3 | Partial observation | Observed repressors (1/3, 2/3, 3/3) | 2-panel: combined + state RMSE | Losing phase-coupled repressors inflates variance sharply |
| 4 | Sampling density | Observation count (10–1000 points) | 2-panel + combined exp3_4 figure | Sparse sampling removes informative oscillatory phases |
| 5 | Initial guesses | Starting (β₀, n₀) on 7×7 grid | 1-panel: heatmap | Non-convex landscape reveals failure basins |
| 6 | Regime comparison | Stable (n=1.5) vs oscillatory (n=3.0) | 2-panel: bar + strip, inset trajs | Oscillatory is harder to reconstruct but more informative |
| 7 | Loss weight λ_f | Physics loss weight {0.01, 0.1, 1, 10, 100} | 3-panel: β+n, combined, state | λ_f ≈ 1 balances physics and data constraints |
| 8 | Convergence | Training curves over 10000 iterations | 2×2: loss, components, β̂, n̂ | β̂ converges faster than n̂; 10k iters is typically sufficient |

The central finding is that **a low trajectory RMSE does not guarantee accurate parameter recovery**. PINNs reconstruct trajectories reliably but parameter identification is more fragile — a distinction that matters for reverse-engineering biological circuits.

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
│   │   ├── exp1_forward_vs_inverse.py  # ★ Core finding: forward–inverse PINN gap
│   │   ├── exp2_noise_sweep.py         # Noise sensitivity (σ sweep)
│   │   ├── exp3_partial_observation.py # Partial observation (1/2/3 repressors)
│   │   ├── exp4_sampling_density.py    # Sampling density (10–100 points)
│   │   ├── exp5_initial_guess.py       # Initial guess sensitivity (7×7 grid)
│   │   ├── exp6_regime_comparison.py   # Stable vs oscillatory regime
│   │   ├── exp7_loss_weights.py        # Physics loss weight λ_f sensitivity
│   │   ├── exp8_convergence.py         # Convergence curves (β̂, n̂, losses)
│   │   └── all_experiments.py          # Run all eight sequentially
│   └── plots/              # Standalone visualisation scripts
│       ├── plot_limit_cycle_3d.py  # 3D phase-space limit cycle
│       └── plot_pinn_schematic.py  # PINN architecture diagram
├── results/                # Output CSVs and per-run plots (generated at runtime)
├── figures/                # Summary figures (generated at runtime)
├── jobs/                   # SLURM scripts for cluster execution
│   ├── experiments_job.sh
│   ├── smoke_test.sh
│   └── pilot_exp*.sh
├── run_pilot.py            # Quick preview run with reduced iterations
└── requirements.txt
```

---

## Quickstart

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Run a single experiment (fast preview, ~5 min on CPU):**
```bash
python run_pilot.py
```

**Run one full experiment:**
```bash
cd scripts/experiments
python exp2_noise_sweep.py
```

**Run all eight experiments (full budget, ~30–60 h on CPU):**
```bash
python scripts/experiments/all_experiments.py
```

**Standalone plots (no training needed):**
```bash
python scripts/plots/plot_limit_cycle_3d.py   # 3D limit cycle, β=8, n=3
python scripts/plots/plot_pinn_schematic.py   # PINN architecture diagram
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
- Trainable scalars: β̂, n̂ (initial guesses configurable per experiment)
- Optimizer: Adam, lr = 10⁻³

Experiments 1–2 and 5 use **5 000 iterations**. Experiments 3–4 use **3 000 iterations** (per the paper's Table 1).

---

## Experiment drivers

Each driver in `scripts/experiments/` follows the same pattern:

1. Define sweep configuration at the top of the file
2. For each condition × seed: call `run_inverse()` with an in-memory dataset
3. Write per-run metrics to `results/<exp_name>/runs/`
4. Aggregate into `results/<exp_name>/<exp_name>_raw.csv` and `<exp_name>_summary.csv`
5. Save summary figure to `figures/<exp_name>.png`
6. Write `results/<exp_name>/run_manifest.json` (parameters + UTC timestamp)

### Statistical testing in figures

Pairwise comparisons use the two-sided Mann–Whitney U test (non-parametric, appropriate for small n=5 seeds). Multiple testing is corrected with Holm–Bonferroni within each panel. Brackets show adjusted p-values; stars indicate `*` p<0.05, `**` p<0.01, `***` p<0.001. Baseline comparisons:

- Noise sweep: each level vs σ=0
- Sampling density: each count vs the densest condition
- Partial observation: each design vs full (3/3) observation
- Regime comparison: oscillatory vs stable at matching β
- Loss weight: each λ_f vs λ_f=1 (baseline)

The combined exp3_4_observability figure (produced by exp4) places partial-observation and
sampling-density results in a 2×2 grid (rows = Param. Recovery / State Recons., columns = experiment).

---

## Computational budget

Default configuration (full runs):

| Experiment | Conditions | Seeds | Iterations | Total |
|---|---|---|---|---|
| 1 Forward vs inverse | 4 noise × fwd+inv | 3 | 5 000 | 120 000 |
| 2 Noise sweep | 5 noise levels | 5 | 10 000 | 250 000 |
| 3 Partial observation | 3 designs | 5 | 10 000 | 150 000 |
| 4 Sampling density | 4 counts | 5 | 10 000 | 200 000 |
| 5 Initial guesses | 7×7 grid | 1 | 10 000 | 490 000 |
| 6 Regime comparison | 4 cases | 5 | 10 000 | 200 000 |
| 7 Loss weight λ_f | 5 λ_f values | 5 | 5 000 | 125 000 |
| 8 Convergence | 2 conditions | 3 | 10 000 | 60 000 |
| **Total** | | | | **1 595 000** |

Approximate wall-clock time: 30–60 h on CPU, 3–6 h on GPU.

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
├── exp2_noise_sweep.png
├── exp3_partial_observation.png
├── exp4_sampling_density.png
├── exp5_initial_guess.png
├── exp6_regime_comparison.png
├── exp1_forward_vs_inverse.png
├── exp7_loss_weights.png
└── exp8_convergence.png
```

---

## Dependencies

Key libraries: `deepxde==1.15.0`, `tensorflow==2.16.2`, `numpy`, `scipy`, `matplotlib`.  
Full list: `requirements.txt`. For cluster use: `requirements_cluster.txt`.

---

## Status of results

> **Current results are pilot / preview runs** (reduced iterations, seeds 0–1). They are used to validate the pipeline and check figure layout — they are **not** the final results reported in the paper. The full-budget runs (5 seeds × 10 000 iterations per condition) are required to reproduce the paper figures and numbers.

```bash
# Quick pilot run to validate the pipeline (~5 min):
python run_pilot.py

# Full experiment suite to reproduce paper results (~20–40 h on CPU):
python scripts/data/generate_all_data.py      # generate datasets
python scripts/experiments/all_experiments.py  # run all five experiments

# Standalone figures (no training required):
python scripts/plots/plot_limit_cycle_3d.py
python scripts/plots/plot_pinn_schematic.py
```

All random seeds are fixed per run; results are deterministic given the same hardware and library versions.
