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

Seven experiments characterise when and how this approach succeeds or fails, probing noise sensitivity (with a direct comparison against a classical ODE fitting baseline), measurement design, sampling density, optimisation landscape geometry, dynamical regime, loss weighting, and training dynamics.

---

## Repository layout

```
pinns-repressilator/
├── datasets/               # 100 pre-generated .npz datasets (β × n × noise grid)
├── scripts/
│   ├── data/               # ODE definition and dataset generation
│   │   └── data.py           # ODE, generate_dataset(); run directly to regenerate datasets/
│   ├── pinns/              # PINN training modules
│   │   ├── forward.py      # Forward problem: known β, n → predict trajectory
│   │   └── inverse.py      # Inverse problem: estimate β, n from observations
│   └── experiments/        # Experiment drivers (one file per experiment)
│       ├── experiment_utils.py         # Shared: seeding, noise model, statistics, plotting
│       ├── exp1_noise_sweep.py         # Noise sensitivity (σ sweep) + PINN vs classical baseline
│       ├── exp2_partial_observation.py # Partial observation (1/2/3 repressors) → table
│       ├── exp3_sampling_density.py    # Sampling density (10–100 points) → table + observability.tex
│       ├── exp4_initial_guess.py       # Initial guess sensitivity (7×7 grid)
│       ├── exp5_regime_comparison.py   # Stable vs oscillatory regime → table
│       ├── exp6_loss_weights.py        # Physics loss weight λ_f sensitivity → table
│       ├── exp7_convergence.py         # Convergence curves (β̂, n̂, losses)
│       └── run_pilot.py                # Quick preview run with reduced iterations (local use)
├── results/                # Output CSVs and per-run metrics (generated at runtime)
├── figures/                # Summary figures (generated at runtime)
├── tables/                 # LaTeX tables (generated at runtime)
└── jobs/                   # SLURM scripts for cluster execution
    ├── exp1_noise_sweep_job.sh
    ├── exp2_partial_observation_job.sh
    ├── exp3_sampling_density_job.sh
    ├── exp4_initial_guess_job.sh
    ├── exp5_regime_comparison_job.sh
    ├── exp6_loss_weights_job.sh
    ├── exp7_convergence_job.sh
    ├── pilot_job.sh        # 10-iteration sanity check across all experiments
    └── test_job.sh         # Cluster environment check (imports, TF, GPU)
```

---

## Quickstart

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_mac.txt        # macOS
# pip install -r requirements_linux.txt  # Linux / cluster
```

**Run a quick pilot to validate the pipeline locally (~5 min on CPU):**
```bash
python scripts/experiments/run_pilot.py
```

The pilot script accepts optional arguments:
```bash
python scripts/experiments/run_pilot.py --train-iterations 500         # more converged preview
python scripts/experiments/run_pilot.py --seeds 1                      # fastest check (1 seed)
python scripts/experiments/run_pilot.py --only exp1_noise_sweep exp4_initial_guess
```

**Run one full experiment locally:**
```bash
python scripts/experiments/exp1_noise_sweep.py
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

All experiments use **10 000 iterations**.

---

## Experiments

Each experiment targets a specific axis of the inverse identification problem. All share the same canonical oscillatory setting (β = 5.0, n = 3.0, noise = 0.05) unless explicitly varied.

### Experiment 1 — Noise sensitivity and baseline comparison

How gracefully does the inverse PINN degrade as measurement noise increases, and does it outperform classical ODE fitting? Noise levels from 0% to 20% of the peak-to-peak signal amplitude are tested under otherwise ideal conditions — all three repressors observed, dense sampling (1000 points). For each (noise, seed) pair, a classical baseline is also run: `scipy.optimize.minimize` (L-BFGS-B) + `scipy.integrate.odeint`, minimising observation MSE from the same initial guess (β₀ = 4.0, n₀ = 2.5). The 4-panel (2×2) figure shows β error (panel A) and n error (panel B) for the PINN only with significance brackets vs σ = 0; panels C and D overlay both PINN (blue, solid) and L-BFGS-B (orange, dashed) for combined parameter error and state RMSE, titled "Parameter Recovery — PINN vs L-BFGS-B" and "Trajectory Reconstruction — PINN vs L-BFGS-B". The ODE residual provides a regularisation channel absent in the classical fit, and this is where that advantage is quantified.

### Experiment 2 — Partial observation

In many real experiments only a subset of molecular species can be measured. This experiment tests three designs: all three repressors (3/3), two repressors (2/3 — x₃ unobserved), and one repressor (1/3 — x₂ and x₃ unobserved). For each (design, seed), both the PINN and an L-BFGS-B baseline are run on the same data. Both methods use the full three-equation ODE: L-BFGS-B integrates it exactly (so unobserved species are always implicitly constrained through the ODE coupling given β and n), while the PINN enforces it approximately via a residual penalty. The key question is therefore not whether the PINN "knows" more physics about unobserved species, but whether the neural-network parameterisation offers a smoother optimisation landscape than direct parameter search. Physics regularisation pays off precisely when data is incomplete — but if L-BFGS-B matches PINN in the 1/3 setting, exact ODE integration is sufficient. Results are a **LaTeX table** (`tables/exp2_partial_observation.tex`) showing PINN and L-BFGS-B side by side with Mann–Whitney p-values comparing the two methods at each design.

### Experiment 3 — Sampling density

How sparse can the time-series be before parameter recovery degrades, and does the PINN's physics-guided interpolation provide an advantage over classical fitting? Observation counts of 10, 25, 50, and 100 points (1%–10% of the 1000-point grid, evenly spaced) are tested with all three repressors observed and 5% noise. For each (count, seed), both the PINN and an L-BFGS-B baseline are run on identical data. L-BFGS-B integrates the full ODE trajectory but computes MSE only at the sparse observed time points; the PINN enforces the ODE residual at dense collocation points across the full domain regardless of observation density. Physics regularisation pays off precisely when data is incomplete — and unlike the partial-observation case, this is a genuine structural asymmetry: L-BFGS-B receives gradient signal only at observed times, while the PINN's collocation residual fills in physics everywhere between observations. If L-BFGS-B matches PINN, exact integration over the sparse subset is sufficient. Results are a **LaTeX table** (`tables/exp3_sampling_density.tex`). Running experiments 2 and 3 in sequence also produces a **combined observability table** (`tables/observability.tex`) covering both measurement design dimensions side by side.

### Experiment 4 — Initial guess sensitivity

Because the joint loss landscape over (β, n, network weights) is non-convex, the initial guesses for β and n can steer the optimiser into different local optima. This experiment runs a 7×7 grid of initial guesses — β₀ ∈ {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, n₀ ∈ {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5} — with one seed per cell (49 total runs at 10 000 iterations each). The true parameter location (β = 5.0, n = 3.0) is included in the grid and marked in the figure. The figure is a heatmap of combined parameter error over the initial-guess plane, showing which starting points reliably converge to the true parameters and which trap the optimiser in a poor solution.

### Experiment 5 — Dynamical regime comparison

The Hill coefficient n controls whether the repressilator settles to a steady state (n ≈ 1.5) or sustains oscillations (n ≈ 3.0). These two regimes produce qualitatively different trajectory shapes, and it is not obvious a priori which is easier to identify from inverse PINN training. This experiment runs four cases — β ∈ {5.0, 8.0} crossed with regime ∈ {stable, oscillatory} — and reports β error, n error, and state RMSE separately. Results are presented as a **LaTeX table** (`tables/exp5_regime_comparison.tex`) with β embedded in the condition label (e.g. "Stable (β=5)") and Holm–Bonferroni-adjusted p-values for β and n errors separately, comparing stable vs oscillatory at each β value.

### Experiment 6 — Physics loss weight sensitivity

The balance between the ODE residual term (λ_f · L_eq) and the observation term (λ_y · L_obs) is a central hyperparameter of the PINN formulation. Too small a λ_f and the physics constraint is effectively disabled; too large and the network ignores the observations and converges to any trajectory satisfying the ODE — not necessarily the one with the correct parameters. This experiment sweeps λ_f over five decades — {0.01, 0.1, 1.0, 10.0, 100.0} — while holding λ_0 = λ_y = 1.0 fixed. Results are presented as a **LaTeX table** (`tables/exp6_loss_weights.tex`) with β error, n error, and state RMSE (mean ± SD) at each λ_f, and separate Mann–Whitney p-values for β and n comparing each value to the λ_f = 1.0 baseline.

### Experiment 7 — Training convergence

This diagnostic experiment examines how the loss components and parameter estimates evolve during training on the canonical condition (β = 5.0, n = 3.0, noise = 0.05). The 2×2 figure shows: total loss (semilog, 5 seeds overlaid), individual loss components L_eq, L_IC, and L_obs (semilog, 5 seeds overlaid), β̂ convergence for seven different β₀ initial guesses (fixing n₀ = 2.5), and n̂ convergence for seven different n₀ initial guesses (fixing β₀ = 4.0). Panels C/D use the same initial-guess grid as experiment 4, revealing which starting points converge reliably vs stall or diverge.

---

## Computational budget

| Experiment | Conditions | Seeds | Iterations | Total |
|---|---|---|---|---|
| 1 Noise sweep + classical | 5 noise levels | 5 | 10 000 | 250 000 |
| 2 Partial observation | 3 designs | 5 | 10 000 | 150 000 |
| 3 Sampling density | 4 counts | 5 | 10 000 | 200 000 |
| 4 Initial guesses | 7×7 grid | 1 | 10 000 | 490 000 |
| 5 Regime comparison | 4 cases | 5 | 10 000 | 200 000 |
| 6 Loss weights | 5 λ_f values | 5 | 10 000 | 250 000 |
| 7 Convergence | 1 cond. + 14 init. guesses | 5 / 1 | 10 000 | 190 000 |
| **Total** | | | | **1 730 000** |

Approximate wall-clock time: 25–55 h on CPU, 2–5 h on GPU.  
L-BFGS-B fitting in exp 1, 2, and 3 adds negligible compute (seconds per fit).

---

## Experiment drivers

Each driver in `scripts/experiments/` follows the same pattern:

1. Define sweep configuration at the top of the file
2. For each condition × seed: call `run_inverse()` with an in-memory dataset
3. Write per-run metrics to `results/<exp_name>/runs/`
4. Aggregate into `results/<exp_name>/<exp_name>_raw.csv` and `<exp_name>_summary.csv`
5. Save output: a figure to `figures/<exp_name>.png` (exp 1, 4, 7) or a LaTeX table to `tables/<exp_name>.tex` (exp 2, 3, 5, 6); exp 3 additionally writes a combined `tables/observability.tex` merging experiments 2 and 3
6. Write `results/<exp_name>/run_manifest.json` (parameters + UTC timestamp)

### Statistical testing

Pairwise comparisons use the two-sided Mann–Whitney U test (non-parametric, appropriate for small n = 5 seeds). Multiple testing is corrected with Holm–Bonferroni within each panel. Brackets/table cells show adjusted p-values; stars indicate `*` p < 0.05, `**` p < 0.01, `***` p < 0.001. Baseline comparisons:

- Noise sweep (exp 1): each noise level vs σ = 0 (PINN panels A/B); PINN vs L-BFGS-B visually in panels C/D
- Partial observation (exp 2): PINN vs L-BFGS-B at each observation design (Mann–Whitney U)
- Sampling density (exp 3): PINN vs L-BFGS-B at each observation count (Mann–Whitney U)
- Regime comparison (exp 5): oscillatory vs stable at matching β
- Loss weight (exp 6): each λ_f vs λ_f = 1 (baseline)

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
├── exp1_noise_sweep.png
├── exp4_initial_guess.png
└── exp7_convergence.png

tables/
├── observability.tex            # paper table: measurement design (exp2 + exp3 combined)
├── exp5_regime_comparison.tex   # paper table: dynamical regime
├── exp6_loss_weights.tex        # paper table: physics loss weight sensitivity
├── exp2_partial_observation.tex # intermediate: partial observation detail
└── exp3_sampling_density.tex    # intermediate: sampling density detail
```

All `.tex` table files use the `booktabs` package (`\toprule`, `\midrule`, `\bottomrule`). Add `\usepackage{booktabs}` to your LaTeX preamble.

---

## Cluster execution

The `jobs/` directory contains SLURM scripts for the cluster. Each experiment has its own job script (`exp{N}_<name>_job.sh`) configured for 12 h wall time and 8 GB memory. Submit each independently so they run in parallel:

```bash
sbatch jobs/exp1_noise_sweep_job.sh
sbatch jobs/exp2_partial_observation_job.sh
# ... etc.
```

**Before submitting full jobs**, run the pilot to verify the environment end-to-end (10 iterations across all experiments, ~10 min):

```bash
sbatch jobs/pilot_job.sh
```

**To check the cluster environment only** (imports, TensorFlow, GPU detection — under 1 min):

```bash
sbatch jobs/test_job.sh
```

All job scripts write stdout/stderr to `jobs/exp{N}_<name>_output.txt` / `_error.txt` on the cluster. Figures, tables, and results land in `figures/`, `tables/`, and `results/` under the project root on the cluster; sync them back with `rsync` or `scp`.

All SLURM scripts activate the `pinns-repressilator-venv` conda environment and set `PYTHONPATH` to the project root.

---

## Dependencies

Key libraries: `deepxde==1.15.0`, `tensorflow==2.16.2`, `numpy`, `scipy`, `matplotlib`.  
Full macOS list: `requirements_mac.txt`. For cluster (Linux): `requirements_linux.txt`.

---

## Reproducibility

All random seeds are fixed per run; results are deterministic given the same hardware and library versions. The pilot script (`scripts/experiments/run_pilot.py`) is intended for pipeline validation and figure layout checks only — it uses a reduced iteration budget and results should not be interpreted as final. The full-budget configuration defined in each experiment driver is required to reproduce the paper figures and tables.
