"""
Parameter space figure — repressilator bifurcation diagram.

Shows the (β, n) plane divided into stable and oscillatory regimes by the
Hopf bifurcation boundary.  The boundary is derived analytically from the
linearisation of the repressilator ODE at its symmetric fixed point.

At the fixed point x* = β/(1 + x*ⁿ), the Jacobian is a 3×3 circulant
matrix whose eigenvalues are -1 - g·ωₖ², where ω = e^(2πi/3) and
g = n·x*ⁿ/(1 + x*ⁿ).  The Hopf condition Re(λ₁) = 0 gives g = 2, which
combined with the fixed-point equation yields the closed-form boundary:

    β_crit(n) = (n / (n − 2)) · (2 / (n − 2))^(1/n),  n > 2

For n ≤ 2 the fixed point is always stable regardless of β.

Key points shown:
  - Canonical oscillatory case: β = 5.0, n = 3.0  (all main experiments)
  - Stable comparison:          β = 5.0, n = 1.5  (exp5)
  - Higher-β oscillatory case:  β = 8.0, n = 3.0  (exp5)
  - Higher-β stable case:       β = 8.0, n = 1.5  (exp5)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from experiments.utils import finalize_figure

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.useoffset": False,
})

BETA_MAX = 9.0
N_MIN    = 2.1
N_MAX    = 6.0

COLOR_STABLE = "#E2E2E2"   # slightly darker light grey
COLOR_OSC    = "#FFFFFF"   # white
COLOR_BOUNDARY = "#333333"

POINT_OSC    = "#0072B2"   # blue  — canonical / exp5 osc
POINT_STABLE = "#E69F00"   # orange — exp5 stable

figure_path = "figures/param_space.png"


def _beta_crit(n_arr):
    """Hopf bifurcation boundary: β_crit(n) for n > 2."""
    n = np.asarray(n_arr, dtype=float)
    return (n / (n - 2.0)) * (2.0 / (n - 2.0)) ** (1.0 / n)


def main():
    os.makedirs("figures", exist_ok=True)

    # --- boundary curve ---
    n_boundary = np.linspace(2.001, N_MAX, 2000)
    beta_boundary = _beta_crit(n_boundary)
    # clip to plot range
    mask = beta_boundary <= BETA_MAX
    n_plot   = n_boundary[mask]
    b_plot   = beta_boundary[mask]
    # extend boundary to n=2 at β→∞  (vertical approach)
    n_plot   = np.concatenate([[2.0], n_plot])
    b_plot   = np.concatenate([[BETA_MAX], b_plot])

    fig, ax = plt.subplots(figsize=(5, 4))

    # --- fill regions ---
    # Stable: left of boundary (low β for n>2)
    ax.fill_betweenx(n_plot, 0, b_plot, color=COLOR_STABLE, zorder=0)
    # Oscillatory: right of boundary
    ax.fill_betweenx(n_plot, b_plot, BETA_MAX, color=COLOR_OSC, zorder=0)

    # --- boundary ---
    ax.plot(b_plot[1:], n_plot[1:], color=COLOR_BOUNDARY, linewidth=0.8, zorder=2)

    # --- canonical point only ---
    ax.scatter(5.0, 3.0, color="black", s=7, zorder=5)
    ax.text(4.92, 3.09, r"$\mathit{Canonical}\ (\beta{=}5,\,n{=}3)$", fontsize=8.5,
            color="black", ha="left", va="bottom")

    # --- region labels ---
    ax.text(1.5, 2.75, "Stable", fontsize=11, color="black",
            ha="center", va="center", style="italic")
    ax.text(5.2, 4.65, "Oscillatory", fontsize=11, color="black",
            ha="center", va="center", style="italic")

    ax.set_xlabel(r"Production Rate $\beta$", fontsize=11)
    ax.set_ylabel(r"Hill Coefficient $n$", fontsize=11)
    ax.set_xlim(0, BETA_MAX)
    ax.set_ylim(N_MIN, N_MAX)
    ax.set_title("Parameter Space")

    finalize_figure(figure_path)
    print(f"Parameter space figure saved to {figure_path}")


if __name__ == "__main__":
    main()
