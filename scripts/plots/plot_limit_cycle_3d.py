"""
plot_limit_cycle_3d.py

3D phase-space plot of the repressilator limit cycle in the space of the
three protein concentrations (x1, x2, x3).

Parameters: β = 8, n = 3
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from generate_data import protein_repressilator_rhs

x0 = [1.0, 1.0, 1.2]
t = np.linspace(0, 50, 1000)
x = scipy.integrate.odeint(protein_repressilator_rhs, x0, t, args=(8, 3))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.view_init(30, 30)
ax.plot(x[:, 0], x[:, 1], x[:, 2])
ax.set_xlabel("x1 concentration")
ax.set_ylabel("x2 concentration")
ax.set_zlabel("x3 concentration")
ax.set_title("Repressilator Limit Cycle (β=8, n=3)")

plt.tight_layout()
plt.savefig("limit_cycle_3d.png", dpi=150)
plt.show()
