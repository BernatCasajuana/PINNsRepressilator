# Comparing numerical derivatives from scipy's odeint with TensorFlow's evaluation of the ODE system (correct formulation).

# %% Import necessary libraries
import os
os.environ["DDE_BACKEND"] = "tensorflow"  # Force TensorFlow backend before importing deepxde
import tensorflow as tf
import deepxde as dde
tf.config.run_functions_eagerly(True)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# %% Define parameters and initial conditions
beta = 10
n = 3
x0 = np.array([1.0, 1.0, 1.2])

# %% Define right-hand side (RHS) of the repressilator ODE system (for scipy.odeint)
def protein_repressilator_rhs(x, t, beta, n):
    x1, x2, x3 = x
    dx1 = beta / (1 + x3 ** n) - x1
    dx2 = beta / (1 + x1 ** n) - x2
    dx3 = beta / (1 + x2 ** n) - x3
    return [dx1, dx2, dx3]

# %% TensorFlow version of the RHS (to use with PINNs in DeepXDE)
def ode_system_manual(t_tensor, y_tensor):
    y1, y2, y3 = y_tensor[:, 0:1], y_tensor[:, 1:2], y_tensor[:, 2:3]
    dy1 = beta / (1 + y3**n) - y1
    dy2 = beta / (1 + y1**n) - y2
    dy3 = beta / (1 + y2**n) - y3
    return tf.concat([dy1, dy2, dy3], axis=1)

# %% Time points to evaluate the solution
t_values = np.linspace(0, 20, 200)

# %% Solving the ODE system with scipy's odeint
sol_odeint = odeint(protein_repressilator_rhs, x0, t_values, args=(beta, n))

# Obtain numerical derivatives from the solution
deriv_numeric = np.gradient(sol_odeint, t_values, axis=0)

# %% Evaluate the TensorFlow RHS
sol_tensor = []
for t in t_values:
    t_tensor = tf.constant([[t]], dtype=tf.float32)
    # Get the solution at the closest time point from odeint
    y_tensor = tf.constant(sol_odeint[np.argmin(np.abs(t_values - t))].reshape(1, 3), dtype=tf.float32)
    dx = ode_system_manual(t_tensor, y_tensor).numpy().flatten()
    sol_tensor.append(dx)
sol_tensor = np.array(sol_tensor)

# %% Plot protein concentrations (solution) from odeint
plt.figure(figsize=(10, 6))
for i, label in enumerate(["Repressor 1", "Repressor 2", "Repressor 3"]):
    plt.plot(t_values, sol_odeint[:, i], label=f"{label}")
plt.xlabel("Time")
plt.ylabel("Protein Concentration")
plt.title("Simulation of Repressilator Dynamics using ODEINT")
plt.legend()
plt.show()

# %% Plot derivatives: numerical from odeint vs TensorFlow RHS evaluation
plt.figure(figsize=(10, 6))
for i, label in enumerate(["Repressor 1", "Repressor 2", "Repressor 3"]):
    plt.plot(t_values, deriv_numeric[:, i], label=f"{label} (Numerical Derivative)")
    plt.plot(t_values, sol_tensor[:, i], "--", label=f"{label} (TF RHS)")
plt.xlabel("Time")
plt.ylabel("Derivative")
plt.title("Comparison: ODEINT Numerical Derivative vs. TensorFlow RHS")
plt.legend()
plt.show()