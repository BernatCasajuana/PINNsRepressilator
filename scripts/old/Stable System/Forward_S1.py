# Prediction of Repressilator dynamics using PINNs and DeepXDE (Parameters: n = 1.5 and beta = 10, corresponding to a stable system without oscillations)

# %% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import deepxde as dde
import tensorflow as tf

# %% Define parameters, initial conditions, and time domain
beta = 10
n = 1.5
x0 = np.array([1, 1, 1.2])
n_points = 1000
t_max = 20
t = np.linspace(0, t_max, n_points)[:, None]

# %% Simulation with ODEINT
def protein_repressilator_rhs(x, t, beta, n):
    x1, x2, x3 = x
    return [
        beta / (1 + x3 ** n) - x1,
        beta / (1 + x1 ** n) - x2,
        beta / (1 + x2 ** n) - x3,
    ]

x_ode = scipy.integrate.odeint(protein_repressilator_rhs, x0, t.flatten(), args=(beta, n))

# Save as .npz file for inverse problem
np.savez("/Users/bernatcasajuana/github/PINNs_Repressilator/Datasets/Repressilator_S1.npz", t=t, y=x_ode)

# %% PINN simulation setup
# Geometry of the problem
geom = dde.geometry.TimeDomain(0, t_max)

# Define ODE system
def ode_system(x, y):
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    dy1 = dde.grad.jacobian(y, x, i=0, j=0)
    dy2 = dde.grad.jacobian(y, x, i=1, j=0)
    dy3 = dde.grad.jacobian(y, x, i=2, j=0)

    eq1 = dy1 - (beta / (1 + y3**n) - y1)
    eq2 = dy2 - (beta / (1 + y1**n) - y2)
    eq3 = dy3 - (beta / (1 + y2**n) - y3)

    return [eq1, eq2, eq3]

# Initial conditions
def boundary(_, on_initial):
    return on_initial
ic1 = dde.icbc.IC(geom, lambda x: x0[0], boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: x0[1], boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda x: x0[2], boundary, component=2)

# Obtain observed data from odeint solution (experimental data in practice)
t_obs = t[::10] # Every 10 time points
x_obs = x_ode[::10] # Corresponding concentrations

# Implement observed data as boundary conditions
observe_bc = []
for i in range(3):
    bc = dde.icbc.PointSetBC(t_obs, x_obs[:, i:i+1], component=i)
    observe_bc.append(bc)

# Problem setup
data = dde.data.PDE(geom, ode_system, [ic1, ic2, ic3] + observe_bc, num_domain=5000, num_boundary=2, num_test=1000)

# Neural network architecture
layer_size = [1] + [100] * 5 + [3]
activation = "sin"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Force positive values for outputs (necessary if in stable parameters region)
net.apply_output_transform(lambda x, y: tf.nn.softplus(y))

# %% Compile and train
# Define data, optimizer, learning rate, metrics and training iterations
model = dde.Model(data, net)
model.compile("adam", lr=0.001) # implement weight for each loss term if needed: loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1])
model.train(epochs=5000)

# Fine tuning with L-BFGS optimizer
model.compile("L-BFGS")
model.train()

# %% Obtain the PINN prediction
y_pred = model.predict(t)

# %% Plot the training loss
loss_history = model.losshistory
loss_train = np.array(loss_history.loss_train)
epochs = np.arange(len(loss_train))

# Transpose the loss array to separate components
loss_components = loss_train.T

# Name the components for clarity
component_names = [
    "Eq1 (dx1/dt)", "Eq2 (dx2/dt)", "Eq3 (dx3/dt)",
    "IC x1", "IC x2", "IC x3",
    "Obs x1", "Obs x2", "Obs x3"
]

plt.figure(figsize=(10, 6))
for i in range(len(component_names)):
    plt.semilogy(epochs, loss_components[i], label=component_names[i])

plt.xlabel("Iterations (x1000)")
plt.ylabel("Loss (log scale)")
plt.title("Training Loss over Iterations")
plt.xlim(0, 5)
plt.legend()
plt.tight_layout()
plt.show()

# %% Plot the prediction results
plt.figure(figsize=(12, 6))
labels = ["Repressor 1", "Repressor 2", "Repressor 3"]
colors = ["tab:blue", "tab:orange", "tab:green"]

for i in range(3):
    plt.plot(t, x_ode[:, i], "-", color=colors[i], label=f"{labels[i]} (ODE simulation)")
    plt.plot(t, y_pred[:, i], "--", color=colors[i], label=f"{labels[i]} (PINN prediction)")

plt.xlabel("Time")
plt.ylabel("Protein Concentration")
plt.title("Repressilator Dynamics: ODE vs PINN")
plt.legend()
plt.tight_layout()
plt.show()