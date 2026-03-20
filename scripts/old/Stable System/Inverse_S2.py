# Parameter estimation in the repressilator model using PINNs and DeepXDE (Parameters: n = 1.5 and beta = 5, corresponding to a stable system without oscillations)

# %% Import necessary libraries
import os
os.environ["DDE_BACKEND"] = "tensorflow"  # Force TensorFlow backend before importing deepxde
import tensorflow as tf
import deepxde as dde
import numpy as np
import csv
import matplotlib.pyplot as plt

# %% Define parameters (with suspected values), initial conditions, and time domain
C1 = dde.Variable(4.0) # C1 = beta
C2 = dde.Variable(1.3) # C2 = n
x0 = np.array([1, 1, 1.2])
t_max = 40

# %% PINN simulation setup
# Geometry of the problem
geom = dde.geometry.TimeDomain(0, t_max)

# Define the ODE system
def ode_system(x, y):
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    dy1 = dde.grad.jacobian(y, x, i=0, j=0)
    dy2 = dde.grad.jacobian(y, x, i=1, j=0)
    dy3 = dde.grad.jacobian(y, x, i=2, j=0)

    eq1 = dy1 - (C1 / (1 + y3**C2) - y1)
    eq2 = dy2 - (C1 / (1 + y1**C2) - y2)
    eq3 = dy3 - (C1 / (1 + y2**C2) - y3)

    return [eq1, eq2, eq3]

# Initial conditions
def boundary(_, on_initial):
    return on_initial
ic1 = dde.icbc.IC(geom, lambda x: x0[0], boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: x0[1], boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda x: x0[2], boundary, component=2)

# Load observed data from odeint simulation (Repressilator.npz file)
data = np.load("/Users/bernatcasajuana/github/PINNs_Repressilator/Datasets/Repressilator_S2.npz")

# Extract time and concentration data
t_full = data["t"]
x_full = data["y"]
t_obs = t_full[::10] # Every 10 time points
x_obs = x_full[::10] # Corresponding concentrations

# Implement observed data as boundary conditions
observe_bc = []
for i in range(3):
    bc = dde.icbc.PointSetBC(t_obs, x_obs[:, i:i+1], component=i)
    observe_bc.append(bc)

# Problem setup, with anchors as extra points where the model will be evaluated
data = dde.data.PDE(geom, ode_system, [ic1, ic2, ic3] + observe_bc, num_domain=1000, num_boundary=2, anchors=t_obs)

# Neural network architecture
layer_size = [1] + [100] * 5 + [3]
activation = "sin"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Force positive values for outputs
net.apply_output_transform(lambda x, y: tf.nn.softplus(y))

# %% Compile and train
model = dde.Model(data, net)

# Callback class to save parameter evolution
class SaveVariablesCallback(dde.callbacks.VariableValue):
    def __init__(self, var_list, period=100):
        super().__init__(var_list, period)
        self.estimated_params = []
        self.var = var_list

    def on_epoch_end(self):
        super().on_epoch_end()
        vals = [v.value().numpy().item() for v in self.var]
        self.estimated_params.append(vals)

# Create callback instance
variable_callback = SaveVariablesCallback([C1, C2], period=100)

# Define data, optimizer, learning rate, training iterations and external trainable variables (C1 and C2)
model.compile("adam", lr=0.001, external_trainable_variables=[C1, C2]) # implement weight for each loss term if needed: loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1])
model.train(iterations=60000, callbacks=[variable_callback])

# Fine tuning with L-BFGS optimizer (if needed)
# model.compile("L-BFGS", external_trainable_variables=[C1, C2])
# model.train()

# %% Obtain the PINN prediction
y_pred = model.predict(t_full)

# %% Obtain the estimated parameters
print(f"Real C1 value = 5.000000")
print(f"Real C2 value = 1.500000")
print(f"Estimated value of C1 = {C1.value():.6f}")
print(f"Estimated value of C2 = {C2.value():.6f}")

# Save parameters evolution
np.savetxt("/Users/bernatcasajuana/github/PINNs_Repressilator/Datasets/Parameters_Evolution_S2.dat", np.array(variable_callback.estimated_params))

# Save in CSV
with open("/Users/bernatcasajuana/github/PINNs_Repressilator/Results/Estimated_Parameters_S2.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Parameter", "Estimated Value"])
    writer.writerow(["C1", f"{C1.value():.6f}"])
    writer.writerow(["C2", f"{C2.value():.6f}"])

# %% Plot the training loss
loss_history = model.losshistory
loss_train = np.array(loss_history.loss_train)
epochs = np.arange(len(loss_train))

# Transpose the loss array to separate components
loss_components = loss_train.T

## Name the components for clarity
component_names = [
    "Eq1 (dx1/dt)", "Eq2 (dx2/dt)", "Eq3 (dx3/dt)",
    "IC x1", "IC x2", "IC x3",
    "Obs x1", "Obs x2", "Obs x3"
]

plt.figure(figsize=(10, 6))
for i in range(len(component_names)):
    plt.semilogy(epochs, loss_components[i], label=component_names[i])

plt.xlabel("Iteration (x1000)")
plt.ylabel("Loss (log scale)")
plt.title("Training Loss over Iterations")
plt.xlim(0, 15)
plt.legend()
plt.tight_layout()
plt.show()

# %% Plot the prediction results
plt.figure(figsize=(12, 6))
labels = ["Repressor 1", "Repressor 2", "Repressor 3"]
colors = ["tab:blue", "tab:orange", "tab:green"]

for i in range(3):
    plt.plot(t_full, x_full[:, i], "-", color=colors[i], label=f"{labels[i]} (ODE simulation)")
    plt.plot(t_full, y_pred[:, i], "--", color=colors[i], label=f"{labels[i]} (PINN prediction)")

plt.xlabel("Time")
plt.ylabel("Protein Concentration")
plt.title("Repressilator Dynamics: ODE vs PINN")
plt.legend()
plt.tight_layout()
plt.show()

# %% Plot the evolution of the estimated parameters
variables = np.loadtxt("/Users/bernatcasajuana/github/PINNs_Repressilator/Datasets/Parameters_Evolution_S2.dat")
plt.figure(figsize=(8, 5))
plt.plot(variables[:, 0], label="C1 (beta)", color="tab:red")
plt.plot(variables[:, 1], label="C2 (n)", color="tab:blue")
plt.xlabel("Iteration")
plt.ylabel("Estimated Parameter Value")
plt.title("Evolution of Estimated Repressilator Parameters")
plt.legend()
plt.tight_layout()
plt.show()