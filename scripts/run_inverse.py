"""
run_inverse.py

Trains a Physics-Informed Neural Network (PINN) to estimate the Repressilator parameters (beta, n) from a single dataset.
Loads the .npz file containing time points, simulated protein concentrations, and true model parameters (beta, n, noise).
Defines the ODE system, initial conditions, observed data, and trainable parameters with initial guesses for the PINN, which is then trained.
Parameter estimates, predictions, and training losses are saved in a dedicated output folder.
"""

# %% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" # Use legacy Keras API for compatibility with DeepXDE
os.environ["DDE_BACKEND"] = "tensorflow"  # Force TensorFlow backend before importing deepxde
import deepxde as dde
import tensorflow as tf
import argparse
import csv

# %% Define ODE system for PINN with trainable parameters
def ode_system(x, y, C1, C2):
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    dy1 = dde.grad.jacobian(y, x, i=0, j=0)
    dy2 = dde.grad.jacobian(y, x, i=1, j=0)
    dy3 = dde.grad.jacobian(y, x, i=2, j=0)

    eq1 = dy1 - (C1 / (1 + y3**C2) - y1)
    eq2 = dy2 - (C1 / (1 + y1**C2) - y2)
    eq3 = dy3 - (C1 / (1 + y2**C2) - y3)

    return [eq1, eq2, eq3]

# %% Main function to run inverse problem and plot the results

dataset_path = "datasets/beta5.0_n2.0_noise0.1.npz" # example dataset path

def run_inverse(dataset_path, outdir_base="results_inverse", C1_guess=5.0, C2_guess=2.0, loss_weights=None):
    # Load dataset
    data_npz = np.load(dataset_path)
    t = data_npz["t"]
    x_obs = data_npz["y"]
    x0 = x_obs[0]
    beta_true, n_true = float(data_npz["beta"]), float(data_npz["n"])
    noise_sigma = data_npz["noise"]

    # Create results directory
    outdir = os.path.join(outdir_base, f"beta{beta_true}_n{n_true}_noise{noise_sigma}")
    os.makedirs(outdir, exist_ok=True)

    # Define time domain
    geom = dde.geometry.TimeDomain(0, float(t.max()))

    # Define initial conditions
    def boundary(_, on_initial):
        return on_initial
    ic1 = dde.icbc.IC(geom, lambda x: x0[0], boundary, component=0)
    ic2 = dde.icbc.IC(geom, lambda x: x0[1], boundary, component=1)
    ic3 = dde.icbc.IC(geom, lambda x: x0[2], boundary, component=2)

    # Observations (subsampling every 10 time points)
    t_obs = t[::10]
    x_obs_sub = x_obs[::10]

    observe_bc = []
    for i in range(3):
        bc = dde.icbc.PointSetBC(t_obs, x_obs_sub[:, i:i+1], component=i)
        observe_bc.append(bc)

    # Trainable parameters
    C1 = dde.Variable(C1_guess) # initial guess for beta
    C2 = dde.Variable(C2_guess) # initial guess for n

    # Define function with parameters
    def ode_func(x, y):
        return ode_system(x, y, C1, C2)

    # Define data object for DeepXDE
    data_ode = dde.data.PDE(
        geom,
        ode_func,
        [ic1, ic2, ic3] + observe_bc,
        num_domain=1000,
        num_boundary=2,
        anchors=t_obs,
    )

    # Neural network architecture
    layer_size = [1] + [100] * 5 + [3] # 5 hidden layers with 100 neurons each
    net = dde.nn.FNN(layer_size, "sin", "Glorot uniform") # Sine activation and Glorot initialization for oscillatory problems
    net.apply_output_transform(lambda x, y: tf.nn.softplus(y)) # positive outputs

    # Define the model
    model = dde.Model(data_ode, net)

    # Callback to save parameter evolution
    class SaveVariablesCallback(dde.callbacks.VariableValue):
        def __init__(self, var_list, period=100):
            super().__init__(var_list, period)
            self.estimated_params = []
            self.var = var_list

        def on_epoch_end(self):
            super().on_epoch_end()
            vals = [v.value().numpy().item() for v in self.var]
            self.estimated_params.append(vals)

    variable_callback = SaveVariablesCallback([C1, C2], period=100)

    # Compile and train the model
    model.compile("adam", lr=0.001, external_trainable_variables=[C1, C2], loss_weights=loss_weights)
    model.train(iterations=10000, callbacks=[variable_callback])

    # Predictions
    y_pred = model.predict(t)

    # Save estimated parameters
    est_beta, est_n = C1.value().numpy().item(), C2.value().numpy().item()
    with open(os.path.join(outdir, "inverse_estimated_parameters.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Parameter", "True Value", "Estimated Value"])
        writer.writerow(["beta", f"{beta_true:.3f}", f"{est_beta:.6f}"])
        writer.writerow(["n", f"{n_true:.3f}", f"{est_n:.6f}"])

    # np.savetxt(os.path.join(outdir, "parameters_evolution.dat"), np.array(variable_callback.estimated_params)) # save parameter evolution

    # Plot training loss
    loss_history = model.losshistory
    loss_train = np.array(loss_history.loss_train) # loss history per component
    epochs = np.arange(len(loss_train))
    loss_components = loss_train.T
    component_names = [
        "Eq1 (dx1/dt)", "Eq2 (dx2/dt)", "Eq3 (dx3/dt)",
        "IC x1", "IC x2", "IC x3",
        "Obs x1", "Obs x2", "Obs x3"
    ]

    plt.figure(figsize=(10, 6))
    for i in range(len(component_names)):
        plt.semilogy(epochs, loss_components[i], label=component_names[i])
    plt.xlabel("Iterations")
    plt.ylabel("Loss (log scale)")
    plt.title(f"Training Loss (beta={beta_true}, n={n_true}, noise={noise_sigma})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "inverse_loss.png")) # save plot
    plt.close()

    # Plot predictions vs data
    plt.figure(figsize=(12, 6))
    labels = ["Repressor 1", "Repressor 2", "Repressor 3"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for i in range(3):
        plt.plot(t, x_obs[:, i], "-", color=colors[i], label=f"{labels[i]} (data)") # obtained data
        plt.plot(t, y_pred[:, i], "--", color=colors[i], label=f"{labels[i]} (PINN)") # PINN prediction
    plt.xlabel("Time")
    plt.ylabel("Protein Concentration")
    plt.title(f"Inverse Problem Repressilator Dynamics Prediction (beta={beta_true}, n={n_true}, noise={noise_sigma})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "inverse_prediction.png")) # save plot
    plt.close()

    print(f"Saved inverse results in {outdir}") # print path to results

# Run the inverse problem with specified dataset, initial guesses and loss weights
run_inverse(dataset_path, outdir_base="results_inverse", C1_guess=5.0, C2_guess=2.0, loss_weights=None)

# %% Run via command line interface
# if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, required=True, help="Path to dataset .npz")
    # parser.add_argument("--C1_guess", type=float, default=5.0, help="Initial guess for beta")
    # parser.add_argument("--C2_guess", type=float, default=2.0, help="Initial guess for n")
    # parser.add_argument("--loss_weights", type=float, nargs="+", default=None, help="Optional loss weights for ODE/IC/Obs")
    # args = parser.parse_args()

# Run inverse with specified dataset, initial guesses and loss weights
    # run_inverse(
        # args.dataset,
        # C1_guess=args.C1_guess,
        # C2_guess=args.C2_guess,
        # loss_weights=args.loss_weights
    # )