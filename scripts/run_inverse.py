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

def _load_dataset(dataset_source):
    if isinstance(dataset_source, str):
        data_npz = np.load(dataset_source)
        dataset_label = dataset_source
    else:
        data_npz = dataset_source
        dataset_label = dataset_source.get("name", "in_memory_dataset")
    return data_npz, dataset_label


def _sanitize_label(text):
    return str(text).replace(os.sep, "_").replace(" ", "_")


def _set_random_seed(random_seed):
    if random_seed is None:
        return
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    dde.config.set_random_seed(random_seed)

def run_inverse(
    dataset_path,
    outdir_base="results/inverse",
    C1_guess=5.0,
    C2_guess=2.0,
    loss_weights=None,
    observation_stride=10,
    observed_components=None,
    train_iterations=10000,
    observation_indices=None,
    random_seed=None,
    save_checkpoint=False,
):
    _set_random_seed(random_seed)

    # Load dataset
    data_npz, dataset_label = _load_dataset(dataset_path)
    t = np.asarray(data_npz["t"])
    x_obs = np.asarray(data_npz["y"])
    y_true = np.asarray(data_npz["y_clean"] if "y_clean" in data_npz else data_npz["y"])
    x0 = x_obs[0]
    beta_true, n_true = float(data_npz["beta"]), float(data_npz["n"])
    noise_sigma = data_npz["noise"]

    if observed_components is None:
        observed_components = [0, 1, 2]

    component_tag = "-".join(str(component + 1) for component in observed_components)

    if observation_indices is None:
        observation_indices = np.arange(0, len(t), observation_stride)
    else:
        observation_indices = np.array(sorted(set(int(index) for index in observation_indices)), dtype=int)
        observation_stride = -1

    observation_count = len(observation_indices)
    dataset_tag = _sanitize_label(os.path.splitext(os.path.basename(dataset_label))[0])
    seed_tag = f"seed-{random_seed}" if random_seed is not None else "seed-none"

    # Create results directory
    outdir = os.path.join(
        outdir_base,
        dataset_tag,
        f"obs-{component_tag}_count-{observation_count}",
        seed_tag,
    )
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
    t_obs = t[observation_indices]
    x_obs_sub = x_obs[observation_indices]

    observe_bc = []
    for i in observed_components:
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
    model.train(iterations=train_iterations, callbacks=[variable_callback])

    # Predictions
    y_pred = model.predict(t)

    if save_checkpoint:
        model.save(os.path.join(outdir, "model_checkpoint"), protocol="backend", verbose=0)

    # Save estimated parameters
    est_beta, est_n = C1.value().numpy().item(), C2.value().numpy().item()
    beta_abs_error = abs(est_beta - beta_true)
    n_abs_error = abs(est_n - n_true)
    beta_rel_error = beta_abs_error / abs(beta_true)
    n_rel_error = n_abs_error / abs(n_true)
    parameter_rel_error = 0.5 * (beta_rel_error + n_rel_error)
    state_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    with open(os.path.join(outdir, "inverse_estimated_parameters.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Parameter", "True Value", "Estimated Value"])
        writer.writerow(["beta", f"{beta_true:.3f}", f"{est_beta:.6f}"])
        writer.writerow(["n", f"{n_true:.3f}", f"{est_n:.6f}"])

    with open(os.path.join(outdir, "inverse_metrics.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value"])
        writer.writerow(["dataset_path", dataset_path])
        writer.writerow(["beta_true", f"{beta_true:.6f}"])
        writer.writerow(["n_true", f"{n_true:.6f}"])
        writer.writerow(["beta_estimated", f"{est_beta:.6f}"])
        writer.writerow(["n_estimated", f"{est_n:.6f}"])
        writer.writerow(["beta_abs_error", f"{beta_abs_error:.6f}"])
        writer.writerow(["n_abs_error", f"{n_abs_error:.6f}"])
        writer.writerow(["beta_rel_error", f"{beta_rel_error:.6f}"])
        writer.writerow(["n_rel_error", f"{n_rel_error:.6f}"])
        writer.writerow(["parameter_rel_error", f"{parameter_rel_error:.6f}"])
        writer.writerow(["state_rmse", f"{state_rmse:.6f}"])
        writer.writerow(["observation_stride", observation_stride])
        writer.writerow(["observation_count", observation_count])
        writer.writerow(["observed_components", ",".join(str(component) for component in observed_components)])
        writer.writerow(["train_iterations", train_iterations])
        writer.writerow(["random_seed", random_seed])

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
    return {
        "dataset_path": dataset_label,
        "beta_true": beta_true,
        "n_true": n_true,
        "beta_estimated": est_beta,
        "n_estimated": est_n,
        "beta_abs_error": beta_abs_error,
        "n_abs_error": n_abs_error,
        "beta_rel_error": beta_rel_error,
        "n_rel_error": n_rel_error,
        "parameter_rel_error": parameter_rel_error,
        "state_rmse": state_rmse,
        "noise": float(noise_sigma),
        "observed_components": list(observed_components),
        "observation_stride": observation_stride,
        "observation_count": observation_count,
        "observation_indices": observation_indices.tolist(),
        "train_iterations": train_iterations,
        "random_seed": random_seed,
        "y_true": y_true,
        "y_pred": y_pred,
        "outdir": outdir,
    }
