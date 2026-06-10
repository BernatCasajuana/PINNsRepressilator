"""
Trains a Physics-Informed Neural Network (PINN) to predict the Repressilator dynamics from a single dataset.
Loads the .npz file containing time points, simulated protein concentrations, and model parameters (beta, n, noise).
The ODE system, initial conditions and the observed data are defined for the PINN, which is then trained.
Predictions and training losses are saved as plots in a dedicated output folder.
"""

# %% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["tf_use_legacy_keras"] = "1"
os.environ["dde_backend"] = "tensorflow"

import deepxde as dde
import tensorflow as tf

PREDICTION_COLORS = ("#0072B2", "#E69F00", "#009E73")


def _normalize_observed_components(observed_components, state_dim):
    if observed_components is None:
        components = list(range(state_dim))
    else:
        try:
            components = [int(component) for component in observed_components]
        except TypeError as exc:
            raise TypeError("observed_components must be an iterable of integers.") from exc

    if not components:
        raise ValueError("observed_components cannot be empty.")

    invalid_components = [component for component in components if component < 0 or component >= state_dim]
    if invalid_components:
        raise ValueError(
            f"observed_components must be between 0 and {state_dim - 1}. Got {invalid_components}."
        )

    if len(set(components)) != len(components):
        raise ValueError(f"observed_components cannot contain duplicates. Got {components}.")

    return components


def _validate_loss_weights(loss_weights, expected_terms):
    if loss_weights is None:
        return None

    normalized_weights = list(loss_weights)
    if len(normalized_weights) != expected_terms:
        raise ValueError(
            "loss_weights length mismatch: expected "
            f"{expected_terms} terms (3 equations + 3 ICs + {expected_terms - 6} observations), "
            f"got {len(normalized_weights)}."
        )

    return normalized_weights


def _build_loss_component_names(observed_components, actual_count):
    expected_names = (
        ["Eq1 (dx1/dt)", "Eq2 (dx2/dt)", "Eq3 (dx3/dt)"]
        + ["IC x1", "IC x2", "IC x3"]
        + [f"Obs x{component + 1}" for component in observed_components]
    )
    if len(expected_names) == actual_count:
        return expected_names
    return [f"Loss {index + 1}" for index in range(actual_count)]


def _build_loss_component_styles(actual_count):
    category_colors = {
        "eq": PREDICTION_COLORS[0],
        "ic": PREDICTION_COLORS[1],
        "obs": PREDICTION_COLORS[2],
    }
    term_linestyles = ["-", "--", ":"]

    styles = []
    for index in range(actual_count):
        if index < 3:
            color = category_colors["eq"]
            term_index = index
        elif index < 6:
            color = category_colors["ic"]
            term_index = index - 3
        else:
            color = category_colors["obs"]
            term_index = (index - 6) % len(term_linestyles)

        styles.append((color, term_linestyles[term_index % len(term_linestyles)]))

    return styles


def _format_noise_for_plot(noise_sigma):
    formatted = f"{float(noise_sigma):.3f}".rstrip("0").rstrip(".")
    if "." not in formatted:
        return f"{formatted}.00"
    decimal_count = len(formatted.split(".", 1)[1])
    if decimal_count == 1:
        return f"{formatted}0"
    return formatted

# %% Define ODE system
def ode_system(x, y, beta, n):
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    dy1 = dde.grad.jacobian(y, x, i=0, j=0)
    dy2 = dde.grad.jacobian(y, x, i=1, j=0)
    dy3 = dde.grad.jacobian(y, x, i=2, j=0)

    eq1 = dy1 - (beta / (1 + y3**n) - y1)
    eq2 = dy2 - (beta / (1 + y1**n) - y2)
    eq3 = dy3 - (beta / (1 + y2**n) - y3)

    return [eq1, eq2, eq3]

# %% Main function to run forward problem and plot the results

def run_forward(
    dataset_path,
    loss_weights=None,
    outdir_base="results/forward",
    observation_stride=10,
    observed_components=None,
    adam_epochs=5000,
    run_lbfgs=True,
):
    # Load dataset
    data_npz = np.load(dataset_path)
    t = np.asarray(data_npz["t"], dtype=float)
    if t.ndim == 1:
        t = t[:, None]
    if t.ndim != 2 or t.shape[1] != 1:
        raise ValueError(f"Expected t to have shape (N, 1) or (N,), got {t.shape}.")

    x_obs = np.asarray(data_npz["y"], dtype=float)
    if x_obs.ndim != 2 or x_obs.shape[1] != 3:
        raise ValueError(f"Expected y to have shape (N, 3), got {x_obs.shape}.")
    if len(t) != len(x_obs):
        raise ValueError(f"t and y must have the same number of rows. Got {len(t)} and {len(x_obs)}.")

    x0 = x_obs[0]                                           
    beta, n = float(data_npz["beta"]), float(data_npz["n"])
    noise_sigma = float(np.asarray(data_npz["noise"]).squeeze())
    noise_text = _format_noise_for_plot(noise_sigma)

    observed_components = _normalize_observed_components(observed_components, state_dim=x_obs.shape[1])
    expected_loss_terms = 6 + len(observed_components)
    loss_weights = _validate_loss_weights(loss_weights, expected_loss_terms)

    observation_stride = int(observation_stride)
    if observation_stride <= 0:
        raise ValueError(f"observation_stride must be a positive integer. Got {observation_stride}.")

    component_tag = "-".join(str(component + 1) for component in observed_components)
    
    # Create results directory
    outdir = os.path.join(
        outdir_base,
        f"beta{beta}_n{n}_noise{noise_sigma}",
        f"obs-{component_tag}_stride-{observation_stride}",
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
    t_obs = t[::observation_stride]
    x_obs_sub = x_obs[::observation_stride]

    observe_bc = []
    for i in observed_components:
        bc = dde.icbc.PointSetBC(t_obs, x_obs_sub[:, i:i+1], component=i)
        observe_bc.append(bc)

    # Define function with parameters
    def ode_func(x, y):
        return ode_system(x, y, beta, n)

    # Define data object for DeepXDE
    data_ode = dde.data.PDE(
        geom,
        ode_func,
        [ic1, ic2, ic3] + observe_bc, 
        num_domain=5000, 
        num_boundary=2, 
        num_test=1000,
    )
    
    # Neural network architecture
    layer_size = [1] + [100] * 5 + [3] # 5 hidden layers with 100 neurons each
    net = dde.nn.FNN(layer_size, "sin", "Glorot uniform") # Sine activation and Glorot initialization for oscillatory problems
    net.apply_output_transform(lambda x, y: tf.nn.softplus(y)) # positive outputs

    # Define the model, compile and train
    model = dde.Model(data_ode, net)
    model.compile("adam", lr=0.001, loss_weights=loss_weights) # Adam optimizer
    model.train(epochs=adam_epochs)

    # Fine-tuning with L-BFGS
    if run_lbfgs:
        model.compile("L-BFGS")
        model.train()

    # Predictions
    y_pred = model.predict(t)

    # Plot training loss
    loss_history = model.losshistory
    loss_train = np.array(loss_history.loss_train) # loss history per component
    loss_steps = np.asarray(getattr(loss_history, "steps", []), dtype=float).reshape(-1)
    total_iterations = float(adam_epochs)
    if loss_steps.size > 0 and np.all(np.diff(loss_steps) >= 0):
        total_iterations = max(total_iterations, float(loss_steps[-1]))
    iteration_axis = np.linspace(0.0, total_iterations, loss_train.shape[0], dtype=float)
    loss_components = loss_train.T
    component_names = _build_loss_component_names(observed_components, actual_count=loss_components.shape[0])
    component_styles = _build_loss_component_styles(loss_components.shape[0])

    plt.figure(figsize=(10, 6))
    for name, loss, (color, linestyle) in zip(component_names, loss_components, component_styles):
        plt.semilogy(iteration_axis, loss, label=name, color=color, linestyle=linestyle)
    axis = plt.gca()
    axis.ticklabel_format(style="plain", axis="x", useOffset=False)
    axis.xaxis.get_offset_text().set_visible(False)
    plt.xlabel("Iterations")
    plt.ylabel("Loss (log scale)")
    plt.title(f"Training Loss ($\\beta$={beta:.1f}, $n$={n:.1f}, $\\sigma$={noise_text})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "forward_loss.png")) # save plot
    plt.close()

    # Plot predictions vs data
    plt.figure(figsize=(12, 6))
    labels = ["Repressor 1", "Repressor 2", "Repressor 3"]
    colors = list(PREDICTION_COLORS)

    for i in range(3):
        plt.plot(t, x_obs[:, i], "-", color=colors[i], label=f"{labels[i]} (data)") # obtained data
        plt.plot(t, y_pred[:, i], "--", color=colors[i], label=f"{labels[i]} (PINN)") # PINN prediction
    plt.xlabel("Time")
    plt.ylabel("Protein Concentration")
    plt.title(f"Repressilator Dynamics Prediction ($\\beta$={beta:.1f}, $n$={n:.1f}, $\\sigma$={noise_text})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "forward_prediction.png")) # save plot
    plt.close()

    print(f"Saved forward results in {outdir}") # print path to results
    return {
        "dataset_path": dataset_path,
        "beta": beta,
        "n": n,
        "noise": float(noise_sigma),
        "observed_components": list(observed_components),
        "observation_stride": observation_stride,
        "outdir": outdir,
    }
