"""
Trains a Physics-Informed Neural Network (PINN) to estimate the Repressilator parameters (beta, n) from a single dataset.
Loads the .npz file containing time points, simulated protein concentrations, and true model parameters (beta, n, noise).
Defines the ODE system, initial conditions, observed data, and trainable parameters with initial guesses for the PINN, which is then trained.
Parameter estimates, predictions, and training losses are saved in a dedicated output folder.
"""

# %% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["tf_use_legacy_keras"] = "1" # Use legacy Keras API for compatibility with DeepXDE
os.environ["dde_backend"] = "tensorflow"  # Force TensorFlow backend before importing deepxde

import deepxde as dde
import tensorflow as tf
import csv

PREDICTION_COLORS = ("#0072B2", "#E69F00", "#009E73")
LOSS_COMPONENT_COLORS = {
    "eq": "#C62828",
    "ic": "#00897B",
    "obs": "#8C564B",
}

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


def _build_observation_indices(total_points, observation_stride, observation_indices):
    if observation_indices is None:
        stride = int(observation_stride)
        if stride <= 0:
            raise ValueError(f"observation_stride must be a positive integer. Got {observation_stride}.")
        return np.arange(0, total_points, stride, dtype=int), stride

    indices = np.array(sorted(set(int(index) for index in observation_indices)), dtype=int)
    if indices.size == 0:
        raise ValueError("observation_indices cannot be empty.")

    if np.any(indices < 0) or np.any(indices >= total_points):
        raise ValueError(
            f"observation_indices must be between 0 and {total_points - 1}. Got {indices.tolist()}."
        )

    return indices, -1


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
    category_colors = LOSS_COMPONENT_COLORS
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


def _variable_to_scalar(variable):
    value = variable.value() if hasattr(variable, "value") else variable

    if hasattr(value, "numpy"):
        try:
            return float(np.asarray(value.numpy(), dtype=float).squeeze())
        except Exception:
            pass

    try:
        return float(np.asarray(dde.backend.to_numpy(value), dtype=float).squeeze())
    except Exception:
        pass

    try:
        return float(np.asarray(tf.keras.backend.get_value(value), dtype=float).squeeze())
    except Exception:
        pass

    try:
        session = tf.compat.v1.keras.backend.get_session()
        return float(np.asarray(session.run(value), dtype=float).squeeze())
    except Exception as exc:
        raise TypeError("Unable to convert trainable variable to scalar for current TensorFlow backend.") from exc


def _format_noise_for_plot(data_npz, noise_sigma):
    if "noise_level" in data_npz:
        noise_value = float(np.asarray(data_npz["noise_level"]).squeeze())
    else:
        noise_value = float(noise_sigma)
    formatted = f"{noise_value:.3f}".rstrip("0").rstrip(".")
    if "." not in formatted:
        return f"{formatted}.00"
    decimal_count = len(formatted.split(".", 1)[1])
    if decimal_count == 1:
        return f"{formatted}0"
    return formatted

def run_inverse(
    dataset_path,
    outdir_base="results/inverse",
    beta_guess=5.0,
    n_guess=2.0,
    loss_weights=None,
    observation_stride=10,
    observed_components=None,
    train_iterations=5000,
    observation_indices=None,
    random_seed=None,
    save_checkpoint=False,
):
    _set_random_seed(random_seed)

    # Load dataset
    data_npz, dataset_label = _load_dataset(dataset_path)
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

    y_true = np.asarray(data_npz["y_clean"] if "y_clean" in data_npz else data_npz["y"], dtype=float)
    if y_true.shape != x_obs.shape:
        raise ValueError(f"Expected y_clean/y to have shape {x_obs.shape}, got {y_true.shape}.")
    x0 = x_obs[0]
    beta_true, n_true = float(data_npz["beta"]), float(data_npz["n"])
    noise_sigma = float(np.asarray(data_npz["noise"]).squeeze())
    noise_text = _format_noise_for_plot(data_npz, noise_sigma)

    observed_components = _normalize_observed_components(observed_components, state_dim=x_obs.shape[1])
    expected_loss_terms = 6 + len(observed_components)
    loss_weights = _validate_loss_weights(loss_weights, expected_loss_terms)

    component_tag = "-".join(str(component + 1) for component in observed_components)

    observation_indices, observation_stride = _build_observation_indices(
        total_points=len(t),
        observation_stride=observation_stride,
        observation_indices=observation_indices,
    )

    observation_count = len(observation_indices)
    # For file paths, strip the extension; for in-memory dataset names don't use splitext
    # (splitext treats e.g. ".05_seed0" in "noise0.05_seed0" as an extension).
    if isinstance(dataset_path, str):
        dataset_tag = _sanitize_label(os.path.splitext(os.path.basename(dataset_label))[0])
    else:
        import re as _re
        dataset_tag = _re.sub(r"_seed\d+$", "", _sanitize_label(dataset_label))
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
    beta_var = dde.Variable(beta_guess)
    n_var = dde.Variable(n_guess)

    # Define function with parameters
    def ode_func(x, y):
        return ode_system(x, y, beta_var, n_var)

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
            if self.model.train_state.step % self.period != 0:
                return
            super().on_epoch_end()
            vals = [_variable_to_scalar(variable) for variable in self.var]
            self.estimated_params.append(vals)

    checkpoint_period = max(1, min(100, train_iterations // 10))
    variable_callback = SaveVariablesCallback([beta_var, n_var], period=checkpoint_period)

    # Compile and train the model
    model.compile("adam", lr=0.001, external_trainable_variables=[beta_var, n_var], loss_weights=loss_weights)
    model.train(iterations=train_iterations, callbacks=[variable_callback])

    # Predictions
    y_pred = model.predict(t)

    if save_checkpoint:
        model.save(os.path.join(outdir, "model_checkpoint"), protocol="backend", verbose=0)

    # Save estimated parameters
    est_beta, est_n = _variable_to_scalar(beta_var), _variable_to_scalar(n_var)
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
        writer.writerow(["dataset_path", dataset_label])
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

    # Plot training loss
    loss_history = model.losshistory
    loss_train = np.array(loss_history.loss_train) # loss history per component
    iteration_axis = np.linspace(0.0, float(train_iterations), loss_train.shape[0], dtype=float)

    param_evo = np.array(variable_callback.estimated_params) if variable_callback.estimated_params else np.empty((0, 2))
    param_evo_iterations = np.arange(1, len(param_evo) + 1, dtype=float) * checkpoint_period
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
    plt.ylabel("Loss (Log Scale)")
    plt.title(f"Training Loss ($\\beta$={beta_true:.1f}, $n$={n_true:.1f}, $\\sigma$={noise_text})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "inverse_loss.png")) # save plot
    plt.close()

    # Plot predictions vs data
    plt.figure(figsize=(12, 6))
    labels = ["Repressor 1", "Repressor 2", "Repressor 3"]
    colors = list(PREDICTION_COLORS)

    # Per-repressor order: data (if observed) then PINN prediction, keeping legend consistent.
    for i in range(3):
        if i in observed_components:
            plt.plot(t, x_obs[:, i], "-", color=colors[i], label=f"{labels[i]} (Data)")
        plt.plot(t, y_pred[:, i], "--", color=colors[i], label=f"{labels[i]} (PINN)")
    plt.xlabel("Time")
    plt.ylabel("Protein Concentration")
    plt.title(f"Dynamics Prediction ($\\beta$={beta_true:.1f}, $n$={n_true:.1f}, $\\sigma$={noise_text})")
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
        "parameter_evolution": param_evo,           # shape (N_checkpoints, 2): [beta_hat, n_hat] every 100 iters
        "param_evo_iterations": param_evo_iterations,
        "loss_train": loss_train,                   # shape (N_steps, N_loss_components)
        "iteration_axis": iteration_axis,
        "outdir": outdir,
    }
