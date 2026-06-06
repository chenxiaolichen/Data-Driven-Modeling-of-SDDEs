import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os

# Global Settings
tf.keras.backend.set_floatx('float64')
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 2


# Data Loading
def load_sdde_data(filename):
    data = np.load(filename)
    x_matrix = data['x_matrix']
    t_values = data['t_values']
    return x_matrix, t_values


# Delay Matrix Reconstruction
def reconstruct_delay_matrix(x_matrix, tau_star, h):
    n_traj, traj_length, n_dimensions = x_matrix.shape
    k = int(tau_star / h)

    all_x_current = []
    all_x_delay = []
    all_x_next = []

    for i in range(n_traj):
        x_traj = x_matrix[i]
        for j in range(traj_length - 1):
            delay_idx = max(0, j - k)
            x_current_j = x_traj[j]
            x_delay_j = x_traj[delay_idx]
            x_next_j = x_traj[j + 1]

            all_x_current.append(x_current_j)
            all_x_delay.append(x_delay_j)
            all_x_next.append(x_next_j)

    x_current = np.array(all_x_current)
    x_delay = np.array(all_x_delay)
    x_next = np.array(all_x_next)
    return x_current, x_delay, x_next


# SDDE Integrators
class SDDEIntegrators:
    @staticmethod
    def euler_maruyama(xn, xn_tau, h, f_sigma, rng):
        dW = rng.normal(loc=0, scale=np.sqrt(h), size=xn.shape)
        xk = xn.reshape(1, -1)
        xk_tau = xn_tau.reshape(1, -1)
        fk, sk = f_sigma([xk, xk_tau])
        return xk + h * fk + sk * dW


# Model Builder
class ModelBuilder:
    @staticmethod
    def define_gaussian_process(n_input_dimensions, n_output_dimensions, n_layers, n_dim_per_layer, name,
                                activation="tanh"):
        input_current = Input((n_input_dimensions,), dtype=tf.float64, name=name + '_current')
        input_delay = Input((n_input_dimensions,), dtype=tf.float64, name=name + '_delay')
        merged = Concatenate()([input_current, input_delay])

        x = merged
        for _ in range(n_layers):
            x = Dense(n_dim_per_layer, activation=activation, dtype=tf.float64)(x)

        drift_output = Dense(n_output_dimensions, dtype=tf.float64)(x)
        diffusion_output = Dense(n_output_dimensions,
                                 activation=lambda x: tf.nn.softplus(x) + 1e-13,
                                 dtype=tf.float64)(x)
        return Model(inputs=[input_current, input_delay], outputs=[drift_output, diffusion_output])


# SDDE Approximation Network
class SDDEApproximationNetwork(tf.keras.Model):
    def __init__(self, sde_model, step_size, method="euler", **kwargs):
        super().__init__(**kwargs)
        self.sde_model = sde_model
        self.step_size = step_size
        self.method = method

    def call(self, inputs):
        x_n, x_n_tau, x_np1 = inputs
        if self.method == "euler":
            drift, diffusion = self.sde_model([x_n, x_n_tau])
            h_tensor = tf.cast(self.step_size, dtype=diffusion.dtype)
            mean = x_n + h_tensor * drift
            std = tf.math.sqrt(h_tensor) * diffusion
            dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
            log_prob = dist.log_prob(x_np1)
            loss = -tf.reduce_mean(log_prob)
            self.add_loss(loss)
            self.add_metric(loss, name="nll", aggregation="mean")
        return self.sde_model([x_n, x_n_tau])


# Data Preprocessing
def preprocess_data(x_current, x_delay, x_next, test_size=0.2, random_state=42):
    N_samples, n_dimensions = x_current.shape
    X = np.hstack([x_current, x_delay])
    y = x_next

    test_size = min(test_size, (N_samples - 1) / N_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    x_current_train = X_train[:, :n_dimensions]
    x_delay_train = X_train[:, n_dimensions:]
    x_current_test = X_test[:, :n_dimensions]
    x_delay_test = X_test[:, n_dimensions:]

    return (x_current_train, x_delay_train, y_train,
            x_current_test, x_delay_test, y_test, n_dimensions)


# Model Training
def train_and_evaluate_model(x_current_train, x_delay_train, y_train,
                             x_current_test, x_delay_test, y_test,
                             n_dimensions, step_size, epochs=300, batch_size=128,
                             model_save_path=None):
    model = ModelBuilder.define_gaussian_process(
        n_input_dimensions=n_dimensions,
        n_output_dimensions=n_dimensions,
        n_layers=2,
        n_dim_per_layer=32,
        name="sdde_model"
    )

    sdde_approximator = SDDEApproximationNetwork(
        sde_model=model,
        step_size=step_size
    )

    sdde_approximator.compile(optimizer=Adam(learning_rate=0.001))
    callbacks = []

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_save_path, monitor='val_nll', save_best_only=True,
            save_weights_only=True, mode='min'
        )
        callbacks.append(checkpoint)

    batch_size = min(batch_size, len(x_current_train))
    history = sdde_approximator.fit(
        [x_current_train, x_delay_train, y_train],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1 if len(x_current_train) >= 100 else 0,
        callbacks=callbacks
    )

    if model_save_path and os.path.exists(model_save_path):
        sdde_approximator.load_weights(model_save_path)

    test_loss = sdde_approximator.evaluate(
        [x_current_test, x_delay_test, y_test], verbose=0
    )
    print(f"Test NLL Loss: {test_loss[0]:.4f}")

    return sdde_approximator


# Model Loading
def load_sdde_model(n_dimensions, step_size, model_weights_path):
    model = ModelBuilder.define_gaussian_process(
        n_input_dimensions=n_dimensions,
        n_output_dimensions=n_dimensions,
        n_layers=2,
        n_dim_per_layer=32,
        name="sdde_model"
    )

    sdde_approximator = SDDEApproximationNetwork(
        sde_model=model,
        step_size=step_size
    )
    sdde_approximator.compile(optimizer=Adam(learning_rate=0.001))

    if not sdde_approximator.built:
        dummy = [tf.random.uniform((1, n_dimensions), dtype=tf.float64) for _ in range(3)]
        sdde_approximator(dummy)

    sdde_approximator.load_weights(model_weights_path)
    return sdde_approximator


# Drift / Diffusion Visualization
def compare_drift_diffusion(apx_model, true_sdde, x_range=None, x_tau_range=None, n_points=50):
    if x_range is None:
        x_range = np.linspace(0, 1, n_points)
    if x_tau_range is None:
        x_tau_range = np.linspace(0, 1, n_points)

    sde_model = apx_model.sde_model

    true_drift = np.zeros((n_points, n_points))
    true_diff = np.zeros((n_points, n_points))
    for i, x in enumerate(x_range):
        for j, x_tau in enumerate(x_tau_range):
            drift, diff = true_sdde(0, np.array([[x]]), np.array([[x_tau]]))
            true_drift[i, j] = drift[0, 0]
            true_diff[i, j] = diff[0, 0]

    apx_drift = np.zeros((n_points, n_points))
    apx_diff = np.zeros((n_points, n_points))
    for i, x in enumerate(x_range):
        for j, x_tau in enumerate(x_tau_range):
            x_tensor = [
                tf.convert_to_tensor(np.array([[x]]), dtype=tf.float64),
                tf.convert_to_tensor(np.array([[x_tau]]), dtype=tf.float64)
            ]
            drift, diff = sde_model(x_tensor)
            apx_drift[i, j] = drift.numpy()[0, 0]
            apx_diff[i, j] = diff.numpy()[0, 0]

    drift_min = min(np.min(true_drift), np.min(apx_drift))
    drift_max = max(np.max(true_drift), np.max(apx_drift))

    # True Drift
    plt.figure(figsize=(10, 8))
    im1 = plt.imshow(true_drift, cmap='viridis', extent=[x_tau_range[0], x_tau_range[-1], x_range[0], x_range[-1]],
                     origin='lower', aspect='auto', vmin=drift_min, vmax=drift_max)
    plt.xlabel('x(t-τ)', fontsize=34)
    plt.ylabel('x(t)', fontsize=34)
    plt.title('True f', fontsize=34)

    cbar1 = plt.colorbar(im1)
    cbar1.ax.linewidth = 2
    cbar1.ax.tick_params(labelsize=30)
    cbar1.set_ticks(np.linspace(drift_min, drift_max, 5))

    plt.tick_params(axis='both', which='major', labelsize=34, width=2, length=6)
    plt.tight_layout()
    plt.savefig(r'E:\1d\true_drift1.png', dpi=600, bbox_inches='tight')
    plt.show()

    # Predicted Drift
    plt.figure(figsize=(10, 8))
    im2 = plt.imshow(apx_drift, cmap='viridis', extent=[x_tau_range[0], x_tau_range[-1], x_range[0], x_range[-1]],
                     origin='lower', aspect='auto', vmin=drift_min, vmax=drift_max)
    plt.xlabel('x(t-τ)', fontsize=34)
    plt.ylabel('x(t)', fontsize=34)
    plt.title('Predicted f', fontsize=34)

    cbar2 = plt.colorbar(im2)
    cbar2.ax.linewidth = 2
    cbar2.ax.tick_params(labelsize=30)
    cbar2.set_ticks(np.linspace(drift_min, drift_max, 5))

    plt.tick_params(axis='both', which='major', labelsize=34, width=2, length=6)
    plt.tight_layout()
    plt.savefig(r'E:\1d\predicted_drift1.png', dpi=600, bbox_inches='tight')
    plt.show()

    # Diffusion Comparison
    plt.figure(figsize=(10, 8))
    true_diff_mean = np.mean(true_diff, axis=1)
    apx_diff_mean = np.mean(apx_diff, axis=1)
    plt.plot(x_range, true_diff_mean, 'b-', linewidth=2, label='True')
    plt.plot(x_range, apx_diff_mean, 'r--', linewidth=2, label='Predicted')
    plt.xlabel('x(t)', fontsize=34)
    plt.ylabel('σ', fontsize=34)
    plt.legend(fontsize=30, frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=34, width=2, length=6)
    plt.tight_layout()
    plt.savefig(r'E:\1d\diffusion_comparison1.png', dpi=600, bbox_inches='tight')
    plt.show()


def simulate_trajectories(sde_model, true_sdde, initial_conditions, tau_true, tau_pred, T, h, rng, save_path=None):
    max_tau = max(tau_true, tau_pred)
    num_steps = int((T + max_tau) / h)
    t_values = np.linspace(-max_tau, T, num_steps + 1)
    start_idx = int(max_tau / h)
    n_dim = initial_conditions[0].shape[-1]

    shared_noise = rng.normal(loc=0, scale=np.sqrt(h), size=(num_steps, n_dim))

    true_colors = ["#0056b4", "#d62728", "#2ca02c", "#ff7f0e"]
    pred_colors = ["#a1c7f0", "#f7a1a1", "#98d798", "#ffbc77"]

    plt.figure(figsize=(10, 8))

    for idx, ic in enumerate(initial_conditions):
        true_x = np.zeros((num_steps + 1, n_dim))
        pred_x = np.zeros((num_steps + 1, n_dim))
        true_x[0] = ic
        pred_x[0] = ic

        for i in range(1, num_steps + 1):
            if t_values[i] <= 0:
                true_x[i] = ic
                pred_x[i] = ic
            else:
                # True trajectory
                di = max(0, i - int(tau_true / h))
                xc, xd = true_x[i-1], true_x[di]
                df, di = true_sdde(t_values[i], xc, xd)
                true_x[i] = xc + h * df + di * shared_noise[i-1]

                # Predicted trajectory
                di_p = max(0, i - int(tau_pred / h))
                xc_p = pred_x[i-1].reshape(1, -1)
                xd_p = pred_x[di_p].reshape(1, -1)
                xt = [tf.convert_to_tensor(xc_p, tf.float64), tf.convert_to_tensor(xd_p, tf.float64)]
                df_p, di_p = sde_model(xt)
                pred_x[i] = xc_p.flatten() + h * df_p.numpy().flatten() + di_p.numpy().flatten() * shared_noise[i-1]

        x0 = ic[0][0]
        plt.plot(t_values[start_idx:], true_x[start_idx:, 0], c=true_colors[idx], lw=4, label=f'$x_0={x0:.1f}$')
        plt.plot(t_values[start_idx:], pred_x[start_idx:, 0], c=pred_colors[idx], ls='--', lw=4)

    plt.xlabel('t', fontsize=34)
    plt.ylabel('x(t)', fontsize=34)
    plt.legend(fontsize=28, frameon=False, ncol=2, loc='upper right')
    plt.tick_params(labelsize=34, width=2, length=6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

    return true_x, pred_x



def true_sdde(t, x, x_delay):
    """True SDDE model: drift = 1.8*x*(1 - x_delay), diffusion = 0.1*x"""
    drift_term = 1.8 * x * (1 - x_delay)
    diffusion_term = 0.1 * x
    return drift_term, diffusion_term



if __name__ == "__main__":
    data_filename = r"E:\1d\sdde_data.npz"
    model_save_path = r"E:\1d\sdde_model_best.h5"


    # Load and process data
    x_matrix, t_values = load_sdde_data(data_filename)
    x_current, x_delay, x_next = reconstruct_delay_matrix(x_matrix, tau_star, h)

    (x_current_train, x_delay_train, y_train,
     x_current_test, x_delay_test, y_test,
     n_dimensions) = preprocess_data(x_current, x_delay, x_next)

    # Train model
    trained_model = train_and_evaluate_model(
        x_current_train, x_delay_train, y_train,
        x_current_test, x_delay_test, y_test,
        n_dimensions=n_dimensions, step_size=h,
        epochs=300, batch_size=128,
        model_save_path=model_save_path
    )

    compare_drift_diffusion(trained_model, true_sdde)