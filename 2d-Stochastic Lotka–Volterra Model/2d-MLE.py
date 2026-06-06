import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import json
import pickle
from scipy.interpolate import interp1d
from sklearn.feature_selection import mutual_info_regression
from skopt import gp_minimize

# ---------------------- SETTINGS ----------------------
tf.keras.backend.set_floatx('float64')
tfd = tfp.distributions
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- GLOBAL VARS FOR BO ----------------------
trajectories = None
t_values = None
dt = None
mi_neighbors = None

# ---------------------- MODEL BUILDER ----------------------
class ModelBuilder:
    @staticmethod
    def define_gaussian_process(n_dimensions, n_layers, n_dim_per_layer, name, activation="tanh"):
        input_x_current = Input((1,), dtype=tf.float64)
        input_x_delay = Input((1,), dtype=tf.float64)
        input_y_current = Input((1,), dtype=tf.float64)
        input_y_delay = Input((1,), dtype=tf.float64)

        merged = Concatenate()([input_x_current, input_x_delay, input_y_current, input_y_delay])
        x = merged
        for _ in range(n_layers):
            x = Dense(n_dim_per_layer, activation=activation, dtype=tf.float64)(x)

        drift_output = Dense(n_dimensions, dtype=tf.float64)(x)
        diffusion_output = Dense(n_dimensions, activation=lambda x: tf.nn.softplus(x) + 1e-13, dtype=tf.float64)(x)

        return Model(
            inputs=[input_x_current, input_x_delay, input_y_current, input_y_delay],
            outputs=[drift_output, diffusion_output]
        )

class SDDEApproximationNetwork(tf.keras.Model):
    def __init__(self, sde_model, step_size, method="euler", **kwargs):
        super().__init__(**kwargs)
        self.sde_model = sde_model
        self.step_size = step_size
        self.method = method

    def call(self, inputs):
        x_current, x_delay, y_current, y_delay, next_state = inputs
        x_current = tf.cast(x_current, tf.float64)
        x_delay = tf.cast(x_delay, tf.float64)
        y_current = tf.cast(y_current, tf.float64)
        y_delay = tf.cast(y_delay, tf.float64)
        next_state = tf.cast(next_state, tf.float64)

        drift, diffusion = self.sde_model([x_current, x_delay, y_current, y_delay])
        current_state = tf.concat([x_current, y_current], axis=1)
        h = tf.cast(self.step_size, diffusion.dtype)
        mean = current_state + h * drift
        std = tf.sqrt(h) * diffusion

        dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
        log_prob = dist.log_prob(next_state)
        loss = -tf.reduce_mean(log_prob)

        self.add_loss(loss)
        self.add_metric(loss, name="nll")
        return [drift, diffusion]

# ---------------------- DATA LOADING ----------------------
def load_sdde_data_from_npz(npz_path):
    data = np.load(npz_path)
    x_matrix = data["x_matrix"]
    t_values = data["t_values"]
    dt = t_values[1] - t_values[0]
    if x_matrix.ndim == 2:
        x_matrix = x_matrix[:, :, np.newaxis]
    return x_matrix, t_values, dt

# ---------------------- BAYESIAN OPTIMIZATION ----------------------
def objective_function_mi(tau):
    tau = tau[0]
    global trajectories, t_values, dt, mi_neighbors

    if tau < dt or tau > t_values[-1] / 2:
        return 1e10

    augmented_states, increments = [], []
    dim = trajectories.shape[2]

    for traj in trajectories:
        interp_funcs = [interp1d(t_values, traj[:, d], kind='cubic', fill_value="extrapolate") for d in range(dim)]
        start_idx, end_idx = int(np.ceil(tau / dt)), len(t_values) - 2

        for i in range(start_idx, end_idx + 1):
            t_curr, x_curr = t_values[i], traj[i]
            x_del = np.array([f(t_curr - tau) for f in interp_funcs])
            augmented_states.append(np.concatenate([x_curr, x_del]))
            increments.append(traj[i+1] - traj[i])

    Z, dX = np.array(augmented_states), np.array(increments)
    if len(Z) < 50:
        return 1e10

    total_mi = 0
    for d in range(dX.shape[1]):
        mi = mutual_info_regression(Z, dX[:, d], n_neighbors=mi_neighbors, random_state=42)
        total_mi += mi[0]
    return -total_mi

def run_bayesian_optimization(data_path, out_dir):
    global trajectories, t_values, dt, mi_neighbors
    trajectories, t_values, dt = load_sdde_data_from_npz(data_path)
    mi_neighbors = 10

    res = gp_minimize(
        func=objective_function_mi,
        dimensions=[(0.1, 2.5)],
        acq_func="EI",
        n_calls=30,
        n_random_starts=15,
        random_state=41,
        verbose=False,
        noise="gaussian"
    )
    best_tau = res.x[0]
    print(f"\n✅ Optimized tau = {best_tau:.4f}")
    return best_tau, trajectories, t_values, dt

# ---------------------- DELAY RECONSTRUCTION ----------------------
def reconstruct_2d_delay_matrix(x_matrix, y_matrix, t_values, dt, best_tau):
    n_traj, n_steps, _ = x_matrix.shape
    tau_step = int(best_tau / dt)

    x_curr, x_del, y_curr, y_del, x_next, y_next = [], [], [], [], [], []
    for i in range(n_traj):
        x_t = x_matrix[i, :, 0]
        y_t = y_matrix[i, :, 0]
        for j in range(n_steps - 1):
            di = max(0, j - tau_step)
            x_curr.append([x_t[j]])
            x_del.append([x_t[di]])
            y_curr.append([y_t[j]])
            y_del.append([y_t[di]])
            x_next.append([x_t[j+1]])
            y_next.append([y_t[j+1]])

    return (np.array(x_curr), np.array(x_del), np.array(y_curr), np.array(y_del),
            np.array(x_next), np.array(y_next))

def preprocess_2d(xc, xd, yc, yd, xn, yn):
    nexts = np.hstack([xn, yn])
    idx = np.arange(len(xc))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    train = (xc[tr], xd[tr], yc[tr], yd[tr], nexts[tr])
    test = (xc[te], xd[te], yc[te], yd[te], nexts[te])
    return train, test

# ---------------------- TRAINING ----------------------
def train_2d_model(train_data, test_data, step_size, out_dir):
    xct, xdt, yct, ydt, nst = train_data

    base = ModelBuilder.define_gaussian_process(2, 3, 64, "model")
    model = SDDEApproximationNetwork(base, step_size)
    model.compile(optimizer=Adam(1e-3))

    cp = ModelCheckpoint(os.path.join(out_dir, "best.h5"), save_best_only=True, save_weights_only=True)
    rlr = ReduceLROnPlateau(factor=0.5, patience=30, min_lr=1e-6)

    model.fit(
        x=[xct, xdt, yct, ydt, nst],
        epochs=500,
        batch_size=256,
        validation_split=0.1,
        callbacks=[cp, rlr]
    )
    model.load_weights(os.path.join(out_dir, "best.h5"))
    return model

# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    DATA_PATH = r"E:\2d\2d_sdde_data.npz"
    OUTPUT_DIR = r"E:\2d"
    STEP_SIZE = 0.01
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # AUTO GET BEST TAU FROM BAYESIAN OPTIMIZATION
    best_tau, x_mat, t_vals, dt = run_bayesian_optimization(DATA_PATH, OUTPUT_DIR)

    # LOAD FULL DATA
    data = np.load(DATA_PATH)
    x_mat = data["x_matrix"]
    y_mat = data["y_matrix"]
    t_vals = data["t_values"]

    # RECONSTRUCT DELAY
    xc, xd, yc, yd, xn, yn = reconstruct_2d_delay_matrix(x_mat, y_mat, t_vals, dt, best_tau)
    train, test = preprocess_2d(xc, xd, yc, yd, xn, yn)

    # TRAIN
    model = train_2d_model(train, test, STEP_SIZE, OUTPUT_DIR)
    print("\n✅ Training completed!")

