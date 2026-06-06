import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import json

# ---------------------- GLOBAL SETTINGS ----------------------
tf.keras.backend.set_floatx('float64')
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
tf.random.set_seed(42)
tfd = tfp.distributions

# ---------------------- PATHS ----------------------
INPUT_DATA_PATH = r"E:\2md\2d_multi_param_sdde_data.npz"
OUTPUT_DIR = r"E:\2md"
MODEL_SAVE_DIR = r"E:\2md"

# ---------------------- AUTO GET BEST TAU ----------------------
def get_best_tau_from_json(output_dir):
    files = [f for f in os.listdir(output_dir) if f.startswith("multi_param_summary_result") and f.endswith(".json")]
    if not files:
        raise FileNotFoundError("No BO summary JSON found!")
    files.sort(reverse=True)
    latest = files[0]
    with open(os.path.join(output_dir, latest), 'r') as f:
        data = json.load(f)
    return data["best_tau"]

BEST_TAU = get_best_tau_from_json(OUTPUT_DIR)
print(f"Optimal tau loaded automatically: {BEST_TAU:.4f}")

# ---------------------- DATA LOADING ----------------------
def load_full_trajectory_data(data_path):
    data = np.load(data_path)
    x_matrix = data["x_matrix"]
    y_matrix = data["y_matrix"]
    t_values = data["t_values"]
    alpha_values = data["alpha_values"]
    dt = t_values[1] - t_values[0]
    return x_matrix, y_matrix, t_values, alpha_values, dt

def reconstruct_delay_matrix(x_matrix, y_matrix, t_values, alpha_values, best_tau, dt):
    tau_steps = int(best_tau / dt)
    n_alpha, n_traj, n_time, _ = x_matrix.shape

    all_alpha, all_xc, all_yc, all_xd, all_yd, all_xn, all_yn = [], [], [], [], [], [], []

    for p, alpha in enumerate(alpha_values):
        for i in range(n_traj):
            x_t = x_matrix[p, i].reshape(-1)
            y_t = y_matrix[p, i].reshape(-1)
            start = tau_steps
            end = n_time - 2
            for j in range(start, end + 1):
                dj = j - tau_steps
                all_alpha.append(alpha)
                all_xc.append(x_t[j])
                all_yc.append(y_t[j])
                all_xd.append(x_t[dj])
                all_yd.append(y_t[dj])
                all_xn.append(x_t[j+1])
                all_yn.append(y_t[j+1])

    return (
        np.array(all_alpha).reshape(-1,1),
        np.array(all_xc).reshape(-1,1),
        np.array(all_yc).reshape(-1,1),
        np.array(all_xd).reshape(-1,1),
        np.array(all_yd).reshape(-1,1),
        np.array(all_xn).reshape(-1,1),
        np.array(all_yn).reshape(-1,1)
    )

# ---------------------- MODEL BUILDER ----------------------
class ModelBuilder:
    @staticmethod
    def define_multi_param_model(n_dim=2, n_layers=3, n_units=30, activation='tanh'):
        inp_a = Input((1,), dtype=tf.float64)
        inp_xc = Input((1,), dtype=tf.float64)
        inp_yc = Input((1,), dtype=tf.float64)
        inp_xd = Input((1,), dtype=tf.float64)
        inp_yd = Input((1,), dtype=tf.float64)

        merged = Concatenate()([inp_a, inp_xc, inp_yc, inp_xd, inp_yd])
        x = merged
        for _ in range(n_layers):
            x = Dense(n_units, activation=activation, dtype=tf.float64)(x)

        drift = Dense(n_dim, dtype=tf.float64)(x)
        diff = Dense(n_dim, activation=lambda x: tf.nn.softplus(x)+1e-13, dtype=tf.float64)(x)
        return Model([inp_a, inp_xc, inp_yc, inp_xd, inp_yd], [drift, diff])

class SDDEMultiParamApproximator(tf.keras.Model):
    def __init__(self, sde_model, step_size, **kwargs):
        super().__init__(**kwargs)
        self.sde_model = sde_model
        self.step_size = tf.cast(step_size, tf.float64)
        self.nll_metric = tf.keras.metrics.Mean(name='nll')

    def call(self, inputs, training=None):
        return self.sde_model(inputs)

    def train_step(self, data):
        feat, lab = data
        a, xn, yn, xt, yt = feat
        xnp1, ynp1 = lab

        with tf.GradientTape() as tape:
            drift, diff = self.sde_model([a, xn, yn, xt, yt], training=True)
            curr = tf.concat([xn, yn], axis=1)
            nxt = tf.concat([xnp1, ynp1], axis=1)
            mean = curr + self.step_size * drift
            std = tf.sqrt(self.step_size) * diff
            dist = tfd.MultivariateNormalDiag(mean, std)
            loss = -tf.reduce_mean(dist.log_prob(nxt))

        grads = tape.gradient(loss, self.sde_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.sde_model.trainable_variables))
        self.nll_metric.update_state(loss)
        return {'nll': self.nll_metric.result()}

    def test_step(self, data):
        feat, lab = data
        a, xn, yn, xt, yt = feat
        xnp1, ynp1 = lab

        drift, diff = self.sde_model([a, xn, yn, xt, yt], training=False)
        curr = tf.concat([xn, yn], axis=1)
        nxt = tf.concat([xnp1, ynp1], axis=1)
        mean = curr + self.step_size * drift
        std = tf.sqrt(self.step_size) * diff
        dist = tfd.MultivariateNormalDiag(mean, std)
        loss = -tf.reduce_mean(dist.log_prob(nxt))
        self.nll_metric.update_state(loss)
        return {'nll': self.nll_metric.result()}

    def reset_metrics(self):
        self.nll_metric.reset_states()

# ---------------------- PREPROCESS & TRAIN ----------------------
def preprocess_data(alpha, xc, yc, xd, yd, xn, yn, test_size=0.2):
    idx = np.arange(len(alpha))
    tr, te = train_test_split(idx, test_size=test_size, random_state=42)
    def s(a,i): return a[i]
    tr_feat = (s(alpha,tr),s(xc,tr),s(yc,tr),s(xd,tr),s(yd,tr))
    tr_lab  = (s(xn,tr),s(yn,tr))
    te_feat = (s(alpha,te),s(xc,te),s(yc,te),s(xd,te),s(yd,te))
    te_lab  = (s(xn,te),s(yn,te))
    return (tr_feat, tr_lab), (te_feat, te_lab)

def train_model(train_data, test_data, step_size, epochs=300, batch_size=256, save_dir='models'):
    tr_feat, tr_lab = train_data
    te_feat, te_lab = test_data

    base = ModelBuilder.define_multi_param_model()
    model = SDDEMultiParamApproximator(base, step_size)
    model.compile(optimizer=Adam(1e-3))

    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f'best_model_tau_{BEST_TAU:.4f}_{ts}.h5')

    class SaveCallback(tf.keras.callbacks.Callback):
        def __init__(self, m, p):
            self.m = m; self.p = p; self.best = float('inf')
        def on_epoch_end(self, e, logs=None):
            v = logs.get('val_nll')
            if v < self.best:
                self.best = v
                self.m.save_weights(self.p)

    cb = SaveCallback(base, path)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_nll', patience=30, mode='min')

    hist = model.fit(tr_feat, tr_lab, epochs=epochs, batch_size=batch_size,
                     validation_data=(te_feat, te_lab), callbacks=[cb, es], verbose=1)
    base.load_weights(path)
    return model, base, hist, path

# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    x_mat, y_mat, t_vals, alphas, dt = load_full_trajectory_data(INPUT_DATA_PATH)
    alpha, xc, yc, xd, yd, xn, yn = reconstruct_delay_matrix(x_mat, y_mat, t_vals, alphas, BEST_TAU, dt)
    train_data, test_data = preprocess_data(alpha, xc, yc, xd, yd, xn, yn)
    model, base, hist, path = train_model(train_data, test_data, dt, epochs=300, batch_size=256, save_dir=MODEL_SAVE_DIR)
    print(f"\nTraining done! Best model saved to: {path}")