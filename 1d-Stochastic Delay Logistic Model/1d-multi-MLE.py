import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os

tfd = tfp.distributions
tf.get_logger().setLevel('ERROR')


class SDDEIntegrators:
    @staticmethod
    def euler_maruyama(xn, xn_tau, alpha, h, f_sigma, rng):
        dW = rng.normal(loc=0, scale=np.sqrt(h), size=xn.shape)
        xk = xn.reshape(1, -1)
        xk_tau = xn_tau.reshape(1, -1)
        alpha_tensor = tf.convert_to_tensor([[alpha]], dtype=tf.float32)
        fk, sk = f_sigma([xk, xk_tau, alpha_tensor])
        return xk + h * fk + sk * dW


class ModelBuilder:
    @staticmethod
    def define_gaussian_process(n_input_dimensions, n_output_dimensions, n_layers, n_dim_per_layer, name,
                                activation="tanh"):
        input_alpha = Input((1,), dtype=tf.float32, name=name + '_alpha')
        input_current = Input((n_input_dimensions,), dtype=tf.float32, name=name + '_current')
        input_delay = Input((n_input_dimensions,), dtype=tf.float32, name=name + '_delay')

        merged = Concatenate()([input_alpha, input_current, input_delay])
        x = merged
        for _ in range(n_layers):
            x = Dense(n_dim_per_layer, activation=activation, dtype=tf.float32)(x)

        drift_output = Dense(n_output_dimensions, dtype=tf.float32)(x)
        diffusion_output = Dense(n_output_dimensions,
                                 activation=lambda x: tf.nn.softplus(x) + 1e-13,
                                 dtype=tf.float32)(x)

        return Model(inputs=[input_alpha, input_current, input_delay],
                     outputs=[drift_output, diffusion_output])


class SDDEApproximationNetwork(tf.keras.Model):
    def __init__(self, sde_model, step_size, method="euler", **kwargs):
        super().__init__(**kwargs)
        self.sde_model = sde_model
        self.step_size = tf.cast(step_size, dtype=tf.float32)
        self.method = method

    def call(self, inputs):
        alpha, x_n, x_n_tau, x_np1 = inputs

        alpha = tf.cast(alpha, tf.float32)
        x_n = tf.cast(x_n, tf.float32)
        x_n_tau = tf.cast(x_n_tau, tf.float32)
        x_np1 = tf.cast(x_np1, tf.float32)

        if self.method == "euler":
            drift, diffusion = self.sde_model([alpha, x_n, x_n_tau])
            drift = tf.cast(drift, tf.float32)
            diffusion = tf.cast(diffusion, tf.float32)

            mean = x_n + self.step_size * drift
            std = tf.math.sqrt(self.step_size) * diffusion
            dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)

            log_prob = dist.log_prob(x_np1)
            loss = -tf.reduce_mean(log_prob)
            self.add_loss(loss)
            self.add_metric(loss, name="nll", aggregation="mean")

        return self.sde_model([alpha, x_n, x_n_tau])


def reconstruct_delay_matrix(all_trajectories, alpha_list, found_tau, dt):
    tau_steps = int(found_tau / dt)
    new_x_current, new_x_delay, new_alpha_values, new_x_next = [], [], [], []

    for alpha, traj_batch in zip(alpha_list, all_trajectories):
        for traj in traj_batch:
            start_idx = tau_steps
            end_idx = traj.shape[0] - 2
            for t_idx in range(start_idx, end_idx + 1):
                new_x_current.append(traj[t_idx])
                new_x_delay.append(traj[t_idx - tau_steps])
                new_alpha_values.append(alpha)
                new_x_next.append(traj[t_idx + 1])

    return (np.array(new_x_current, dtype=np.float32),
            np.array(new_x_delay, dtype=np.float32),
            np.array(new_alpha_values, dtype=np.float32).reshape(-1, 1),
            np.array(new_x_next, dtype=np.float32))


def preprocess_data(x_current, x_delay, x_next, alpha_values, test_size=0.2, random_state=42):
    X = np.hstack([alpha_values, x_current, x_delay])
    y = x_next
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    n_dim = x_current.shape[1]
    alpha_train = X_train[:, :1]
    x_current_train = X_train[:, 1:1 + n_dim]
    x_delay_train = X_train[:, 1 + n_dim:]

    alpha_test = X_test[:, :1]
    x_current_test = X_test[:, 1:1 + n_dim]
    x_delay_test = X_test[:, 1 + n_dim:]

    return alpha_train, x_current_train, x_delay_train, y_train, \
           alpha_test, x_current_test, x_delay_test, y_test, n_dim


def train_and_evaluate_model(alpha_train, x_current_train, x_delay_train, y_train,
                             alpha_test, x_current_test, x_delay_test, y_test,
                             n_dimensions, step_size, epochs=300, batch_size=128,
                             model_save_path=None):
    model = ModelBuilder.define_gaussian_process(
        n_input_dimensions=n_dimensions,
        n_output_dimensions=n_dimensions,
        n_layers=2,
        n_dim_per_layer=32,
        name="multi_alpha_sdde_model",
        activation="tanh"
    )

    sdde_approximator = SDDEApproximationNetwork(model, step_size, method="euler")
    sdde_approximator.compile(optimizer=Adam(learning_rate=0.001))
    callbacks = []

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        cp = tf.keras.callbacks.ModelCheckpoint(
            model_save_path, monitor='val_nll', save_best_only=True,
            save_weights_only=True, mode='min', verbose=1
        )
        callbacks.append(cp)

    history = sdde_approximator.fit(
        [alpha_train, x_current_train, x_delay_train, y_train],
        epochs=epochs, batch_size=batch_size, verbose=1,
        validation_split=0.1, callbacks=callbacks
    )

    if model_save_path and os.path.exists(model_save_path):
        sdde_approximator.load_weights(model_save_path)

    test_loss = sdde_approximator.evaluate(
        [alpha_test, x_current_test, x_delay_test, y_test], verbose=0
    )
    print(f"Test NLL loss: {test_loss[0]:.4f}")

    return sdde_approximator, history


if __name__ == "__main__":
    # Paths
    DATA_PATH = r"E:\1md\multi_alpha_sdde_data.npz"
    MODEL_SAVE_PATH = r"E:\1md\multi_alpha_sdde_model_best.h5"
    PLOT_SAVE_DIR = r"E:\1md"

    # Hyperparameters

    EPOCHS = 300
    BATCH_SIZE = 128
    TEST_SIZE = 0.2


    # Load data
    x_next, t_values, dt, found_tau, all_trajectories, alpha_list = load_multi_alpha_data_and_bo_result(
        data_path=DATA_PATH,
        known_tau=KNOWN_TAU
    )

    # Reconstruct delay samples using identified tau
    new_x_current, new_x_delay, new_alpha_values, new_x_next = reconstruct_delay_matrix(
        all_trajectories, alpha_list, found_tau, dt
    )

    # Train/test split
    (alpha_train, x_current_train, x_delay_train, y_train,
     alpha_test, x_current_test, x_delay_test, y_test,
     n_dim) = preprocess_data(
        new_x_current, new_x_delay, new_x_next, new_alpha_values,
        test_size=TEST_SIZE, random_state=42
    )

    # Train model
    sdde_approximator, history = train_and_evaluate_model(
        alpha_train, x_current_train, x_delay_train, y_train,
        alpha_test, x_current_test, x_delay_test, y_test,
        n_dimensions=n_dim,
        step_size=dt,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_path=MODEL_SAVE_PATH
    )
