#BO
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.feature_selection import mutual_info_regression
from skopt import gp_minimize
from skopt.plots import plot_convergence
import time
import warnings
import os
import json
import pickle


plt.rcParams["font.family"] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=FutureWarning)




def load_sdde_data_from_npz(npz_path):

    try:

        data = np.load(npz_path)
        x_matrix = data["x_matrix"]
        t_values = data["t_values"]
        dt = t_values[1] - t_values[0]


        if x_matrix.ndim == 2:
            x_matrix = x_matrix[:, :, np.newaxis]


        return x_matrix, t_values, dt
    except FileNotFoundError:
        print(f"错误：文件未找到！请检查路径 '{npz_path}' 是否正确。")
        return None, None, None
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return None, None, None


def objective_function_mi(tau):

    tau = tau[0]


    global trajectories, t_values, dt, mi_neighbors


    if tau < dt or tau > t_values[-1] / 2:
        return 1e10

    # 准备增广状态Z(t) = [X(t), X(t-tau)] 和增量 dX(t)
    augmented_states, increments = [], []
    dim = trajectories.shape[2]  # 正确获取数据维度

    for traj in trajectories:
        interp_funcs = [interp1d(t_values, traj[:, d], kind='cubic', fill_value="extrapolate") for d in range(dim)]
        start_idx, end_idx = int(np.ceil(tau / dt)), len(t_values) - 2
        for i in range(start_idx, end_idx + 1):
            t_current, x_current = t_values[i], traj[i]
            x_delayed = np.array([f(t_current - tau) for f in interp_funcs])

            z = np.concatenate([x_current, x_delayed])
            augmented_states.append(z)

            dx = traj[i + 1] - traj[i]
            increments.append(dx)

    Z, dX = np.array(augmented_states), np.array(increments)
    if len(Z) < 50: return 1e10

    # 计算 Z 和 dX 各个分量之间的互信息并求和
    total_mi = 0
    for d in range(dX.shape[1]):  # 循环遍历 dX 的所有维度 (1D或2D)
        mi_score = mutual_info_regression(Z, dX[:, d], n_neighbors=mi_neighbors, random_state=42)
        total_mi += mi_score[0]

    # 返回负的互信息，因为优化器是最小化
    objective_value = -total_mi
    return objective_value


# --- 主程序---
if __name__ == "__main__":

    INPUT_DATA_PATH = r"E:\2D-BO-MI\2d_sdde_data.npz"
    OUTPUT_DIR = r"E:\2D-BO-MI"

    # 超参数
    MI_NEIGHBORS = 15
    N_CALLS = 30
    N_RANDOM_STARTS = 10
    SEARCH_SPACE = [(0.1, 2.5)]

    # --- 执行优化 ---
    trajectories, t_values, dt = load_sdde_data_from_npz(INPUT_DATA_PATH)
    mi_neighbors = MI_NEIGHBORS

    if trajectories is not None:

        start_time = time.time()

        result = gp_minimize(
            func=objective_function_mi,
            dimensions=SEARCH_SPACE,
            acq_func="EI",
            n_calls=N_CALLS,
            n_random_starts=N_RANDOM_STARTS,
            random_state=41,
            verbose=False,
            noise="gaussian"
        )

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        print(f"\n--- 优化结束 (耗时: {duration:.2f} 秒) ---")

        # --- 结果汇总与保存 ---
        found_tau = result.x[0]
        max_mi = -result.fun

        print(f"\n[最终结果] 识别出的最优 Tau: {found_tau:.4f}")
        print(f"           对应的最大互信息值 (MI): {max_mi:.6f}")

        run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        pickle_path = os.path.join(OUTPUT_DIR, f'optimization_result_{run_timestamp}.pkl')
        json_path = os.path.join(OUTPUT_DIR, f'summary_result_{run_timestamp}.json')

        with open(pickle_path, 'wb') as f: pickle.dump(result, f)
        print(f"\n[保存成功] 完整结果已保存到: {pickle_path}")

        summary_data = {
            'best_tau': float(found_tau),
            'max_mutual_information': float(max_mi),
            'min_objective_value': float(result.fun),
            'hyperparameters': {
                'mi_neighbors': MI_NEIGHBORS, 'n_calls': N_CALLS, 'n_random_starts': N_RANDOM_STARTS,
                'search_space': SEARCH_SPACE, 'input_data_path': INPUT_DATA_PATH
            },
            'run_details': {'timestamp': run_timestamp, 'duration_seconds': duration},
            'evaluation_history': {
                'taus_evaluated': np.array(result.x_iters).ravel().tolist(),
                'objective_values': np.array(result.func_vals).tolist()  # 确保转换为Python list
            }
        }
        with open(json_path, 'w', encoding='utf-8') as f: json.dump(summary_data, f, indent=4, ensure_ascii=False)
        print(f"[保存成功] 摘要已保存到: {json_path}")

        # ---生成并保存本次运行的图表 ---


        plt.figure(figsize=(6, 5))

        plot_convergence(result)
        plt.gca().set_title('')
        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel("Objective Function Min Value", fontsize=15)  # 更新Y轴标签
        plt.tick_params(labelsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(r"E:2D-BO-MI\loss.png", dpi=100, bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(6, 5))
        evaluated_taus = np.array(result.x_iters).flatten()
        evaluated_errors = result.func_vals
        min_idx = np.argmin(evaluated_errors)
        found_tau = evaluated_taus[min_idx]

        plt.scatter(evaluated_taus, evaluated_errors, c='skyblue', s=80, alpha=0.8, edgecolors='navy',  # 统一蓝色
                    label='BO Evaluation point')

        plt.scatter(evaluated_taus[min_idx], evaluated_errors[min_idx],
                    c='red', s=200, marker='*', edgecolors='darkred',
                    label=f'Identified τ = {found_tau:.4f}')

        plt.xlabel(' τ', fontsize=15)
        plt.ylabel('-J(τ)', fontsize=15)
        plt.tick_params(labelsize=14)
        plt.legend(frameon=True, fontsize=13.5)

        plt.savefig(r"E:\2D-BO-MI\tau_loss_scatter.png", dpi=300, bbox_inches="tight")
        plt.show()

#MLE
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp  # 仅保留基础导入
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


class ModelBuilder:

    @staticmethod
    def define_gaussian_process(n_dimensions, n_layers, n_dim_per_layer, name, activation="tanh"):


        input_x_current = Input((1,), dtype=tf.float64, name=f"{name}_x_current")
        input_x_delay = Input((1,), dtype=tf.float64, name=f"{name}_x_delay")
        input_y_current = Input((1,), dtype=tf.float64, name=f"{name}_y_current")
        input_y_delay = Input((1,), dtype=tf.float64, name=f"{name}_y_delay")


        merged = Concatenate()([input_x_current, input_x_delay, input_y_current, input_y_delay])


        x = merged
        for i in range(n_layers):
            x = Dense(n_dim_per_layer, activation=activation, dtype=tf.float64)(x)


        drift_output = Dense(n_dimensions, dtype=tf.float64, name=f"{name}_drift")(x)  # [drift_x, drift_y]
        diffusion_output = Dense(n_dimensions,
                                 activation=lambda x: tf.nn.softplus(x) + 1e-13,
                                 dtype=tf.float64,
                                 name=f"{name}_diffusion")(x)  # [diffusion_x, diffusion_y]

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


        if self.method == "euler":

            drift, diffusion = self.sde_model([x_current, x_delay, y_current, y_delay])


            current_state = tf.concat([x_current, y_current], axis=1)  # [x_t, y_t]
            step_size_tensor = tf.cast(self.step_size, dtype=diffusion.dtype)
            mean = current_state + step_size_tensor * drift
            std = tf.math.sqrt(step_size_tensor) * diffusion


            dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)


            log_prob = dist.log_prob(next_state)
            loss = -tf.reduce_mean(log_prob)


            self.add_loss(loss)
            self.add_metric(loss, name="nll", aggregation="mean")

        return [drift, diffusion]

# -------------------------- 数据加载与延迟矩阵重构 --------------------------
def get_best_tau():

    best_tau = 1.0089
    return best_tau


def load_raw_2d_data(raw_data_path):

    data = np.load(raw_data_path)


    x_matrix = data['x_matrix']  # shape: (n_traj, n_steps, 1)
    y_matrix = data['y_matrix']  # shape: (n_traj, n_steps, 1)
    t_values = data['t_values']  # 非负时间轴
    dt = t_values[1] - t_values[0]


    return x_matrix, y_matrix, t_values, dt


def reconstruct_2d_delay_matrix(x_matrix, y_matrix, t_values, dt, best_tau):

    n_traj, n_steps, _ = x_matrix.shape
    tau_step = int(best_tau / dt)


    all_x_current = []
    all_x_delay = []
    all_y_current = []
    all_y_delay = []
    all_x_next = []
    all_y_next = []

    for traj_idx in range(n_traj):
        x_traj = x_matrix[traj_idx, :, 0]
        y_traj = y_matrix[traj_idx, :, 0]


        for step_idx in range(n_steps - 1):

            delay_idx = max(0, step_idx - tau_step)


            x_current = x_traj[step_idx]
            x_delay = x_traj[delay_idx]
            y_current = y_traj[step_idx]
            y_delay = y_traj[delay_idx]
            x_next = x_traj[step_idx + 1]
            y_next = y_traj[step_idx + 1]


            all_x_current.append([x_current])
            all_x_delay.append([x_delay])
            all_y_current.append([y_current])
            all_y_delay.append([y_delay])
            all_x_next.append([x_next])
            all_y_next.append([y_next])


    x_current = np.array(all_x_current, dtype=np.float64)
    x_delay = np.array(all_x_delay, dtype=np.float64)
    y_current = np.array(all_y_current, dtype=np.float64)
    y_delay = np.array(all_y_delay, dtype=np.float64)
    x_next = np.array(all_x_next, dtype=np.float64)
    y_next = np.array(all_y_next, dtype=np.float64)


    return x_current, x_delay, y_current, y_delay, x_next, y_next


def preprocess_2d_data(x_current, x_delay, y_current, y_delay, x_next, y_next, test_size=0.2, random_state=42):

    next_state = np.hstack([x_next, y_next])  # shape: (n_samples, 2)


    indices = np.arange(len(x_current))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state, shuffle=True)


    train_data = (
        x_current[train_idx], x_delay[train_idx],
        y_current[train_idx], y_delay[train_idx],
        next_state[train_idx]
    )
    test_data = (
        x_current[test_idx], x_delay[test_idx],
        y_current[test_idx], y_delay[test_idx],
        next_state[test_idx]
    )


    return train_data, test_data


# -------------------------- 模型训练与评估 --------------------------
def train_2d_sdde_model(train_data, test_data, step_size, output_dir, epochs=500, batch_size=256):

    x_current_train, x_delay_train, y_current_train, y_delay_train, next_state_train = train_data
    x_current_test, x_delay_test, y_current_test, y_delay_test, next_state_test = test_data


    base_model = ModelBuilder.define_gaussian_process(
        n_dimensions=2,
        n_layers=3,
        n_dim_per_layer=64,
        name="2d_sdde_model",
        activation="tanh"
    )


    sdde_approximator = SDDEApproximationNetwork(
        sde_model=base_model,
        step_size=step_size,
        method="euler"
    )


    sdde_approximator.compile(optimizer=Adam(learning_rate=0.001))


    model_save_path = os.path.join(output_dir, "2d_sdde_model_best.h5")
    os.makedirs(output_dir, exist_ok=True)

    callbacks = [

        ModelCheckpoint(
            model_save_path,
            monitor='val_nll',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='min'
        ),

        ReduceLROnPlateau(
            monitor='val_nll',
            factor=0.5,
            patience=30,
            min_lr=1e-6,
            verbose=1
        )
    ]


    history = sdde_approximator.fit(
        x=[x_current_train, x_delay_train, y_current_train, y_delay_train, next_state_train],
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.1,
        callbacks=callbacks,
        shuffle=True
    )

    sdde_approximator.load_weights(model_save_path)



    test_loss = sdde_approximator.evaluate(
        x=[x_current_test, x_delay_test, y_current_test, y_delay_test, next_state_test],
        verbose=0
    )


    return sdde_approximator, history, base_model


# -------------------------- 主程序入口 --------------------------
if __name__ == "__main__":

    RAW_DATA_PATH = r"E:\2D-BO-MI\2d_sdde_data.npz"
    OUTPUT_DIR = r"E:\2D-BO-MI\2d单参数\trained_model"
    STEP_SIZE = 0.01
    EPOCHS = 500
    BATCH_SIZE = 256

    # ====================== 执行流程 ======================
    try:

        best_tau = get_best_tau()


        x_matrix, y_matrix, t_values, dt = load_raw_2d_data(RAW_DATA_PATH)


        x_current, x_delay, y_current, y_delay, x_next, y_next = reconstruct_2d_delay_matrix(
            x_matrix, y_matrix, t_values, dt, best_tau
        )


        train_data, test_data = preprocess_2d_data(
            x_current, x_delay, y_current, y_delay, x_next, y_next
        )


        sdde_approximator, history, base_model = train_2d_sdde_model(
            train_data, test_data, STEP_SIZE, OUTPUT_DIR, EPOCHS, BATCH_SIZE
        )


        plot_history_path = os.path.join(OUTPUT_DIR, 'training_history.png')
        plot_training_history(history, plot_history_path)



    except Exception as e:

        raise


# ======================  模型加载函数 ======================
def load_sdde_model(n_dimensions, step_size, model_weights_path):

    model = ModelBuilder.define_gaussian_process(
        n_dimensions=n_dimensions,
        n_layers=3,
        n_dim_per_layer=64,
        name="sdde_model",
        activation="tanh"
    )

    sdde_approximator = SDDEApproximationNetwork(
        sde_model=model,
        step_size=step_size,
        method="euler"
    )


    sdde_approximator.compile(optimizer=Adam(learning_rate=0.001))


    if not sdde_approximator.built:
        if n_dimensions == 1:
            dummy_input = [
                tf.random.uniform(shape=(1, 1), dtype=tf.float64),
                tf.random.uniform(shape=(1, 1), dtype=tf.float64),
                tf.random.uniform(shape=(1, 1), dtype=tf.float64)
            ]
        else:
            dummy_input = [
                tf.random.uniform(shape=(1, 1), dtype=tf.float64),  # x_current
                tf.random.uniform(shape=(1, 1), dtype=tf.float64),  # x_delay
                tf.random.uniform(shape=(1, 1), dtype=tf.float64),  # y_current
                tf.random.uniform(shape=(1, 1), dtype=tf.float64),  # y_delay
                tf.random.uniform(shape=(1, 2), dtype=tf.float64)  # next_state
            ]
        _ = sdde_approximator(dummy_input)


    sdde_approximator.load_weights(model_weights_path)

    return sdde_approximator
model_weights_path = r"E:\2D-BO-MI\2d单参数\trained_model\2d_sdde_model_best.h5"
loaded_model = load_sdde_model(
    n_dimensions=2,
    step_size=0.01,
    model_weights_path=model_weights_path
)


#可视化
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def true_2d_sdde(t, x, y, x_delay, y_delay):

    drift_x = 1.6 * x * (1 - x_delay) + 0.1 * x * y_delay
    drift_y = 2 * y * (1 - y_delay) + 0.1 * y * x_delay
    diffusion_x = 0.1 * x
    diffusion_y = 0.1 * y


    return (drift_x, drift_y), (diffusion_x, diffusion_y)



def simulate_2d_trajectory(sde_model, true_sdde, initial_condition, tau_true, tau_pred, T, h, rng):

    max_tau = max(tau_true, tau_pred)
    num_steps = int((T + max_tau) / h)
    t_values = np.linspace(-max_tau, T, num_steps + 1)
    start_index = int(max_tau / h)
    n_dimensions = 2


    shared_noise = rng.normal(loc=0, scale=np.sqrt(h), size=(num_steps, n_dimensions))


    initial_condition = np.asarray(initial_condition).flatten()


    true_trajectory = np.zeros((num_steps + 1, n_dimensions))
    pred_trajectory = np.zeros_like(true_trajectory)
    true_trajectory[0] = initial_condition
    pred_trajectory[0] = initial_condition


    delay_steps_true = int(tau_true / h)
    delay_steps_pred = int(tau_pred / h)


    for i in range(1, num_steps + 1):

        delay_index_true = max(0, i - delay_steps_true)
        delay_index_pred = max(0, i - delay_steps_pred)


        if t_values[i] <= 0:
            true_trajectory[i] = initial_condition
            pred_trajectory[i] = initial_condition
        else:

            x_true, y_true = true_trajectory[i - 1]
            x_delay_true, y_delay_true = true_trajectory[delay_index_true]
            (drift_x_true, drift_y_true), (diff_x_true, diff_y_true) = true_sdde(
                t=t_values[i], x=x_true, y=y_true, x_delay=x_delay_true, y_delay=y_delay_true
            )
            noise_x, noise_y = shared_noise[i - 1]
            true_trajectory[i] = [
                x_true + h * drift_x_true + diff_x_true * noise_x,
                y_true + h * drift_y_true + diff_y_true * noise_y
            ]


            x_current_pred, y_current_pred = pred_trajectory[i - 1]
            x_delay_pred, y_delay_pred = pred_trajectory[delay_index_pred]

            x_tensor = [
                tf.convert_to_tensor([[x_current_pred]], dtype=tf.float64),  # x_current
                tf.convert_to_tensor([[x_delay_pred]], dtype=tf.float64),  # x_delay
                tf.convert_to_tensor([[y_current_pred]], dtype=tf.float64),  # y_current
                tf.convert_to_tensor([[y_delay_pred]], dtype=tf.float64)  # y_delay
            ]
            drift_pred, diff_pred = sde_model(x_tensor)
            drift_x_pred, drift_y_pred = drift_pred.numpy().flatten()
            diff_x_pred, diff_y_pred = diff_pred.numpy().flatten()

            pred_trajectory[i] = [
                x_current_pred + h * drift_x_pred + diff_x_pred * noise_x,
                y_current_pred + h * drift_y_pred + diff_y_pred * noise_y
            ]

    return true_trajectory, pred_trajectory, t_values, start_index



color_style_config = [

    {
        "true_color": "#1f77b4",  # 真实轨迹：深蓝色实线
        "true_ls": "-",
        "pred_color": "#8ecae6",  # 预测轨迹：浅蓝色虚线
        "pred_ls": "--",
        "label": ""
    },

    {
        "true_color": "#9467bd",  # 真实轨迹：深紫色实线
        "true_ls": "-",
        "pred_color": "#d8bfd8",  # 预测轨迹：浅紫色虚线
        "pred_ls": "--",
        "label": ""
    },

    {
        "true_color": "#ff7f0e",  # 真实轨迹：深橙色实线
        "true_ls": "-",
        "pred_color": "#ffd700",  # 预测轨迹：金黄色虚线
        "pred_ls": "--",
        "label": ""
    }
]


# --------------------------合并绘图函数 --------------------------
def plot_multiple_trajectories(sde_model, true_sdde, initial_conditions, tau_true, tau_pred, T, h, rng, save_dir):


    if len(initial_conditions) != 3:
        raise ValueError("初始值数量必须为4组，与颜色配置匹配！")

    # 创建画布
    fig_x, ax_x = plt.subplots(figsize=(6, 5))
    fig_y, ax_y = plt.subplots(figsize=(6, 5))


    for idx, (x0, y0) in enumerate(initial_conditions):

        style = color_style_config[idx]
        style["label"] = f"$x_0={x0:.1f}$, $y_0={y0:.1f}$"
        print(f"处理初始值 {style['label']}...")


        true_traj, pred_traj, t_values, start_index = simulate_2d_trajectory(
            sde_model=sde_model,
            true_sdde=true_sdde,
            initial_condition=(x0, y0),
            tau_true=tau_true,
            tau_pred=tau_pred,
            T=T,
            h=h,
            rng=rng
        )


        t_plot = t_values[start_index:]


        ax_x.plot(t_plot, true_traj[start_index:, 0],
                  color=style["true_color"],
                  linestyle=style["true_ls"],
                  linewidth=2.0,
                  label=style["label"])
        ax_x.plot(t_plot, pred_traj[start_index:, 0],
                  color=style["pred_color"],
                  linestyle=style["pred_ls"],
                  linewidth=2.0)


        ax_y.plot(t_plot, true_traj[start_index:, 1],
                  color=style["true_color"],
                  linestyle=style["true_ls"],
                  linewidth=2.0,
                  label=style["label"])
        ax_y.plot(t_plot, pred_traj[start_index:, 1],
                  color=style["pred_color"],
                  linestyle=style["pred_ls"],
                  linewidth=2.0)


    ax_x.set_xlabel('t', fontsize=15)
    ax_x.set_ylabel('x(t)', fontsize=15)
    ax_x.tick_params(axis='both', labelsize=12)

    fig_x.tight_layout()
    fig_x.savefig(f"{save_dir}/combined_x_trajectories.png", dpi=300, bbox_inches='tight')


    ax_y.set_xlabel('t', fontsize=15)
    ax_y.set_ylabel('y(t)', fontsize=15)
    ax_y.tick_params(axis='both', labelsize=12)
    ax_y.legend(fontsize=14, frameon=False, loc='best')
    fig_y.tight_layout()
    fig_y.savefig(f"{save_dir}/combined_y_trajectories.png", dpi=300, bbox_inches='tight')

    plt.show()



# --------------------------  主程序入口 --------------------------
if __name__ == "__main__":
    # 配置参数
    save_directory = r'E:\2D-BO-MI\2d单参数'
    model_path = r"E:\2D-BO-MI\2d单参数\trained_model\2d_sdde_model_best.h5"


    initial_conditions = [
        (0.5, 1.5),
        (2.5, 1.0),
        (0.4, 0.8)
    ]


    tau_true = 1.0
    tau_pred = 1.0089
    T = 10
    h = 0.01
    seed = 42


    loaded_model = load_sdde_model(n_dimensions=2, step_size=h, model_weights_path=model_path)
    sde_model = loaded_model.sde_model


    rng = np.random.default_rng(seed=seed)


    plot_multiple_trajectories(
        sde_model=sde_model,
        true_sdde=true_2d_sdde,
        initial_conditions=initial_conditions,
        tau_true=tau_true,
        tau_pred=tau_pred,
        T=T,
        h=h,
        rng=rng,
        save_dir=save_directory
    )

    print("程序执行完成！")


#多参数BO+MLE
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.feature_selection import mutual_info_regression
from skopt import gp_minimize
from skopt.plots import plot_convergence
import time
import warnings
import os
import json
import pickle


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=FutureWarning)

trajectories = None
t_values = None
dt = None
mi_neighbors = None



def load_multi_param_sdde_data(npz_path):

    try:

        data = np.load(npz_path)
        x_matrix = data["x_matrix"]  # (n_alpha, n_traj, n_time, 1)
        y_matrix = data["y_matrix"]  # (n_alpha, n_traj, n_time, 1)
        t_values = data["t_values"]
        alpha_values = data["alpha_values"]

        dt = t_values[1] - t_values[0]
        n_alpha, n_traj, n_time, _ = x_matrix.shape


        global trajectories
        x_flat = x_matrix.reshape(-1, n_time, 1)  # (总轨迹数, n_time, 1)
        y_flat = y_matrix.reshape(-1, n_time, 1)  # (总轨迹数, n_time, 1)
        trajectories = np.concatenate([x_flat, y_flat], axis=2)  # 合并为2D轨迹


        return t_values, dt, alpha_values
    except FileNotFoundError:

        return None, None, None
    except Exception as e:

        return None, None, None


def objective_function_mi(tau):

    tau = tau[0]


    global trajectories, t_values, dt, mi_neighbors

    if tau < dt or tau > t_values[-1] / 2:
        return 1e10


    augmented_states, increments = [], []
    dim = trajectories.shape[2]

    for traj in trajectories:
        interp_funcs = [interp1d(t_values, traj[:, d], kind='cubic', fill_value="extrapolate")
                        for d in range(dim)]
        start_idx, end_idx = int(np.ceil(tau / dt)), len(t_values) - 2
        for i in range(start_idx, end_idx + 1):
            t_current = t_values[i]
            x_current = traj[i]  # (2,)
            x_delayed = np.array([f(t_current - tau) for f in interp_funcs])  # (2,)

            z = np.concatenate([x_current, x_delayed])  # (4,)
            augmented_states.append(z)

            dx = traj[i + 1] - traj[i]  # (2,)
            increments.append(dx)

    Z, dX = np.array(augmented_states), np.array(increments)
    if len(Z) < 50:
        return 1e10


    total_mi = 0
    for d in range(dim):
        mi_score = mutual_info_regression(Z, dX[:, d], n_neighbors=mi_neighbors, random_state=42)
        total_mi += mi_score[0]

    objective_value = -total_mi
    print(f"测试 tau = {tau:.4f}, 目标函数值 (-MI) = {objective_value:.6f}")
    return objective_value


# --- 主程序入口 ---
if __name__ == "__main__":

    INPUT_DATA_PATH = r"E:\2D-BO-MI\2d多参数\2d_multi_param_sdde_data.npz"
    OUTPUT_DIR = r"E:\2D-BO-MI\2d多参数"


    MI_NEIGHBORS = 20
    N_CALLS = 30
    N_RANDOM_STARTS = 15
    SEARCH_SPACE = [(0.1, 2.5)]


    t_values, dt, alpha_values = load_multi_param_sdde_data(INPUT_DATA_PATH)
    mi_neighbors = MI_NEIGHBORS

    if trajectories is not None:

        start_time = time.time()


        result = gp_minimize(
            func=objective_function_mi,
            dimensions=SEARCH_SPACE,
            acq_func="EI",
            n_calls=N_CALLS,
            n_random_starts=N_RANDOM_STARTS,
            random_state=111, #种子数
            verbose=False,
            noise="gaussian"
        )

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        print(f"\n--- 优化结束 (耗时: {duration:.2f} 秒) ---")


        found_tau = result.x[0]
        max_mi = -result.fun

        print(f"\n[最终结果] 识别出的最优 Tau: {found_tau:.4f}")
        print(f"           对应的最大互信息值 (MI): {max_mi:.6f}")

        run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        pickle_path = os.path.join(OUTPUT_DIR, f'multi_param_optimization_result_{run_timestamp}.pkl')
        json_path = os.path.join(OUTPUT_DIR, f'multi_param_summary_result_{run_timestamp}.json')

        with open(pickle_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"\n[保存成功] 完整结果已保存到: {pickle_path}")


        summary_data = {
            'best_tau': float(found_tau),
            'max_mutual_information': float(max_mi),
            'min_objective_value': float(result.fun),
            'hyperparameters': {
                'mi_neighbors': MI_NEIGHBORS, 'n_calls': N_CALLS, 'n_random_starts': N_RANDOM_STARTS,
                'search_space': SEARCH_SPACE, 'input_data_path': INPUT_DATA_PATH
            },
            'multi_param_info': {
                'alpha_values': alpha_values.tolist(),
                'total_trajectories': trajectories.shape[0],
                'trajectories_per_alpha': trajectories.shape[0] // len(alpha_values)
            },
            'run_details': {'timestamp': run_timestamp, 'duration_seconds': duration},
            'evaluation_history': {
                'taus_evaluated': np.array(result.x_iters).ravel().tolist(),
                'objective_values': np.array(result.func_vals).tolist()
            }
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        print(f"[保存成功] 摘要已保存到: {json_path}")


        print("\n--- 正在生成并保存图表 ---")

        # 收敛曲线
        plt.figure(figsize=(6, 5))
        plot_convergence(result)
        plt.gca().set_title('')
        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel("Objective Function Min Value", fontsize=15)  # 更新Y轴标签
        plt.tick_params(labelsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        conv_path = os.path.join(OUTPUT_DIR, f'multi_param_convergence_{run_timestamp}.png')
        plt.savefig(conv_path, dpi=100, bbox_inches="tight")
        plt.show()

        # Tau-目标函数散点图
        plt.figure(figsize=(6, 5))
        evaluated_taus = np.array(result.x_iters).flatten()
        evaluated_errors = result.func_vals
        min_idx = np.argmin(evaluated_errors)
        found_tau = evaluated_taus[min_idx]

        plt.scatter(evaluated_taus, evaluated_errors, c='skyblue', s=80, alpha=0.8, edgecolors='navy',  # 统一蓝色
                    label='BO Evaluation point')

        plt.scatter(evaluated_taus[min_idx], evaluated_errors[min_idx],
                    c='red', s=200, marker='*', edgecolors='darkred',
                    label=f'Identified τ = {found_tau:.4f}')

        plt.xlabel(' τ', fontsize=15)
        plt.ylabel('-J(τ)', fontsize=15)
        plt.tick_params(labelsize=14)
        plt.legend(frameon=True, fontsize=13.5)
        scatter_path = os.path.join(OUTPUT_DIR, f'multi_param_tau_loss_scatter_{run_timestamp}.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
        plt.show()



#MLE
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


tf.keras.backend.set_floatx('float64')
plt.rcParams['font.family'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
tf.random.set_seed(42)
tfd = tfp.distributions

BEST_TAU = 1.0069  # 贝叶斯优化识别出的最优tau值



def load_full_trajectory_data(data_path):

    data = np.load(data_path)
    x_matrix = data["x_matrix"]  # (n_alpha, n_traj, n_time, 1)
    y_matrix = data["y_matrix"]  # (n_alpha, n_traj, n_time, 1)
    t_values = data["t_values"]
    alpha_values = data["alpha_values"]

    dt = t_values[1] - t_values[0]
    n_alpha, n_traj, n_time, _ = x_matrix.shape

    return x_matrix, y_matrix, t_values, alpha_values, dt


def reconstruct_delay_matrix(x_matrix, y_matrix, t_values, alpha_values, best_tau, dt):


    tau_steps = int(best_tau / dt)
    n_alpha, n_traj, n_time, _ = x_matrix.shape


    all_alpha = []
    all_x_current = []
    all_y_current = []
    all_x_delay = []
    all_y_delay = []
    all_x_next = []
    all_y_next = []


    for p, alpha in enumerate(alpha_values):
        for i in range(n_traj):
            x_traj = x_matrix[p, i].reshape(-1)  # (n_time,)
            y_traj = y_matrix[p, i].reshape(-1)  # (n_time,)


            start_idx = tau_steps
            end_idx = n_time - 2

            for j in range(start_idx, end_idx + 1):
                delay_j = j - tau_steps

                all_alpha.append(alpha)
                all_x_current.append(x_traj[j])
                all_y_current.append(y_traj[j])
                all_x_delay.append(x_traj[delay_j])
                all_y_delay.append(y_traj[delay_j])
                all_x_next.append(x_traj[j + 1])
                all_y_next.append(y_traj[j + 1])

    return (
        np.array(all_alpha).reshape(-1, 1),
        np.array(all_x_current).reshape(-1, 1),
        np.array(all_y_current).reshape(-1, 1),
        np.array(all_x_delay).reshape(-1, 1),
        np.array(all_y_delay).reshape(-1, 1),
        np.array(all_x_next).reshape(-1, 1),
        np.array(all_y_next).reshape(-1, 1)
    )



class ModelBuilder:


    @staticmethod
    def define_multi_param_model(n_dimensions=2, n_layers=3, n_units=30, activation='tanh'):

        input_alpha = Input((1,), dtype=tf.float64, name='alpha_input')
        input_x_current = Input((1,), dtype=tf.float64, name='x_current_input')
        input_y_current = Input((1,), dtype=tf.float64, name='y_current_input')
        input_x_delay = Input((1,), dtype=tf.float64, name='x_delay_input')
        input_y_delay = Input((1,), dtype=tf.float64, name='y_delay_input')


        merged = Concatenate()([
            input_alpha, input_x_current, input_y_current, input_x_delay, input_y_delay
        ])


        x = merged
        for _ in range(n_layers):
            x = Dense(n_units, activation=activation, dtype=tf.float64)(x)


        drift_output = Dense(n_dimensions, dtype=tf.float64, name='drift_output')(x)
        diffusion_output = Dense(
            n_dimensions,
            activation=lambda x: tf.nn.softplus(x) + 1e-13,
            dtype=tf.float64,
            name='diffusion_output'
        )(x)

        model = Model(
            inputs=[input_alpha, input_x_current, input_y_current, input_x_delay, input_y_delay],
            outputs=[drift_output, diffusion_output],
            name='2d_multi_param_sdde_model'
        )
        return model


class SDDEMultiParamApproximator(tf.keras.Model):

    def __init__(self, sde_model, step_size, **kwargs):
        super().__init__(**kwargs)
        self.sde_model = sde_model
        self.step_size = tf.cast(step_size, dtype=tf.float64)
        self.nll_metric = tf.keras.metrics.Mean(name='nll')

    def call(self, inputs, training=None):

        return self.sde_model(inputs)

    def train_step(self, data):

        features, labels = data
        alpha, x_n, y_n, x_tau, y_tau = features
        x_np1, y_np1 = labels

        with tf.GradientTape() as tape:

            drift, diffusion = self.sde_model([alpha, x_n, y_n, x_tau, y_tau], training=True)


            current_state = tf.concat([x_n, y_n], axis=1)
            next_state = tf.concat([x_np1, y_np1], axis=1)
            mean = current_state + self.step_size * drift
            std = tf.math.sqrt(self.step_size) * diffusion

            dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)
            log_prob = dist.log_prob(next_state)
            loss = -tf.reduce_mean(log_prob)


        gradients = tape.gradient(loss, self.sde_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.sde_model.trainable_variables))


        self.nll_metric.update_state(loss)
        return {'nll': self.nll_metric.result()}

    def test_step(self, data):

        features, labels = data
        alpha, x_n, y_n, x_tau, y_tau = features
        x_np1, y_np1 = labels

        drift, diffusion = self.sde_model([alpha, x_n, y_n, x_tau, y_tau], training=False)
        current_state = tf.concat([x_n, y_n], axis=1)
        next_state = tf.concat([x_np1, y_np1], axis=1)
        mean = current_state + self.step_size * drift
        std = tf.math.sqrt(self.step_size) * diffusion

        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)
        log_prob = dist.log_prob(next_state)
        loss = -tf.reduce_mean(log_prob)

        self.nll_metric.update_state(loss)
        return {'nll': self.nll_metric.result()}

    def reset_metrics(self):
        self.nll_metric.reset_states()



def preprocess_data(alpha, x_curr, y_curr, x_del, y_del, x_next, y_next, test_size=0.2):

    indices = np.arange(len(alpha))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)

    def split(arr, idx):
        return arr[idx]


    train_features = (
        split(alpha, train_idx),
        split(x_curr, train_idx),
        split(y_curr, train_idx),
        split(x_del, train_idx),
        split(y_del, train_idx)
    )
    train_labels = (
        split(x_next, train_idx),
        split(y_next, train_idx)
    )


    test_features = (
        split(alpha, test_idx),
        split(x_curr, test_idx),
        split(y_curr, test_idx),
        split(x_del, test_idx),
        split(y_del, test_idx)
    )
    test_labels = (
        split(x_next, test_idx),
        split(y_next, test_idx)
    )

    return (train_features, train_labels), (test_features, test_labels)


def train_model(train_data, test_data, step_size, epochs=300, batch_size=256, save_dir='models'):

    train_features, train_labels = train_data
    test_features, test_labels = test_data


    base_model = ModelBuilder.define_multi_param_model()
    sdde_model = SDDEMultiParamApproximator(base_model, step_size=step_size)
    sdde_model.compile(optimizer=Adam(learning_rate=1e-3))


    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_model_path = os.path.join(save_dir, f'2d_multi_param_base_model_best_tau_{BEST_TAU:.4f}_{timestamp}.h5')


    class SaveBaseModelCallback(tf.keras.callbacks.Callback):
        def __init__(self, base_model, save_path):
            super().__init__()
            self.base_model = base_model
            self.save_path = save_path
            self.best_val_nll = float('inf')

        def on_epoch_end(self, epoch, logs=None):
            current_val_nll = logs.get('val_nll')
            if current_val_nll < self.best_val_nll:
                self.best_val_nll = current_val_nll
                self.base_model.save_weights(self.save_path, overwrite=True)


    save_callback = SaveBaseModelCallback(base_model, base_model_path)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_nll',
        patience=30,
        mode='min',
        restore_best_weights=False
    )



    history = sdde_model.fit(
        x=train_features,
        y=train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_features, test_labels),
        callbacks=[save_callback, early_stop],
        verbose=1
    )


    base_model.load_weights(base_model_path)


    return sdde_model, base_model, history, base_model_path

# --- 主程序入口 ---
if __name__ == "__main__":
    # -------------------------- 配置参数 --------------------------
    FULL_DATA_PATH = r"E:\2D-BO-MI\2d多参数\2d_multi_param_sdde_data.npz"
    MODEL_SAVE_DIR = r"E:\2D-BO-MI\2d多参数\models"
    EPOCHS = 300
    BATCH_SIZE = 256


    x_matrix, y_matrix, t_values, alpha_values, dt = load_full_trajectory_data(FULL_DATA_PATH)


    (alpha, x_curr, y_curr, x_del, y_del, x_next, y_next) = reconstruct_delay_matrix(
        x_matrix, y_matrix, t_values, alpha_values, BEST_TAU, dt
    )



    train_data, test_data = preprocess_data(alpha, x_curr, y_curr, x_del, y_del, x_next, y_next)
    train_features, _ = train_data
    test_features, _ = test_data



    sdde_model, base_model, history, model_path = train_model(
        train_data, test_data,
        step_size=dt,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir=MODEL_SAVE_DIR
    )




