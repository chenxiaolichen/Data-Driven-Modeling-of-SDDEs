# 1. 从 .npz 文件加载随机时滞微分方程(SDDE)的轨迹数据。
# 2. 使用基于互信息(Mutual Information)的目标函数和贝叶斯优化来识别时滞参数 tau。
# 3. 将完整的优化结果保存为 .pkl 文件，将摘要保存为 .json 文件。
# ==============================================================================

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




def load_sdde_data_from_npz(npz_path):
    """从指定的 .npz 文件加载轨迹数据。"""
    try:
        data = np.load(npz_path)
        x_matrix, t_values = data["x_matrix"], data["t_values"]
        dt = t_values[1] - t_values[0]
        if x_matrix.ndim == 2:
            x_matrix = x_matrix[:, :, np.newaxis]
        return x_matrix, t_values, dt
    except Exception as e:
        return None, None, None


def objective_function_mi(tau):
    """核心目标函数：计算给定tau的负互信息值。"""
    tau = tau[0]
    global trajectories, t_values, dt, mi_neighbors
    if tau < dt or tau > t_values[-1] / 2: return 1e10
    augmented_states, increments = [], []
    dim = trajectories.shape[2]
    for traj in trajectories:
        interp_funcs = [interp1d(t_values, traj[:, d], kind='cubic', fill_value="extrapolate") for d in range(dim)]
        start_idx, end_idx = int(np.ceil(tau / dt)), len(t_values) - 2
        for i in range(start_idx, end_idx + 1):
            t_current, x_current = t_values[i], traj[i]
            x_delayed = np.array([f(t_current - tau) for f in interp_funcs])
            augmented_states.append(np.concatenate([x_current, x_delayed]))
            increments.append(traj[i + 1] - traj[i])
    Z, dX = np.array(augmented_states), np.array(increments)
    if len(Z) < 50: return 1e10
    total_mi = 0
    for d in range(dX.shape[1]):
        mi_score = mutual_info_regression(Z, dX[:, d], n_neighbors=mi_neighbors, random_state=42)
        total_mi += mi_score[0]
    objective_value = -total_mi
    return objective_value


# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 1. 参数和路径配置 ---
    MI_NEIGHBORS = 10
    N_CALLS = 30
    N_RANDOM_STARTS = 10
    SEARCH_SPACE = [(0.1, 2.5)]
    INPUT_DATA_PATH = r"E:\sdde_data.npz"
    OUTPUT_DIR = r"E:\1D-BO-MI\results"

    # --- 2. 执行优化 ---
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
            random_state=12,
            verbose=False,
            noise="gaussian"
        )

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        print(f"\n--- 优化结束 (耗时: {duration:.2f} 秒) ---")

        # --- 3. 结果汇总与保存 ---
        found_tau = result.x[0]
        max_mi = -result.fun

        print(f"\n[最终结果] 识别出的最优 Tau: {found_tau:.4f}")
        print(f"           对应的最大互信息值 (MI): {max_mi:.6f}")

        run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        pickle_path = os.path.join(OUTPUT_DIR, f'optimization_result_{run_timestamp}.pkl')
        json_path = os.path.join(OUTPUT_DIR, f'summary_result_{run_timestamp}.json')

        with open(pickle_path, 'wb') as f:
            pickle.dump(result, f)


        # 创建并保存一个可读的 JSON 摘要文件
        summary_data = {
            'best_tau': float(found_tau),
            'max_mutual_information': float(max_mi),
            'min_objective_value': float(result.fun),
            'hyperparameters': {
                'mi_neighbors': MI_NEIGHBORS,
                'n_calls': N_CALLS,
                'n_random_starts': N_RANDOM_STARTS,
                'search_space': SEARCH_SPACE,
                'input_data_path': INPUT_DATA_PATH
            },
            'run_details': {
                'timestamp': run_timestamp,
                'duration_seconds': duration
            },
            'evaluation_history': {
                'taus_evaluated': np.array(result.x_iters).ravel().tolist(),
                'objective_values': np.array(result.func_vals).tolist()
            }
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)


        # --- 4. 生成并保存本次运行的图表 ---

        plt.figure(figsize=(6, 5))
        plot_convergence(result)
        plt.gca().set_title('')
        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel("Objective Function Min Value", fontsize=15)
        plt.tick_params(labelsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(r"E:\1D-BO-MI\results\loss.png", dpi=300, bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(6, 5))
        evaluated_taus = np.array(result.x_iters).flatten()
        evaluated_errors = result.func_vals
        min_idx = np.argmin(evaluated_errors)
        found_tau = evaluated_taus[min_idx]


        plt.scatter(evaluated_taus, evaluated_errors, c='skyblue', s=80, alpha=0.8, edgecolors='navy',
                    label='BO Evaluation point')

        plt.scatter(evaluated_taus[min_idx], evaluated_errors[min_idx],
                    c='red', s=200, marker='*', edgecolors='darkred',
                    label=f'Identified τ = {found_tau:.4f}')

        plt.xlabel(' τ', fontsize=15)
        plt.ylabel('-J(τ)', fontsize=15)
        plt.tick_params(labelsize=14)
        plt.legend(frameon=True, fontsize=13.5)

        plt.savefig(r"E:\1D-BO-MI\results\tau_loss_scatter.png", dpi=300, bbox_inches="tight")
        plt.show()


# --- MLE ---
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os
# ====================== 1. 导入保存的数据 ======================
def load_sdde_data(filename):
    """加载预先生成的SDDE数据（完全匹配sample_data的输出格式）"""
    data = np.load(filename)
    x_matrix = data['x_matrix']  # 3维：(n_traj, traj_length, n_dimensions) → (100, 1000, 1)
    t_values = data['t_values']  # 1维：(traj_length,) → (1000,)
    return x_matrix, t_values


# ====================== 2. 延迟矩阵重构（完全对齐原始数据生成逻辑） ======================
def reconstruct_delay_matrix(x_matrix, tau_star, h):

    n_traj, traj_length, n_dimensions = x_matrix.shape  # 解析3维维度
    k = int(tau_star / h)  # 延迟对应的时间步数

    # 存储所有轨迹的(current, delay, next)数据对
    all_x_current = []
    all_x_delay = []
    all_x_next = []

    for i in range(n_traj):  # 遍历每条轨迹
        x_traj = x_matrix[i]  # 单条轨迹：(traj_length, n_dimensions) → (1000, 1)

        for j in range(traj_length - 1):  # 遍历轨迹的每个时间步（除最后一个，避免x_next越界）
            # 计算延迟索引：j是当前时间步，delay_idx = j - k
            delay_idx = j - k
            # 若delay_idx < 0，取0（用初始状态填充）
            delay_idx = max(0, delay_idx)

            # 提取当前状态、延迟状态、下一个状态
            x_current_j = x_traj[j]  # 当前状态：t时刻
            x_delay_j = x_traj[delay_idx]  # 延迟状态：t - tau*时刻（或x(0)）
            x_next_j = x_traj[j + 1]  # 下一个状态：t + h时刻

            # 添加到列表
            all_x_current.append(x_current_j)
            all_x_delay.append(x_delay_j)
            all_x_next.append(x_next_j)

    # 转换为numpy数组
    x_current = np.array(all_x_current)
    x_delay = np.array(all_x_delay)
    x_next = np.array(all_x_next)

    return x_current, x_delay, x_next

# ====================== 3. 模型定义======================
class SDDEIntegrators:
    """实现延迟随机微分方程(SDDE)积分方法"""
    @staticmethod
    def euler_maruyama(xn, xn_tau, h, f_sigma, rng):
        dW = rng.normal(loc=0, scale=np.sqrt(h), size=xn.shape)
        xk = xn.reshape(1, -1)
        xk_tau = xn_tau.reshape(1, -1)
        fk, sk = f_sigma([xk, xk_tau])
        return xk + h * fk + sk * dW


class ModelBuilder:
    """构建神经网络模型"""

    @staticmethod
    def define_gaussian_process(n_input_dimensions, n_output_dimensions, n_layers, n_dim_per_layer, name,
                                activation="tanh", diffusivity_type="diagonal"):
        input_current = Input((n_input_dimensions,), dtype=tf.float64, name=name + '_current')
        input_delay = Input((n_input_dimensions,), dtype=tf.float64, name=name + '_delay')
        merged = Concatenate()([input_current, input_delay])

        x = merged
        for i in range(n_layers):
            x = Dense(n_dim_per_layer, activation=activation, dtype=tf.float64)(x)

        drift_output = Dense(n_output_dimensions, dtype=tf.float64)(x)
        diffusion_output = Dense(n_output_dimensions,
                                 activation=lambda x: tf.nn.softplus(x) + 1e-13,
                                 dtype=tf.float64)(x)
        return Model(inputs=[input_current, input_delay], outputs=[drift_output, diffusion_output])


class SDDEApproximationNetwork(tf.keras.Model):
    """SDDE参数识别网络"""

    def __init__(self, sde_model, step_size, method="euler", **kwargs):
        super().__init__(**kwargs)
        self.sde_model = sde_model
        self.step_size = step_size
        self.method = method

    def call(self, inputs):
        x_n, x_n_tau, x_np1 = inputs
        if self.method == "euler":
            drift, diffusion = self.sde_model([x_n, x_n_tau])
            step_size_tensor = tf.cast(self.step_size, dtype=diffusion.dtype)
            mean = x_n + step_size_tensor * drift
            std = tf.math.sqrt(step_size_tensor) * diffusion
            dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
            log_prob = dist.log_prob(x_np1)
            loss = -tf.reduce_mean(log_prob)
            self.add_loss(loss)
            self.add_metric(loss, name="nll", aggregation="mean")
        return self.sde_model([x_n, x_n_tau])


# ====================== 4. 数据预处理 ======================
def preprocess_data(x_current, x_delay, x_next, test_size=0.2, random_state=42):
    """数据划分与预处理"""
    # 校验输入维度一致性
    assert x_current.shape == x_delay.shape == x_next.shape, \
        f"x_current、x_delay、x_next维度不一致！分别为{x_current.shape}、{x_delay.shape}、{x_next.shape}"

    N_samples, n_dimensions = x_current.shape
    if N_samples < 10:
        raise ValueError(f"有效数据量过少（仅{N_samples}个样本），无法划分训练集/测试集！")

    # 拼接当前状态和延迟状态（模型输入：current + delay）
    X = np.hstack([x_current, x_delay])  # (N_samples, 2*n_dim)
    y = x_next  # (N_samples, n_dim)

    # 动态调整测试集占比，避免训练集为空
    test_size = min(test_size, (N_samples - 1) / N_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # 拆分回当前状态和延迟状态
    x_current_train = X_train[:, :n_dimensions]
    x_delay_train = X_train[:, n_dimensions:]
    x_current_test = X_test[:, :n_dimensions]
    x_delay_test = X_test[:, n_dimensions:]


    return (x_current_train, x_delay_train, y_train,
            x_current_test, x_delay_test, y_test, n_dimensions)


# ====================== 5. 模型训练与评估 ======================
def train_and_evaluate_model(x_current_train, x_delay_train, y_train,
                             x_current_test, x_delay_test, y_test,
                             n_dimensions, step_size, epochs=300, batch_size=128,
                             model_save_path=None):
    """训练并评估SDDE模型，支持保存模型"""
    # 构建模型（输入/输出维度=状态维度n_dimensions）
    model = ModelBuilder.define_gaussian_process(
        n_input_dimensions=n_dimensions,
        n_output_dimensions=n_dimensions,
        n_layers=2,
        n_dim_per_layer=32,
        name="sdde_model",
        activation="tanh"
    )

    sdde_approximator = SDDEApproximationNetwork(
        sde_model=model,
        step_size=step_size,
        method="euler"
    )

    # 编译模型
    sdde_approximator.compile(optimizer=Adam(learning_rate=0.001))

    # 创建回调函数以保存最佳模型
    callbacks = []
    if model_save_path:
        model_dir = os.path.dirname(model_save_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_nll',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='min'
        )
        callbacks.append(model_checkpoint)

    # 动态调整batch_size
    batch_size = min(batch_size, len(x_current_train))


    # 训练模型
    history = sdde_approximator.fit(
        [x_current_train.astype(np.float64), x_delay_train.astype(np.float64), y_train.astype(np.float64)],
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.1 if len(x_current_train) >= 100 else 0,
        callbacks=callbacks
    )

    # 加载最佳模型权重
    if model_save_path and os.path.exists(model_save_path):
        sdde_approximator.load_weights(model_save_path)


    # 评估测试集
    test_loss = sdde_approximator.evaluate(
        [x_current_test.astype(np.float64), x_delay_test.astype(np.float64), y_test.astype(np.float64)],
        verbose=0
    )

    return sdde_approximator, history


# ====================== 6. 模型加载函数 ======================
def load_sdde_model(n_dimensions, step_size, model_weights_path):
    # 构建模型结构
    model = ModelBuilder.define_gaussian_process(
        n_input_dimensions=n_dimensions,
        n_output_dimensions=n_dimensions,
        n_layers=2,
        n_dim_per_layer=32,
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
        dummy_input = [
            tf.random.uniform(shape=(1, n_dimensions), dtype=tf.float64),
            tf.random.uniform(shape=(1, n_dimensions), dtype=tf.float64),
            tf.random.uniform(shape=(1, n_dimensions), dtype=tf.float64)
        ]
        _ = sdde_approximator(dummy_input)

    # 加载权重
    sdde_approximator.load_weights(model_weights_path)


    return sdde_approximator


# ====================== 主程序：执行训练流程 ======================
if __name__ == "__main__":
    # ====================== 参数配置 ======================
    data_filename = r"E:\1D-BO-MI\sdde_data.npz"
    model_save_path = r"E:\1D-BO-MI\单参数models/sdde_model_best.h5"
    tau_star = 0.9948  #BO识别结果
    h = 0.01  # 时间步长

    try:
        # ====================== 加载数据 ======================
        x_matrix, t_values = load_sdde_data(data_filename)


        # ====================== 基于tau*重新构造延迟矩阵X_tau* ======================
        x_current, x_delay, x_next = reconstruct_delay_matrix(x_matrix, tau_star, h)

        # ====================== 数据预处理 ======================
        (x_current_train, x_delay_train, y_train,
         x_current_test, x_delay_test, y_test,
         n_dimensions) = preprocess_data(x_current, x_delay, x_next)

        # ====================== 训练与评估模型 ======================
        print(f"开始训练模型，将保存至: {model_save_path}")
        sdde_approximator, history = train_and_evaluate_model(
            x_current_train, x_delay_train, y_train,
            x_current_test, x_delay_test, y_test,
            n_dimensions=n_dimensions, step_size=h, epochs=300, batch_size=128,
            model_save_path=model_save_path
        )

        # 绘制训练损失曲线
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('NLL Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['nll'], label='训练NLL')
        if 'val_nll' in history.history:
            plt.plot(history.history['val_nll'], label='验证NLL')
        plt.xlabel('Epoch')
        plt.ylabel('NLL')
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"程序执行出错：{str(e)}")
        # 输出关键信息用于调试
        x_matrix, _ = load_sdde_data(data_filename)
        print(f"当前x_matrix的维度：{x_matrix.shape}")
        print(f"tau_star={tau_star}，h={h}，延迟步数k={int(tau_star / h)}")
def predict_next_state(sde_model, x_current, x_delay, step_size):
    """使用底层sde_model预测下一个状态（仅需两个输入）"""
    x_tensor = [
        tf.convert_to_tensor(x_current, dtype=tf.float64),
        tf.convert_to_tensor(x_delay, dtype=tf.float64)
    ]
    drift, diffusion = sde_model(x_tensor)  # 直接调用底层模型
    mean_pred = x_current + step_size * drift.numpy()
    return mean_pred

# 加载模型前确保模型已创建变量
model_weights_path = r"E:\1D-BO-MI\单参数models/sdde_model_best.h5"
loaded_model = load_sdde_model(n_dimensions=1, step_size=0.01, model_weights_path=model_weights_path)
data_filename = r"E:\1D-BO-MI\sdde_data.npz"

def true_sdde(t, x, x_delay):
    """真实SDDE模型：drift = 1.8*x*(1 - x_delay), diffusion = 0.1*x"""
    drift_term = 1.8 * x * (1 - x_delay)
    diffusion_term = 0.1 * x
    return drift_term, diffusion_term

#可视化
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.linalg import norm


def compare_drift_diffusion(apx_model, true_sdde, x_range=None, x_tau_range=None, n_points=50):
    """比较估计的和真实的漂移与扩散函数，并计算L₂误差"""

    if x_range is None:
        x_range = np.linspace(0, 1, n_points)
    if x_tau_range is None:
        x_tau_range = np.linspace(0, 1, n_points)

    sde_model = apx_model.sde_model

    # 计算真实的漂移和扩散
    true_drift = np.zeros((n_points, n_points))
    true_diff = np.zeros((n_points, n_points))

    for i, x in enumerate(x_range):
        for j, x_tau in enumerate(x_tau_range):
            drift, diff = true_sdde(0, np.array([[x]]), np.array([[x_tau]]))
            true_drift[i, j] = drift[0, 0]
            true_diff[i, j] = diff[0, 0]

    # 计算估计的漂移和扩散
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

    # -------------------------- 计算L₂误差 --------------------------
    # 漂移函数的L₂误差（全局）
    drift_l2_error = norm(true_drift - apx_drift, ord=2)
    drift_l2_error_normalized = drift_l2_error / norm(true_drift, ord=2)

    # 扩散函数的L₂误差（全局）
    diff_l2_error = norm(true_diff - apx_diff, ord=2)
    diff_l2_error_normalized = diff_l2_error / norm(true_diff, ord=2)

    # 扩散函数均值的L₂误差（对应绘图中的均值曲线）
    true_diff_mean = np.mean(true_diff, axis=1)
    apx_diff_mean = np.mean(apx_diff, axis=1)
    diff_mean_l2_error = norm(true_diff_mean - apx_diff_mean, ord=2)
    diff_mean_l2_error_normalized = diff_mean_l2_error / norm(true_diff_mean, ord=2)

    # 打印L₂误差结果
    print("=" * 50)
    print("漂移函数L₂误差（全局）：{:.6f}".format(drift_l2_error))
    print("漂移函数归一化L₂误差：{:.6f}".format(drift_l2_error_normalized))
    print("-" * 30)
    print("扩散函数L₂误差（全局）：{:.6f}".format(diff_l2_error))
    print("扩散函数归一化L₂误差：{:.6f}".format(diff_l2_error_normalized))
    print("-" * 30)
    print("扩散函数均值L₂误差：{:.6f}".format(diff_mean_l2_error))
    print("扩散函数均值归一化L₂误差：{:.6f}".format(diff_mean_l2_error_normalized))
    print("=" * 50)



    drift_min = min(np.min(true_drift), np.min(apx_drift))
    drift_max = max(np.max(true_drift), np.max(apx_drift))

    # 真实漂移函数
    plt.figure(figsize=(6, 5))
    im1 = plt.imshow(true_drift, cmap='viridis', extent=[x_tau_range[0], x_tau_range[-1], x_range[0], x_range[-1]],
                     origin='lower', aspect='auto', vmin=drift_min, vmax=drift_max)
    plt.xlabel('x(t-τ)', fontsize=15)
    plt.ylabel('x(t)', fontsize=15)
    plt.title('true f', fontsize=15)
    cbar1 = plt.colorbar(im1)

    cbar1.set_ticks(np.linspace(drift_min, drift_max, 5))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(r'E:\1D-BO-MI\单参数models\true_drift1.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 预测漂移函数
    plt.figure(figsize=(6, 5))
    im2 = plt.imshow(apx_drift, cmap='viridis', extent=[x_tau_range[0], x_tau_range[-1], x_range[0], x_range[-1]],
                     origin='lower', aspect='auto', vmin=drift_min, vmax=drift_max)
    plt.xlabel('x(t-τ)', fontsize=15)
    plt.ylabel('x(t)', fontsize=15)
    plt.title('predicted f', fontsize=15)
    cbar2 = plt.colorbar(im2)

    cbar2.set_ticks(np.linspace(drift_min, drift_max, 5))
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(r'E:\1D-BO-MI\单参数models\predicted_drift1.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制扩散函数对比
    plt.figure(figsize=(6, 5))

    # 计算真实扩散函数
    true_diff_mean = np.mean(true_diff, axis=1)
    # 计算预测扩散函数在不同x_tau下的平均值
    apx_diff_mean = np.mean(apx_diff, axis=1)

    # 绘制真实扩散函数曲线
    plt.plot(x_range, true_diff_mean, 'b-', linewidth=2, label='true')
    # 绘制预测扩散函数曲线
    plt.plot(x_range, apx_diff_mean, 'r--', linewidth=2, label='predict')

    plt.xlabel('x(t)', fontsize=15)
    plt.ylabel('σ', fontsize=15)
    plt.legend(fontsize=14, frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(r'E:\1D-BO-MI\单参数models\diffusion_comparison1.png', dpi=300, bbox_inches='tight')
    plt.show()



# 调用函数
error_results = compare_drift_diffusion(loaded_model, true_sdde)


# -------------------  模型加载函数-------------------
def load_sdde_model(n_dimensions, step_size, model_weights_path):
    # 构建底层FNN模型
    sde_model = define_gaussian_process(
        n_input_dimensions=n_dimensions,
        n_output_dimensions=n_dimensions,
        n_layers=2,  # 与训练时一致
        n_dim_per_layer=32,  # 与训练时一致
        name="sdde_model"
    )
    # 构建包装器模型
    apx_model = SDDEApproximationNetwork(sde_model, step_size)
    # 虚拟调用创建变量（触发模型权重初始化）
    dummy_input = [
        tf.random.uniform(shape=(1, n_dimensions), dtype=tf.float64),
        tf.random.uniform(shape=(1, n_dimensions), dtype=tf.float64)
    ]
    _ = apx_model(dummy_input)  # 现在call()方法已实现，不会报错
    # 加载训练好的权重
    apx_model.load_weights(model_weights_path)
    print(f"成功加载模型权重: {model_weights_path}")
    return apx_model


# ------------------- 真实SDDE模型 -------------------
def true_sdde(t, x, x_delay):
    drift_term = 1.8 * x * (1 - x_delay)
    diffusion_term = 0.1 * x
    return drift_term, diffusion_term


# ------------------- 多初始值轨迹模拟函数 -------------------
def simulate_trajectories(sde_model, true_sdde, initial_conditions, tau_true, tau_pred, T, h, rng, save_path=None):

    max_tau = max(tau_true, tau_pred)
    num_steps = int((T + max_tau) / h)
    t_values = np.linspace(-max_tau, T, num_steps + 1)
    start_index = int(max_tau / h)  # 只绘制t>=0的部分
    n_dimensions = initial_conditions[0].shape[-1]

    # 共享噪声
    shared_noise = rng.normal(loc=0, scale=np.sqrt(h), size=(num_steps, n_dimensions))

    # 绘图设置
    plt.figure(figsize=(6, 5))
    plt.rcParams['font.family'] = 'Times New Roman'
    true_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    pred_colors = ['#aec7e8', '#98df8a', '#ffbb78', '#c5b0d5']

    all_true_traj = []
    all_pred_traj = []

    for idx, initial_condition in enumerate(initial_conditions):

        true_x = np.zeros((num_steps + 1, n_dimensions))
        pred_x = np.zeros((num_steps + 1, n_dimensions))
        true_x[0] = initial_condition
        pred_x[0] = initial_condition


        for i in range(1, num_steps + 1):
            if t_values[i] <= 0:
                true_x[i] = initial_condition
                pred_x[i] = initial_condition
            else:

                delay_idx_true = max(0, i - int(tau_true / h))
                x_current_true = true_x[i - 1]
                x_delay_true = true_x[delay_idx_true]
                drift_true, diff_true = true_sdde(t_values[i], x_current_true, x_delay_true)
                true_x[i] = x_current_true + h * drift_true + diff_true * shared_noise[i - 1]

                delay_idx_pred = max(0, i - int(tau_pred / h))
                x_current_pred = pred_x[i - 1].reshape(1, -1)
                x_delay_pred = pred_x[delay_idx_pred].reshape(1, -1)
                x_tensor = [
                    tf.convert_to_tensor(x_current_pred, dtype=tf.float64),
                    tf.convert_to_tensor(x_delay_pred, dtype=tf.float64)
                ]
                drift_pred, diff_pred = sde_model(x_tensor)
                pred_x[i] = x_current_pred.flatten() + h * drift_pred.numpy().flatten() + diff_pred.numpy().flatten() * \
                            shared_noise[i - 1]

        all_true_traj.append(true_x)
        all_pred_traj.append(pred_x)


        x0_value = initial_condition[0][0]

        plt.plot(t_values[start_index:], true_x[start_index:, 0],
                 color=true_colors[idx % len(true_colors)], linestyle='-',
                 linewidth=2.5, label=f'$x_0={x0_value:.1f}$')

        plt.plot(t_values[start_index:], pred_x[start_index:, 0],
                 color=pred_colors[idx % len(pred_colors)], linestyle='--',
                 linewidth=2.5)


    plt.xlabel('t', fontsize=15)
    plt.ylabel('x(t)', fontsize=15)
    plt.legend(fontsize=14, frameon=False, loc='upper right', bbox_to_anchor=(0.95, 1))
    plt.tick_params(labelsize=14)

    plt.tight_layout()


    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    return all_true_traj, all_pred_traj


# ------------------- 执行参数设置与运行 -------------------

h = 0.01
T = 10.0
tau_true = 1.0  # 真实轨迹的延迟时间
tau_pred = 0.9948  # 预测轨迹的延迟时间
model_weights_path = r"E:1D-BO-MI\单参数models\sdde_model_best.h5"
n_dimensions = 1


loaded_model = load_sdde_model(n_dimensions=n_dimensions, step_size=h, model_weights_path=model_weights_path)


sde_model = loaded_model.sde_model


rng = np.random.default_rng(seed=45)

# 多个初始条件
initial_conditions = [
    np.array([[0.1]]),
    np.array([[0.2]]),
    np.array([[0.6]]),
    np.array([[0.8]])

]


true_trajectories, pred_trajectories = simulate_trajectories(
    sde_model=sde_model,
    true_sdde=true_sdde,
    initial_conditions=initial_conditions,
    tau_true=tau_true,
    tau_pred=tau_pred,
    T=T,
    h=h,
    rng=rng,
    save_path=r'E:\1D-BO-MI\单参数models\multiple_trajectories_small.png'
)

#多参数
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


trajectories = None  # 所有alpha的轨迹集合（n_total_traj, n_time, n_dim）
t_values = None  # 时间点
dt = None  # 时间步长
mi_neighbors = None  # 互信息计算的近邻数


def load_multi_alpha_data_from_npz(npz_path):

    try:

        data = np.load(npz_path)
        t_values = data["t_values"]
        dt = t_values[1] - t_values[0]


        all_trajectories = []
        for key in data.files:
            if key.startswith('x_matrix_alpha_'):
                alpha_traj = data[key]  # (n_traj, n_time, n_dim)
                all_trajectories.append(alpha_traj)

        # 合并所有轨迹（n_total_traj, n_time, n_dim）
        trajectories = np.concatenate(all_trajectories, axis=0)


        if trajectories.ndim == 2:
            trajectories = trajectories[:, :, np.newaxis]
        return trajectories, t_values, dt
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
        # 三次样条插值
        interp_funcs = [interp1d(t_values, traj[:, d], kind='cubic', fill_value="extrapolate")
                        for d in range(dim)]

        # 有效时间索引（确保延迟状态不越界）
        start_idx = int(np.ceil(tau / dt))
        end_idx = len(t_values) - 2  # 避免x_next越界

        for i in range(start_idx, end_idx + 1):
            t_current = t_values[i]
            x_current = traj[i]
            # 计算延迟状态x(t - tau)
            x_delayed = np.array([f(t_current - tau) for f in interp_funcs])
            # 构建增强状态（当前状态 + 延迟状态）
            augmented_states.append(np.concatenate([x_current, x_delayed]))
            # 构建状态增量（dX = X(t+1) - X(t)）
            increments.append(traj[i + 1] - traj[i])

    # 转换为数组
    Z = np.array(augmented_states)  # (n_samples, 2*dim)
    dX = np.array(increments)  # (n_samples, dim)

    # 样本数过少时返回极大值
    if len(Z) < 50:
        return 1e10

    # 计算总互信息（所有维度求和）
    total_mi = 0
    for d in range(dX.shape[1]):
        mi_score = mutual_info_regression(Z, dX[:, d], n_neighbors=mi_neighbors, random_state=42)
        total_mi += mi_score[0]

    # 贝叶斯优化是最小化，因此返回负互信息
    objective_value = -total_mi
    return objective_value


def save_bo_results(result, found_tau, max_mi, duration, hyperparams, input_path, output_dir):
    """保存贝叶斯优化结果（完整结果+摘要）"""
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(output_dir, exist_ok=True)


    pickle_path = os.path.join(output_dir, f'bo_result_multi_alpha_{run_timestamp}.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(result, f)

    # 保存可读摘要（json）
    summary_data = {
        'best_tau': float(found_tau),
        'max_mutual_information': float(max_mi),
        'min_objective_value': float(result.fun),
        'hyperparameters': {
            'mi_neighbors': hyperparams['mi_neighbors'],
            'n_calls': hyperparams['n_calls'],
            'n_random_starts': hyperparams['n_random_starts'],
            'search_space': hyperparams['search_space'],
            'input_data_path': input_path
        },
        'run_details': {
            'timestamp': run_timestamp,
            'duration_seconds': duration
        },
        'evaluation_history': {
            'taus_evaluated': np.array(result.x_iters).ravel().tolist(),
            'objective_values': np.array(result.func_vals).tolist()
        }
    }
    json_path = os.path.join(output_dir, f'bo_summary_multi_alpha_{run_timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    print(f"[保存成功] 结果摘要：{json_path}")

    return pickle_path, json_path


def plot_bo_results(result, found_tau, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # --- 绘图1：收敛曲线 ---
    plt.figure(figsize=(6, 5))
    plt.rcParams['font.family'] = 'Times New Roman'

    plot_convergence(result)
    plt.gca().set_title('')
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Objective Function Min Value", fontsize=15)  # 更新Y轴标签
    plt.tick_params(labelsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(r"E:\1D-BO-MI\多参数models\bo_results\loss.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --- 绘图2：tau-目标函数散点图 ---
    plt.figure(figsize=(6, 5))
    plt.rcParams['font.family'] = 'Times New Roman'

    evaluated_taus = np.array(result.x_iters).flatten()
    evaluated_errors = result.func_vals
    min_idx = np.argmin(evaluated_errors)
    found_tau = evaluated_taus[min_idx]


    plt.scatter(evaluated_taus, evaluated_errors, c='skyblue', s=80, alpha=0.8, edgecolors='navy',
                label='BO Evaluation point')

    plt.scatter(evaluated_taus[min_idx], evaluated_errors[min_idx],
                c='red', s=200, marker='*', edgecolors='darkred',
                label=f'Identified τ = {found_tau:.4f}')

    plt.xlabel('τ', fontsize=15)
    plt.ylabel('-J(τ)', fontsize=15)  # 更新Y轴标签
    plt.tick_params(labelsize=12)
    plt.legend(frameon=True, fontsize=13.5)

    plt.savefig(r"E:\1D-BO-MI\多参数models\bo_results\tau_loss_scatter.png", dpi=300, bbox_inches="tight")
    plt.show()





if __name__ == "__main__":
    # -------------------------- 贝叶斯优化参数配置 --------------------------
    INPUT_DATA_PATH = r"E:\1D-BO-MI\多参数models\multi_alpha_sdde_data.npz"
    OUTPUT_DIR = r"E:\1D-BO-MI\多参数models\bo_results"

    # 贝叶斯优化超参数
    HYPERPARAMS = {
        'mi_neighbors': 10,  # 互信息计算的近邻数
        'n_calls': 30,  # 总评估次数
        'n_random_starts': 10,  # 随机初始点数量
        'search_space': [(0.5, 2.5)]  # τ搜索空间
    }


    # 1. 加载多参数数据
    trajectories, t_values, dt = load_multi_alpha_data_from_npz(INPUT_DATA_PATH)
    mi_neighbors = HYPERPARAMS['mi_neighbors']

    if trajectories is None:

        exit(1)

    # 2. 执行贝叶斯优化
    print("\n开始优化...")
    start_time = time.time()
    result = gp_minimize(
        func=objective_function_mi,
        dimensions=HYPERPARAMS['search_space'],
        acq_func="EI",  # 期望改进
        n_calls=HYPERPARAMS['n_calls'],
        n_random_starts=HYPERPARAMS['n_random_starts'],
        random_state=88, #  种子数
        verbose=False,
        noise="gaussian"
    )
    end_time = time.time()
    duration = round(end_time - start_time, 2)

    # 3. 解析结果
    found_tau = result.x[0]
    max_mi = -result.fun

    print(f"\n" + "=" * 50)
    print("贝叶斯优化完成！")
    print(f"耗时：{duration:.2f} 秒")
    print(f"最优τ：{found_tau:.4f}")
    print(f"最大互信息（MI）：{max_mi:.6f}")
    print("=" * 50)

    # 4. 保存结果
    save_bo_results(result, found_tau, max_mi, duration, HYPERPARAMS, INPUT_DATA_PATH, OUTPUT_DIR)

    # 5. 绘制结果图
    plot_bo_results(result, found_tau, OUTPUT_DIR)
