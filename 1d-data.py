import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
def true_sdde(t, x, x_delay):
    drift_term = 1.8 * x * (1 - x_delay)
    diffusion_term = 0.1 * x
    return drift_term, diffusion_term


def sample_data(sdde, tau, T, h, n_dimensions, n_traj, seed=None):

    rng = np.random.default_rng(seed)

    # 计算总步数和时间点
    num_steps = int((T + tau) / h)
    t_values = np.linspace(-tau, T, num_steps + 1)
    start_index = int(tau / h)  # 非零时间点的起始索引

    # 创建存储所有轨迹的矩阵
    x_matrix = np.zeros((n_traj, num_steps + 1 - start_index, n_dimensions))

    # SDDE函数适配器
    def sdde_adapter(t, x_list):
        x, x_tau = x_list
        return sdde(t, x, x_tau)

    # 生成每条轨迹
    for i in range(n_traj):
        x_values = np.zeros((num_steps + 1, n_dimensions))
        x0 = rng.uniform(low=0, high=2.5, size=(1, n_dimensions))
        x_values[0] = x0

        # 模拟轨迹
        for j in range(1, num_steps + 1):
            if t_values[j] <= 0:
                x_values[j] = x0
            else:
                delay_idx = j - int(tau / h)
                x_delay = x_values[delay_idx]

                # 获取漂移和扩散
                drift, diffusion = sdde_adapter(t_values[j], [x_values[j - 1], x_delay])

                # 生成布朗运动增量并更新状态
                noise = rng.normal(0, np.sqrt(h), size=(1, n_dimensions))
                x_values[j] = x_values[j - 1] + h * drift + diffusion * noise

        # 存储轨迹（只保存非负时间点）
        x_matrix[i] = x_values[start_index:]

    # 提取训练数据
    all_x_current = []
    all_x_delay = []
    all_x_next = []

    for i in range(n_traj):
        x_traj = x_matrix[i]
        traj_length = len(x_traj)

        for j in range(traj_length - 1):
            delay_idx = max(0, j - int(tau / h))
            all_x_current.append(x_traj[j])
            all_x_delay.append(x_traj[delay_idx])
            all_x_next.append(x_traj[j + 1])

    return (
        np.array(all_x_current),
        np.array(all_x_delay),
        np.array(all_x_next),
        x_matrix,
        t_values[start_index:]
    )
def save_data(filename, x_current, x_delay, x_next, x_matrix, t_values):
    """保存生成的数据到NPZ文件"""
    np.savez(filename,
             x_current=x_current,
             x_delay=x_delay,
             x_next=x_next,
             x_matrix=x_matrix,
             t_values=t_values)
    print(f"数据已成功保存到 {filename}")


def visualize_trajectories(x_matrix, t_values, n_samples=100, title="SDDE轨迹", save_path=None):
    """可视化随机延迟微分方程的轨迹"""
    plt.figure(figsize=(6, 5))

    # 随机选择n_samples条轨迹进行展示
    indices = np.random.choice(len(x_matrix), min(n_samples, len(x_matrix)), replace=False)

    for i in indices:
        plt.plot(t_values, x_matrix[i, :, 0], alpha=0.7)

    plt.xlabel('$t$', fontsize=12)
    plt.ylabel('$x(t)$', fontsize=12)

    plt.savefig(save_path, dpi=500, bbox_inches='tight')

    plt.show()
# ====================== 直接设置参数 ======================
tau = 1.0          # 延迟时间
T = 10.0           # 终止时间
h = 0.01           # 时间步长
n_dimensions = 1   # 状态维度
n_traj = 30       # 轨迹数量
seed = 42          # 随机种子
output_file = r"E:\sdde_data.npz"
plot_samples = 100  # 可视化轨迹数量
save_path = r"E:\trajectory_plot.png"

x_current, x_delay, x_next, x_matrix, t_values = sample_data(
    true_sdde, tau, T, h, n_dimensions, n_traj, seed
)

# 保存数据
save_data(output_file, x_current, x_delay, x_next, x_matrix, t_values)

# 可视化轨迹
title = f"SDDE(τ={tau}, T={T})"
visualize_trajectories(x_matrix, t_values, n_samples=plot_samples, title=title, save_path=save_path)



#多参数数据生成
def true_sdde(t, x, x_delay, alpha=1.8):
    drift_term = alpha * x * (1 - x_delay)
    diffusion_term = 0.1 * x
    return drift_term, diffusion_term


def sample_data(sdde, tau, T, h, n_dimensions, n_traj, alphas, seed=None):
    # 设置随机数种子
    rng = np.random.default_rng(seed)


    num_steps = int((T + tau) / h)
    t_values = np.linspace(-tau, T, num_steps + 1)
    start_index = int(tau / h)


    all_x_current = []
    all_x_delay = []
    all_x_next = []
    all_alpha_values = []
    x_matrices = {}


    for alpha in alphas:
        x_matrix = np.zeros((n_traj, num_steps + 1 - start_index, n_dimensions))


        def sdde_adapter(t, x_list):
            x, x_tau = x_list
            return sdde(t, x, x_tau, alpha)


        for i in range(n_traj):
            x_values = np.zeros((num_steps + 1, n_dimensions))

            x0 = rng.uniform(low=0, high=2.5, size=(1, n_dimensions))
            x_values[0] = x0


            for j in range(1, num_steps + 1):
                if t_values[j] <= 0:

                    x_values[j] = x0
                else:

                    delay_idx = j - int(tau / h)
                    x_delay = x_values[delay_idx]


                    drift, diffusion = sdde_adapter(t_values[j], [x_values[j - 1], x_delay])


                    noise = rng.normal(0, np.sqrt(h), size=(1, n_dimensions))
                    x_values[j] = x_values[j - 1] + h * drift + diffusion * noise


            x_matrix[i] = x_values[start_index:]


        x_matrices[alpha] = x_matrix


        for i in range(n_traj):
            x_traj = x_matrix[i]
            traj_length = len(x_traj)

            for j in range(traj_length - 1):

                delay_idx = max(0, j - int(tau / h))
                all_x_current.append(x_traj[j])
                all_x_delay.append(x_traj[delay_idx])
                all_x_next.append(x_traj[j + 1])
                all_alpha_values.append(alpha)


    return (
        np.array(all_x_current),
        np.array(all_x_delay),
        np.array(all_x_next),
        np.array(all_alpha_values).reshape(-1, 1),  # (n_samples, 1)
        x_matrices,
        t_values[start_index:]
    )
def save_data(filename, x_current, x_delay, x_next, alpha_values, x_matrices, t_values):

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    x_matrices_dict = {}
    for alpha in x_matrices:
        x_matrices_dict[f'x_matrix_alpha_{alpha}'] = x_matrices[alpha]

    np.savez(
        filename,
        x_current=x_current,
        x_delay=x_delay,
        x_next=x_next,
        alpha_values=alpha_values,
        t_values=t_values,
        **x_matrices_dict,
        true_tau=1.0
    )



if __name__ == "__main__":
    # -------------------------- 数据生成参数配置 --------------------------
    OUTPUT_DIR = r"E:\1D-BO-MI\多参数models"
    DATA_FILENAME = os.path.join(OUTPUT_DIR, 'multi_alpha_sdde_data.npz')
    alphas = [1.3, 1.5, 1.7, 1.8]
    true_tau = 1.0  # 真实延迟时间
    T = 10.0
    h = 0.01
    n_dimensions = 1
    n_traj_per_alpha = 200  # 每个alpha生成100条轨迹
    seed = 42  # 随机种子


    # 生成数据
    x_current, x_delay, x_next, alpha_values, x_matrices, t_values = sample_data(
        sdde=true_sdde,
        tau=true_tau,
        T=T,
        h=h,
        n_dimensions=n_dimensions,
        n_traj=n_traj_per_alpha,
        alphas=alphas,
        seed=seed
    )

    # 保存数据
    save_data(DATA_FILENAME, x_current, x_delay, x_next, alpha_values, x_matrices, t_values)

