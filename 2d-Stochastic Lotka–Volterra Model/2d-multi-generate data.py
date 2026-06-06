import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams['axes.unicode_minus'] = False

def custom_sdde(t, x, y, x_delay, y_delay, tau, alpha):
    drift_x = alpha * x * (1 - x_delay) + 0.1 * x * y_delay
    drift_y = 2 * y * (1 - y_delay) + 0.1 * y * x_delay
    diffusion_x = 0.1 * x
    diffusion_y = 0.1 * y
    return drift_x, diffusion_x, drift_y, diffusion_y

def sample_data(sdde, tau, T, h, n_dimensions, n_traj, alpha_values, seed=None):
    rng = np.random.default_rng(seed)
    num_steps = int((T + tau) / h)
    t_values = np.linspace(-tau, T, num_steps + 1)
    start_index = int(tau / h)
    n_alpha = len(alpha_values)

    x_matrix = np.zeros((n_alpha, n_traj, num_steps + 1 - start_index, n_dimensions))
    y_matrix = np.zeros((n_alpha, n_traj, num_steps + 1 - start_index, n_dimensions))

    def sdde_adapter(t, x_list, y_list, tau, alpha):
        x, x_tau = x_list
        y, y_tau = y_list
        return sdde(t, x, y, x_tau, y_tau, tau, alpha)

    for a_idx, alpha in enumerate(alpha_values):
        print(f"Generating trajectories for alpha = {alpha} ...")
        for i in range(n_traj):
            x_values = np.zeros((num_steps + 1, n_dimensions))
            y_values = np.zeros((num_steps + 1, n_dimensions))
            x0 = rng.uniform(low=0, high=3, size=(1, n_dimensions))
            y0 = rng.uniform(low=0, high=2, size=(1, n_dimensions))
            x_values[0] = x0
            y_values[0] = y0

            for j in range(1, num_steps + 1):
                if t_values[j] <= 0:
                    x_values[j] = x0
                    y_values[j] = y0
                else:
                    delay_idx = j - int(tau / h)
                    x_delay = x_values[delay_idx]
                    y_delay = y_values[delay_idx]

                    drift_x, diffusion_x, drift_y, diffusion_y = sdde_adapter(
                        t_values[j], [x_values[j-1], x_delay], [y_values[j-1], y_delay], tau, alpha
                    )

                    noise_x = rng.normal(0, np.sqrt(h), size=(1, n_dimensions))
                    noise_y = rng.normal(0, np.sqrt(h), size=(1, n_dimensions))
                    x_values[j] = x_values[j-1] + h * drift_x + diffusion_x * noise_x
                    y_values[j] = y_values[j-1] + h * drift_y + diffusion_y * noise_y

            x_matrix[a_idx, i] = x_values[start_index:]
            y_matrix[a_idx, i] = y_values[start_index:]

    all_alpha = []
    all_x_current = []
    all_y_current = []
    all_x_delay = []
    all_y_delay = []
    all_x_next = []
    all_y_next = []

    for a_idx, alpha in enumerate(alpha_values):
        for i in range(n_traj):
            x_traj = x_matrix[a_idx, i]
            y_traj = y_matrix[a_idx, i]
            traj_length = len(x_traj)

            for j in range(traj_length - 1):
                delay_idx = max(0, j - int(tau / h))
                all_alpha.append(alpha)
                all_x_current.append(x_traj[j])
                all_y_current.append(y_traj[j])
                all_x_delay.append(x_traj[delay_idx])
                all_y_delay.append(y_traj[delay_idx])
                all_x_next.append(x_traj[j+1])
                all_y_next.append(y_traj[j+1])

    return (
        np.array(all_alpha),
        np.array(all_x_current),
        np.array(all_y_current),
        np.array(all_x_delay),
        np.array(all_y_delay),
        np.array(all_x_next),
        np.array(all_y_next),
        x_matrix,
        y_matrix,
        t_values[start_index:],
        alpha_values
    )

def save_data(filename, alpha, x_current, y_current, x_delay, y_delay, x_next, y_next,
              x_matrix, y_matrix, t_values, alpha_values):
    np.savez(filename,
             alpha=alpha,
             alpha_values=alpha_values,
             x_current=x_current,
             y_current=y_current,
             x_delay=x_delay,
             y_delay=y_delay,
             x_next=x_next,
             y_next=y_next,
             x_matrix=x_matrix,
             y_matrix=y_matrix,
             t_values=t_values)
    print(f"Data saved to {filename}")

def visualize_trajectories(x_matrix, y_matrix, t_values, alpha_values, n_samples=50, save_path=None):
    n_alpha = len(alpha_values)
    fig, axes = plt.subplots(n_alpha, 2, figsize=(15, 5 * n_alpha))
    if n_alpha == 1:
        axes = [axes]

    for a_idx, alpha in enumerate(alpha_values):
        indices = np.random.choice(x_matrix.shape[1], min(n_samples, x_matrix.shape[1]), replace=False)

        ax1 = axes[a_idx][0]
        for i in indices:
            ax1.plot(t_values, x_matrix[a_idx, i, :, 0], alpha=0.7)
        ax1.set_xlabel('$t$', fontsize=14)
        ax1.set_ylabel('$x(t)$', fontsize=13)
        ax1.set_title(f'alpha = {alpha} - X trajectories', fontsize=15)
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2 = axes[a_idx][1]
        for i in indices:
            ax2.plot(t_values, y_matrix[a_idx, i, :, 0], alpha=0.7)
        ax2.set_xlabel('$t$', fontsize=14)
        ax2.set_ylabel('$y(t)$', fontsize=13)
        ax2.set_title(f'alpha = {alpha} - Y trajectories', fontsize=15)
        ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

# ====================== Parameters ======================
tau = 1.0
T = 10.0
h = 0.01
n_dimensions = 1
n_traj = 200
seed = 42
alpha_values = np.array([1.3, 1.5, 1.6, 1.8])
output_file = r"E:\2md\2d_multi_param_sdde_data.npz"
plot_samples = 50
plot_file = r"E:\2md\trajectory_plot.png"


(alpha, x_current, y_current, x_delay, y_delay, x_next, y_next,
 x_matrix, y_matrix, t_values, alpha_values) = sample_data(
    custom_sdde, tau, T, h, n_dimensions, n_traj, alpha_values, seed
)

save_data(output_file, alpha, x_current, y_current, x_delay, y_delay, x_next, y_next,
          x_matrix, y_matrix, t_values, alpha_values)

visualize_trajectories(x_matrix, y_matrix, t_values, alpha_values,
                       n_samples=plot_samples, save_path=plot_file)

