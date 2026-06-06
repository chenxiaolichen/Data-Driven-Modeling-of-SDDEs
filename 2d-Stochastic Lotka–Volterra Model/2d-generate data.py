import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams['axes.unicode_minus'] = False

def custom_sdde(t, x, y, x_delay, y_delay, tau):
    drift_x = 1.6 * x * (1 - x_delay) + 0.1 * x * y_delay
    drift_y = 2 * y * (1 - y_delay) + 0.1 * y * x_delay
    diffusion_x = 0.1 * x
    diffusion_y = 0.1 * y
    return drift_x, diffusion_x, drift_y, diffusion_y

def sample_data(sdde, tau, T, h, n_dimensions, n_traj, seed=None):
    rng = np.random.default_rng(seed)
    num_steps = int((T + tau) / h)
    t_values = np.linspace(-tau, T, num_steps + 1)
    start_index = int(tau / h)

    x_matrix = np.zeros((n_traj, num_steps + 1 - start_index, n_dimensions))
    y_matrix = np.zeros((n_traj, num_steps + 1 - start_index, n_dimensions))

    def sdde_adapter(t, x_list, y_list, tau):
        x, x_tau = x_list
        y, y_tau = y_list
        return sdde(t, x, y, x_tau, y_tau, tau)

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
                    t_values[j], [x_values[j - 1], x_delay], [y_values[j - 1], y_delay], tau
                )

                noise_x = rng.normal(0, np.sqrt(h), size=(1, n_dimensions))
                noise_y = rng.normal(0, np.sqrt(h), size=(1, n_dimensions))
                x_values[j] = x_values[j - 1] + h * drift_x + diffusion_x * noise_x
                y_values[j] = y_values[j - 1] + h * drift_y + diffusion_y * noise_y

        x_matrix[i] = x_values[start_index:]
        y_matrix[i] = y_values[start_index:]

    all_x_current, all_y_current = [], []
    all_x_delay, all_y_delay = [], []
    all_x_next, all_y_next = [], []

    for i in range(n_traj):
        x_traj = x_matrix[i]
        y_traj = y_matrix[i]
        traj_length = len(x_traj)
        for j in range(traj_length - 1):
            delay_idx = max(0, j - int(tau / h))
            all_x_current.append(x_traj[j])
            all_y_current.append(y_traj[j])
            all_x_delay.append(x_traj[delay_idx])
            all_y_delay.append(y_traj[delay_idx])
            all_x_next.append(x_traj[j + 1])
            all_y_next.append(y_traj[j + 1])

    return (
        np.array(all_x_current), np.array(all_y_current),
        np.array(all_x_delay), np.array(all_y_delay),
        np.array(all_x_next), np.array(all_y_next),
        x_matrix, y_matrix, t_values[start_index:]
    )

def save_data(filename, x_current, y_current, x_delay, y_delay, x_next, y_next, x_matrix, y_matrix, t_values):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(filename,
             x_current=x_current, y_current=y_current,
             x_delay=x_delay, y_delay=y_delay,
             x_next=x_next, y_next=y_next,
             x_matrix=x_matrix, y_matrix=y_matrix,
             t_values=t_values)
    print(f"Data saved to {filename}")

def visualize_trajectories(x_matrix, y_matrix, t_values, n_samples=500, title="SDDE trajectories", save_path=None):
    plt.figure(figsize=(12, 6))
    indices = np.random.choice(len(x_matrix), min(n_samples, len(x_matrix)), replace=False)
    for i in indices:
        plt.plot(t_values, x_matrix[i, :, 0], alpha=0.7)
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('$x(t)$', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.figure(figsize=(12, 6))
    for i in indices:
        plt.plot(t_values, y_matrix[i, :, 0], alpha=0.7)
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('$y(t)$', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()

# ====================== Parameters ======================
tau = 1.0
T = 10.0
h = 0.01
n_dimensions = 1
n_traj = 400
seed = 42
output_file = r"E:\2d\2d_sdde_data.npz"
plot_samples = 100
plot_file = None

# ====================== Run ======================
print("Generating data...")
x_current, y_current, x_delay, y_delay, x_next, y_next, x_matrix, y_matrix, t_values = sample_data(
    custom_sdde, tau, T, h, n_dimensions, n_traj, seed
)

save_data(output_file, x_current, y_current, x_delay, y_delay, x_next, y_next, x_matrix, y_matrix, t_values)

title = f"SDDE trajectories (τ={tau}, T={T}, trajectories={n_traj})"
visualize_trajectories(x_matrix, y_matrix, t_values, n_samples=plot_samples, title=title, save_path=plot_file)