import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

def true_sdde(t, x, x_delay):
    """True SDDE model: drift = 1.8*x*(1-x_delay), diffusion = 0.1*x"""
    drift_term = 1.8 * x * (1 - x_delay)
    diffusion_term = 0.1 * x
    return drift_term, diffusion_term


def sample_data(sdde, tau, T, h, n_dimensions, n_traj, seed=None):
    """Generate multiple SDDE trajectory data

    Args:
        sdde: SDDE model function returning (drift, diffusion)
        tau: time delay
        T: final time
        h: time step
        n_dimensions: state dimension
        n_traj: number of trajectories
        seed: random seed for reproducibility

    Returns:
        tuple: (x_current, x_delay, x_next, x_matrix, t_values)
    """
    rng = np.random.default_rng(seed)

    num_steps = int((T + tau) / h)
    t_values = np.linspace(-tau, T, num_steps + 1)
    start_index = int(tau / h)

    x_matrix = np.zeros((n_traj, num_steps + 1 - start_index, n_dimensions))

    def sdde_adapter(t, x_list):
        x, x_tau = x_list
        return sdde(t, x, x_tau)

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
    """Save generated data to NPZ file"""
    np.savez(filename,
             x_current=x_current,
             x_delay=x_delay,
             x_next=x_next,
             x_matrix=x_matrix,
             t_values=t_values)
    print(f"Data saved to {filename}")


def visualize_trajectories(x_matrix, t_values, n_samples=100, title="SDDE Trajectories", save_path=None):
    """Visualize SDDE trajectories"""
    plt.figure(figsize=(6, 5))

    indices = np.random.choice(len(x_matrix), min(n_samples, len(x_matrix)), replace=False)

    for i in indices:
        plt.plot(t_values, x_matrix[i, :, 0], alpha=0.7)

    plt.xlabel('$t$', fontsize=12)
    plt.ylabel('$x(t)$', fontsize=12)

    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.show()


# ====================== Parameters ======================
tau = 1.0
T = 10.0
h = 0.01
n_dimensions = 1
n_traj = 100
seed = 42
output_file = r"E:\1d\sdde_data.npz"
plot_samples = 100
save_path = r"E:\1d\trajectory_plot.png"


# ====================== Run ======================
print("Starting data generation...")
x_current, x_delay, x_next, x_matrix, t_values = sample_data(
    true_sdde, tau, T, h, n_dimensions, n_traj, seed
)

save_data(output_file, x_current, x_delay, x_next, x_matrix, t_values)

title = f"SDDE Trajectories (τ={tau}, T={T}, trajectories={n_traj})"
visualize_trajectories(x_matrix, t_values, n_samples=plot_samples, title=title, save_path=save_path)