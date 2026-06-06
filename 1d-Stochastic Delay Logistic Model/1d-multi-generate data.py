import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams['axes.unicode_minus'] = False


def true_sdde(t, x, x_delay, alpha=1.8):
    drift_term = alpha * x * (1 - x_delay)
    diffusion_term = 0.1 * x
    return drift_term, diffusion_term


def sample_data(sdde, tau, T, h, n_dimensions, n_traj, alphas, seed=None):
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
        np.array(all_alpha_values).reshape(-1, 1),
        x_matrices,
        t_values[start_index:]
    )


def save_data(filename, x_current, x_delay, x_next, alpha_values, x_matrices, t_values):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    x_matrices_dict = {f'x_matrix_alpha_{a}': mat for a, mat in x_matrices.items()}

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
    print(f"Multi-parameter data saved to: {filename}")


if __name__ == "__main__":
    OUTPUT_DIR = r"E:\1md"
    DATA_FILENAME = os.path.join(OUTPUT_DIR, 'multi_alpha_sdde_data.npz')

    alphas = [1.3, 1.5, 1.7, 1.8]
    true_tau = 1.0
    T = 10.0
    h = 0.01
    n_dimensions = 1
    n_traj_per_alpha = 200
    seed = 42

    
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

    save_data(DATA_FILENAME, x_current, x_delay, x_next, alpha_values, x_matrices, t_values)

