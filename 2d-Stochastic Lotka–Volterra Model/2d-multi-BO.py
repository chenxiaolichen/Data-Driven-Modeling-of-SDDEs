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

plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=FutureWarning)

trajectories = None
t_values = None
dt = None
mi_neighbors = None

def load_multi_param_sdde_data(npz_path):
    try:
        print(f"Loading multi-parameter data from '{npz_path}'...")
        data = np.load(npz_path)
        x_matrix = data["x_matrix"]
        y_matrix = data["y_matrix"]
        t_values = data["t_values"]
        alpha_values = data["alpha_values"]

        dt = t_values[1] - t_values[0]
        n_alpha, n_traj, n_time, _ = x_matrix.shape

        global trajectories
        x_flat = x_matrix.reshape(-1, n_time, 1)
        y_flat = y_matrix.reshape(-1, n_time, 1)
        trajectories = np.concatenate([x_flat, y_flat], axis=2)

        print("Data loaded successfully!")
        print(f"  - Number of alpha values: {n_alpha}")
        print(f"  - Trajectories per alpha: {n_traj}")
        print(f"  - Total trajectories: {trajectories.shape[0]}")
        print(f"  - Shape: {trajectories.shape}")
        print(f"  - Time step dt: {dt:.4f}")

        return t_values, dt, alpha_values
    except FileNotFoundError:
        print(f"Error: File not found at '{npz_path}'")
        return None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
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
            x_current = traj[i]
            x_delayed = np.array([f(t_current - tau) for f in interp_funcs])
            z = np.concatenate([x_current, x_delayed])
            augmented_states.append(z)
            dx = traj[i+1] - traj[i]
            increments.append(dx)

    Z, dX = np.array(augmented_states), np.array(increments)
    if len(Z) < 50:
        return 1e10

    total_mi = 0
    for d in range(dim):
        mi_score = mutual_info_regression(Z, dX[:, d], n_neighbors=mi_neighbors, random_state=42)
        total_mi += mi_score[0]

    objective_value = -total_mi
    print(f"Test tau = {tau:.4f}, Objective (-MI) = {objective_value:.6f}")
    return objective_value

if __name__ == "__main__":
    INPUT_DATA_PATH = r"E:\2md\2d_multi_param_sdde_data.npz"
    OUTPUT_DIR = r"E:\2md"

    MI_NEIGHBORS = 10
    N_CALLS = 30
    N_RANDOM_STARTS = 15
    SEARCH_SPACE = [(0.1, 2.5)]

    t_values, dt, alpha_values = load_multi_param_sdde_data(INPUT_DATA_PATH)
    mi_neighbors = MI_NEIGHBORS

    if trajectories is not None:
        print(f"\n--- Starting Bayesian Optimization ---")
        start_time = time.time()

        result = gp_minimize(
            func=objective_function_mi,
            dimensions=SEARCH_SPACE,
            acq_func="EI",
            n_calls=N_CALLS,
            n_random_starts=N_RANDOM_STARTS,
            random_state=111,
            verbose=False,
            noise="gaussian"
        )

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        print(f"\n--- Optimization finished. Time: {duration:.2f}s ---")

        found_tau = result.x[0]
        max_mi = -result.fun

        print(f"\n[Result] Optimal tau: {found_tau:.4f}")
        print(f"         Max mutual information: {max_mi:.6f}")

        run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        pickle_path = os.path.join(OUTPUT_DIR, f'multi_param_optimization_result_{run_timestamp}.pkl')
        json_path = os.path.join(OUTPUT_DIR, f'multi_param_summary_result_{run_timestamp}.json')

        with open(pickle_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"\nSaved full result: {pickle_path}")

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
            'multi_param_info': {
                'alpha_values': alpha_values.tolist(),
                'total_trajectories': trajectories.shape[0],
                'trajectories_per_alpha': trajectories.shape[0] // len(alpha_values)
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
            json.dump(summary_data, f, indent=4)
        print(f"Saved summary: {json_path}")

        print("\nGenerating plots...")

        plt.figure(figsize=(6, 5))
        plot_convergence(result)
        plt.gca().set_title('')
        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel("Objective Function Min Value", fontsize=15)
        plt.tick_params(labelsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        conv_path = os.path.join(OUTPUT_DIR, f'multi_param_convergence_{run_timestamp}.png')
        plt.savefig(conv_path, dpi=100, bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(6, 5))
        evaluated_taus = np.array(result.x_iters).flatten()
        evaluated_errors = result.func_vals
        min_idx = np.argmin(evaluated_errors)

        plt.scatter(evaluated_taus, evaluated_errors, c='skyblue', s=80, alpha=0.8, edgecolors='navy', label='BO Evaluation point')
        plt.scatter(evaluated_taus[min_idx], evaluated_errors[min_idx], c='red', s=200, marker='*', edgecolors='darkred', label=f'Identified τ = {found_tau:.4f}')

        plt.xlabel(r'$\tau$', fontsize=15)
        plt.ylabel(r'$-J(\tau)$', fontsize=15)
        plt.tick_params(labelsize=14)
        plt.legend(frameon=True, fontsize=13.5)
        scatter_path = os.path.join(OUTPUT_DIR, f'multi_param_tau_loss_scatter_{run_timestamp}.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
        plt.show()

