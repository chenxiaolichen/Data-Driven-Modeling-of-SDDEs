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

trajectories = None
t_values = None
dt = None
mi_neighbors = None

def load_sdde_data_from_npz(npz_path):
    try:
        print(f"Loading data from '{npz_path}'...")
        data = np.load(npz_path)
        x_matrix = data["x_matrix"]
        t_values = data["t_values"]
        dt = t_values[1] - t_values[0]

        if x_matrix.ndim == 2:
            x_matrix = x_matrix[:, :, np.newaxis]

        print("Data loaded successfully!")
        print(f"  - Trajectories: {x_matrix.shape[0]}")
        print(f"  - Shape: {x_matrix.shape}")
        print(f"  - Time step dt: {dt:.4f}")

        return x_matrix, t_values, dt
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
        interp_funcs = [interp1d(t_values, traj[:, d], kind='cubic', fill_value="extrapolate") for d in range(dim)]
        start_idx, end_idx = int(np.ceil(tau / dt)), len(t_values) - 2

        for i in range(start_idx, end_idx + 1):
            t_current, x_current = t_values[i], traj[i]
            x_delayed = np.array([f(t_current - tau) for f in interp_funcs])
            z = np.concatenate([x_current, x_delayed])
            augmented_states.append(z)
            dx = traj[i+1] - traj[i]
            increments.append(dx)

    Z, dX = np.array(augmented_states), np.array(increments)
    if len(Z) < 50:
        return 1e10

    total_mi = 0
    for d in range(dX.shape[1]):
        mi_score = mutual_info_regression(Z, dX[:, d], n_neighbors=mi_neighbors, random_state=42)
        total_mi += mi_score[0]

    objective_value = -total_mi
    print(f"Test tau = {tau:.4f}, Objective (-MI) = {objective_value:.6f}")
    return objective_value

if __name__ == "__main__":
    INPUT_DATA_PATH = r"E:\2d\2d_sdde_data.npz"
    OUTPUT_DIR = r"E:\2d"

    MI_NEIGHBORS = 10
    N_CALLS = 30
    N_RANDOM_STARTS = 15
    SEARCH_SPACE = [(0.1, 2.5)]

    trajectories, t_values, dt = load_sdde_data_from_npz(INPUT_DATA_PATH)
    mi_neighbors = MI_NEIGHBORS

    if trajectories is not None:
        print("\nStarting Bayesian Optimization...")
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
        print(f"\nOptimization finished. Time: {duration:.2f}s")

        found_tau = result.x[0]
        max_mi = -result.fun

        print(f"\n[Result] Optimal tau: {found_tau:.4f}")
        print(f"         Max MI: {max_mi:.6f}")

        run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        pickle_path = os.path.join(OUTPUT_DIR, f'optimization_result_{run_timestamp}.pkl')
        json_path = os.path.join(OUTPUT_DIR, f'summary_result_{run_timestamp}.json')

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
        plt.savefig(r"E:\2d\loss.png", dpi=100, bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(6, 5))
        evaluated_taus = np.array(result.x_iters).flatten()
        evaluated_errors = result.func_vals
        min_idx = np.argmin(evaluated_errors)
        found_tau = evaluated_taus[min_idx]

        plt.scatter(evaluated_taus, evaluated_errors, c='skyblue', s=80, alpha=0.8, edgecolors='navy', label='BO Evaluation point')
        plt.scatter(evaluated_taus[min_idx], evaluated_errors[min_idx], c='red', s=200, marker='*', edgecolors='darkred', label=f'Identified τ = {found_tau:.4f}')

        plt.xlabel('τ', fontsize=15)
        plt.ylabel('-J(τ)', fontsize=15)
        plt.tick_params(labelsize=14)
        plt.legend(frameon=True, fontsize=13.5)

        plt.savefig(r"E:\2d\tau_loss_scatter.png", dpi=300, bbox_inches="tight")
        plt.show()