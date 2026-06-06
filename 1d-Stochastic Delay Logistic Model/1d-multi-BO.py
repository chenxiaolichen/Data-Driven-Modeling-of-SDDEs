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


plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=FutureWarning)


trajectories = None
t_values = None
dt = None
mi_neighbors = None


def load_multi_alpha_data_from_npz(npz_path):
    try:
        print(f"Loading multi-alpha data from '{npz_path}'...")
        data = np.load(npz_path)
        t_values = data["t_values"]
        dt = t_values[1] - t_values[0]

        all_trajectories = []
        for key in data.files:
            if key.startswith('x_matrix_alpha_'):
                alpha_traj = data[key]
                all_trajectories.append(alpha_traj)

        trajectories = np.concatenate(all_trajectories, axis=0)
        print(f"Data loaded successfully!")
        print(f"- Total trajectories: {trajectories.shape[0]}")
        print(f"- Time steps: {trajectories.shape[1]}")
        print(f"- Dimensions: {trajectories.shape[2]}")

        if trajectories.ndim == 2:
            trajectories = trajectories[:, :, np.newaxis]
        return trajectories, t_values, dt
    except Exception as e:
        print(f"Data loading failed: {e}")
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

        start_idx = int(np.ceil(tau / dt))
        end_idx = len(t_values) - 2

        for i in range(start_idx, end_idx + 1):
            t_current = t_values[i]
            x_current = traj[i]
            x_delayed = np.array([f(t_current - tau) for f in interp_funcs])
            augmented_states.append(np.concatenate([x_current, x_delayed]))
            increments.append(traj[i + 1] - traj[i])

    Z = np.array(augmented_states)
    dX = np.array(increments)

    if len(Z) < 50:
        return 1e10

    total_mi = 0
    for d in range(dX.shape[1]):
        mi_score = mutual_info_regression(Z, dX[:, d], n_neighbors=mi_neighbors, random_state=42)
        total_mi += mi_score[0]

    objective_value = -total_mi
    print(f"Test tau = {tau:.4f}, Negative MI = {objective_value:.6f}")
    return objective_value


def save_bo_results(result, found_tau, max_mi, duration, hyperparams, input_path, output_dir):
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    pickle_path = os.path.join(output_dir, f'bo_result_multi_alpha_{run_timestamp}.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"\nSaved full results: {pickle_path}")

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
        json.dump(summary_data, f, indent=4)
    print(f"Saved summary: {json_path}")

    return pickle_path, json_path


# --------------------------
# Updated Plotting Function
# --------------------------
def plot_bo_results(result, found_tau, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Convergence Curve
    plt.figure(figsize=(6, 5))
    plt.rcParams['font.family'] = 'Times New Roman'
    plot_convergence(result)
    plt.gca().set_title('')
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Objective Function Min Value", fontsize=15)
    plt.tick_params(labelsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(r"E:\1md\loss.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Tau vs Loss Scatter Plot
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
    plt.ylabel('-J(τ)', fontsize=15)
    plt.tick_params(labelsize=12)
    plt.legend(frameon=True, fontsize=13.5)

    plt.savefig(r"E:\1md\tau_loss_scatter.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    INPUT_DATA_PATH = r"E:\1md\multi_alpha_sdde_data.npz"
    OUTPUT_DIR = r"E:\1md"

    HYPERPARAMS = {
        'mi_neighbors': 10,
        'n_calls': 30,
        'n_random_starts': 10,
        'search_space': [(0.5, 2.5)]
    }



    trajectories, t_values, dt = load_multi_alpha_data_from_npz(INPUT_DATA_PATH)
    mi_neighbors = HYPERPARAMS['mi_neighbors']

    if trajectories is None:
        print("Data loading failed, exiting.")
        exit(1)

    start_time = time.time()
    result = gp_minimize(
        func=objective_function_mi,
        dimensions=HYPERPARAMS['search_space'],
        acq_func="EI",
        n_calls=HYPERPARAMS['n_calls'],
        n_random_starts=HYPERPARAMS['n_random_starts'],
        random_state=42,
        verbose=False,
        noise="gaussian"
    )
    end_time = time.time()
    duration = round(end_time - start_time, 2)

    found_tau = result.x[0]
    max_mi = -result.fun

    save_bo_results(result, found_tau, max_mi, duration, HYPERPARAMS, INPUT_DATA_PATH, OUTPUT_DIR)
    plot_bo_results(result, found_tau, OUTPUT_DIR)