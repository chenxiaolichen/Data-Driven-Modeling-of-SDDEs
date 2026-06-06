import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.feature_selection import mutual_info_regression
from skopt import gp_minimize
import time
import warnings
import os
import pandas as pd

# Global settings
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

def load_sdde_data_from_npz(npz_path):
    try:
        data = np.load(npz_path)
        x_matrix, t_values = data["x_matrix"], data["t_values"]
        dt = t_values[1] - t_values[0]
        if x_matrix.ndim == 2:
            x_matrix = x_matrix[:, :, np.newaxis]
        return x_matrix, t_values, dt
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None, None, None

def prepare_data_for_tau(tau, trajectories, t_values, dt):
    augmented_states, increments = [], []
    dim = trajectories.shape[2]
    for traj in trajectories:
        interp_funcs = [interp1d(t_values, traj[:, d], kind='cubic', fill_value="extrapolate") for d in range(dim)]
        start_idx, end_idx = int(np.ceil(tau / dt)), len(t_values) - 2
        for i in range(start_idx, end_idx + 1):
            t_curr = t_values[i]
            x_delayed = np.array([f(t_curr - tau) for f in interp_funcs])
            augmented_states.append(np.concatenate([traj[i], x_delayed]))
            increments.append(traj[i+1] - traj[i])
    Z, dX = np.array(augmented_states), np.array(increments)
    if len(Z) < 50:
        return None, None
    return Z, dX

def objective_function_mi_negative(tau_input):
    tau = tau_input[0]
    global trajectories, t_values, dt, mi_neighbors

    if tau < dt or tau > t_values[-1] / 2:
        return 1e10

    Z, dX = prepare_data_for_tau(tau, trajectories, t_values, dt)
    if Z is None:
        return 1e10

    mi_score = mutual_info_regression(Z, dX[:, 0], n_neighbors=mi_neighbors, random_state=42)
    total_mi = mi_score[0]
    return -total_mi

if __name__ == "__main__":
    INPUT_DATA_PATH = r"E:\1d\sdde_data.npz"
    OUTPUT_DIR = r"E:\1d"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MI_NEIGHBORS = 10
    N_CALLS = 30
    N_RANDOM_STARTS = 10
    SEARCH_SPACE = [(0.1, 2.5)]
    N_SCAN_STEPS_GS = 250

    trajectories, t_values, dt = load_sdde_data_from_npz(INPUT_DATA_PATH)
    mi_neighbors = MI_NEIGHBORS
    summary_list = []

    if trajectories is not None:
        print("\n--- Starting Efficiency Comparison ---")

        # Grid Search
        print("Running Grid Search...")
        start_gs = time.time()
        taus_gs = np.linspace(SEARCH_SPACE[0][0], SEARCH_SPACE[0][1], N_SCAN_STEPS_GS)
        mis_gs = []
        for tau in taus_gs:
            mis_gs.append(-objective_function_mi_negative([tau]))
        duration_gs = time.time() - start_gs
        best_idx_gs = np.argmax(mis_gs)
        best_tau_gs = taus_gs[best_idx_gs]
        summary_list.append({
            'Method': 'Grid Search (GS)',
            'Identified τ': best_tau_gs,
            'Runtime (s)': duration_gs,
            'Evaluations': N_SCAN_STEPS_GS
        })

        # Bayesian Optimization
        print("Running Bayesian Optimization...")
        start_bo = time.time()
        result_bo = gp_minimize(
            func=objective_function_mi_negative, dimensions=SEARCH_SPACE, acq_func="EI",
            n_calls=N_CALLS, n_random_starts=N_RANDOM_STARTS,
            random_state=12, verbose=False, noise="gaussian"
        )
        duration_bo = time.time() - start_bo
        taus_bo = np.array(result_bo.x_iters).flatten()
        mis_bo = -np.array(result_bo.func_vals)
        best_tau_bo = result_bo.x[0]
        best_mi_bo = -result_bo.fun
        summary_list.append({
            'Method': 'Bayesian Optimization (BO)',
            'Identified τ': best_tau_bo,
            'Runtime (s)': duration_bo,
            'Evaluations': N_CALLS
        })

        # Random Search
        print("Running Random Search...")
        start_rs = time.time()
        np.random.seed(12)
        taus_rs = np.random.uniform(SEARCH_SPACE[0][0], SEARCH_SPACE[0][1], N_CALLS)
        mis_rs = [-objective_function_mi_negative([tau]) for tau in taus_rs]
        duration_rs = time.time() - start_rs
        best_idx_rs = np.argmax(mis_rs)
        best_tau_rs = taus_rs[best_idx_rs]
        summary_list.append({
            'Method': 'Random Search (RS)',
            'Identified τ': best_tau_rs,
            'Runtime (s)': duration_rs,
            'Evaluations': N_CALLS
        })

        # Print summary table
        print("\n" + "="*80)
        print("          Quantitative Performance Comparison")
        print("="*80)
        summary_df = pd.DataFrame(summary_list)
        summary_df['Identified τ'] = summary_df['Identified τ'].map('{:.4f}'.format)
        summary_df['Runtime (s)'] = summary_df['Runtime (s)'].map('{:.2f}'.format)
        print(summary_df.to_string(index=False))
        print("="*80)

        csv_path = os.path.join(OUTPUT_DIR, "performance_comparison_table.csv")
        summary_df.to_csv(csv_path, index=False)

        # Advanced dual-cost plot
        print("\n--- Generating efficiency comparison figure ---")
        best_mi_gs = np.max(mis_gs)
        best_mi_rs = np.max(mis_rs)

        methods = ['GS', 'RS', 'BO']
        y_positions = [2, 1, 0]
        runtimes = [duration_gs, duration_rs, duration_bo]
        evals = [N_SCAN_STEPS_GS, N_CALLS, N_CALLS]
        colors = ['gray', 'mediumseagreen', 'skyblue']

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=(3, 1.2), height_ratios=(1, 1), wspace=0.05, hspace=0.5)

        ax_main = fig.add_subplot(gs[:, 0])
        ax_time = fig.add_subplot(gs[0, 1])
        ax_eval = fig.add_subplot(gs[1, 1])

        # Main plot
        ax_main.plot(taus_gs, mis_gs, color=colors[0], linestyle='--', linewidth=2.5, label='Grid Search (GS)')
        ax_main.scatter(taus_rs, mis_rs, c=colors[1], s=100, alpha=0.7, edgecolors='darkgreen', marker='s', label='Random Search (RS)')
        ax_main.scatter(taus_bo, mis_bo, c=colors[2], s=100, alpha=0.8, edgecolors='navy', label='Bayesian Optimization (BO)')
        ax_main.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)

        ax_main.set_xlabel('τ', fontsize=25)
        ax_main.set_ylabel('Mutual Information J(τ)', fontsize=25)
        ax_main.tick_params(labelsize=25)
        ax_main.legend(fontsize=19.5, loc='best', ncol=1, frameon=False)
        ax_main.spines['right'].set_visible(False)

        # Annotations
        ax_main.text(0.93, 0.95,
                     f'GS Identified τ = {best_tau_gs:.4f}\n\nRS Identified τ = {best_tau_rs:.4f}\n\nBO Identified τ = {best_tau_bo:.4f}',
                     transform=ax_main.transAxes, fontsize=19,
                     verticalalignment='top', horizontalalignment='right',
                     linespacing=1.0,
                     bbox=dict(boxstyle='round,pad=0.2', edgecolor='gray', facecolor='white', linewidth=1))

        # Border width
        for ax in [ax_main, ax_time, ax_eval]:
            ax.spines['top'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)

        # Runtime bar plot
        bar_height = 0.5
        ax_time.barh(y_positions, runtimes, height=bar_height, color=colors, alpha=0.8)
        ax_time.set_title('Runtime', fontsize=20, pad=0, y=1.0)
        ax_time.set_xlabel('Time (s)', fontsize=20)
        ax_time.tick_params(axis='x', labelsize=20)
        ax_time.set_yticks(y_positions)
        ax_time.set_yticklabels(methods, fontsize=20)
        ax_time.spines['top'].set_visible(False)
        ax_time.spines['right'].set_visible(False)

        for i, v in enumerate(runtimes):
            ax_time.text(v + max(runtimes)*0.05, y_positions[i], f"{v:.2f}s", va='center', fontsize=13)

        # Evaluation bar plot
        ax_eval.barh(y_positions, evals, height=bar_height, color=colors, alpha=0.8)
        ax_eval.set_title('Evaluations', fontsize=20, pad=0, y=1.0)
        ax_eval.set_xlabel('Number', fontsize=20)
        ax_eval.tick_params(axis='x', labelsize=20)
        ax_eval.set_yticks(y_positions)
        ax_eval.set_yticklabels(methods, fontsize=20)
        ax_eval.spines['top'].set_visible(False)
        ax_eval.spines['right'].set_visible(False)

        for i, v in enumerate(evals):
            ax_eval.text(v + max(evals)*0.05, y_positions[i], f"{v}", va='center', fontsize=13)

        # Adjust axis position
        pos = ax_time.get_position()
        ax_time.set_position([pos.x0, pos.y0 - 0.02, pos.width, pos.height])

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "figure_2_efficiency_dual_cost_plot.png"), dpi=600, bbox_inches="tight")
        plt.show()