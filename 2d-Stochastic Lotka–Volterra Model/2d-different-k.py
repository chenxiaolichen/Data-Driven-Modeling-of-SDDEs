import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.feature_selection import mutual_info_regression
import time
import warnings
import os
import pandas as pd

plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

def load_sdde_data_from_npz(npz_path):
    try:
        data = np.load(npz_path)
        x_matrix = data["x_matrix"]
        t_values = data["t_values"]
        dt = t_values[1] - t_values[0]
        if x_matrix.ndim == 2:
            x_matrix = x_matrix[:, :, np.newaxis]
        return x_matrix, t_values, dt
    except Exception as e:
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

def calculate_mi_for_tau(tau, trajectories, t_values, dt, k_neighbors):
    if tau < dt or tau > t_values[-1] / 2:
        return np.nan
    Z, dX = prepare_data_for_tau(tau, trajectories, t_values, dt)
    if Z is None:
        return np.nan
    total_mi = 0
    for d in range(dX.shape[1]):
        mi_score = mutual_info_regression(Z, dX[:, d], n_neighbors=k_neighbors, random_state=42)
        total_mi += mi_score[0]
    return total_mi

if __name__ == "__main__":
    INPUT_DATA_PATH = r"E:\2d\2d_sdde_data.npz"
    OUTPUT_DIR = r"E:\2d"
    TRUE_TAU = 1.0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    K_VALUES_TO_TEST = [5, 10, 15, 20, 25, 30]
    N_SCAN_STEPS = 250
    SEARCH_SPACE = (0.1, 2.5)
    CACHE_FILE_PATH = os.path.join(OUTPUT_DIR, "k_sensitivity_cache_2D.npz")

    trajectories, t_values, dt = load_sdde_data_from_npz(INPUT_DATA_PATH)

    if trajectories is not None:
        print("\n--- Starting k-sensitivity analysis (2D System) ---")

        all_results = {}
        computation_times = []

        if os.path.exists(CACHE_FILE_PATH):
            print(f"Loading cache: {CACHE_FILE_PATH}")
            try:
                cached_data = np.load(CACHE_FILE_PATH, allow_pickle=True)
                all_results = cached_data['results'].item()
                computation_times = cached_data['times'].tolist()
                print(f"Loaded {len(all_results)} results: k={list(all_results.keys())}")
            except Exception as e:
                print(f"Cache load failed: {e}")
                all_results = {}
                computation_times = []

        existing_keys = {int(k) for k in all_results.keys()}
        k_to_run = [k for k in K_VALUES_TO_TEST if k not in existing_keys]

        if k_to_run:
            print(f"\nCalculating k: {k_to_run}")
            for k in k_to_run:
                print(f"Calculating k={k}...")
                t0 = time.time()
                taus_scan = np.linspace(SEARCH_SPACE[0], SEARCH_SPACE[1], N_SCAN_STEPS)
                mi_scores = [calculate_mi_for_tau(tau, trajectories, t_values, dt, k) for tau in taus_scan]
                elapsed = time.time() - t0
                all_results[k] = {'taus': taus_scan, 'mis': np.array(mi_scores)}
                computation_times.append(round(elapsed, 2))
                print(f"k={k} done. Time: {elapsed:.2f}s")

            np.savez(CACHE_FILE_PATH, results=all_results, times=computation_times)
            print(f"\nCache saved to: {CACHE_FILE_PATH}")

        sorted_k_values = sorted(all_results.keys())

        identified_taus = []
        for k in sorted_k_values:
            data = all_results[k]
            best_tau = data['taus'][np.nanargmax(data['mis'])]
            identified_taus.append(best_tau)

        # Table
        summary_list = []
        for i, k in enumerate(sorted_k_values):
            data = all_results[k]
            best_idx = np.nanargmax(data['mis'])
            best_tau = data['taus'][best_idx]
            error = abs(best_tau - TRUE_TAU)
            runtime = computation_times[i] if i < len(computation_times) else 0
            summary_list.append({
                'k Value': k,
                'Identified τ': best_tau,
                'Error |τ - τ_true|': error,
                'Runtime (s)': runtime
            })

        summary_df = pd.DataFrame(summary_list)

        summary_df['Identified τ'] = summary_df['Identified τ'].map('{:.4f}'.format)
        summary_df['Error |τ - τ_true|'] = summary_df['Error |τ - τ_true|'].map('{:.4f}'.format)
        summary_df['Runtime (s)'] = summary_df['Runtime (s)'].map('{:.2f}'.format)

        csv_path = os.path.join(OUTPUT_DIR, "k_sensitivity_table_2D_full.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"Table saved to: {csv_path}")



        # -------------------- Plot 1: MI Landscape --------------------
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        colors = plt.cm.plasma(np.linspace(0.0, 0.9, len(sorted_k_values)))
        linestyles = ['--'] * len(sorted_k_values)

        for i, k in enumerate(sorted_k_values):
            data = all_results[k]
            ax1.plot(data['taus'], data['mis'], label=f'k = {k}',
                     color=colors[i], linestyle=linestyles[i], linewidth=2)

        ax1.axvline(x=TRUE_TAU, color='red', linestyle='--', linewidth=2.5)
        ax1.set_xlabel('τ', fontsize=34)
        ax1.set_ylabel('J(τ)', fontsize=34)
        ax1.tick_params(labelsize=30)
        ax1.legend(loc='upper right', fontsize=28, frameon=True)

        for spine in ax1.spines.values():
            spine.set_linewidth(2)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "figure_2D_mi_landscape.png"),
                    dpi=600, bbox_inches="tight")
        plt.show()

        # -------------------- Plot 2: Stability & Cost --------------------
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax_time = ax2.twinx()

        bar_width = 2
        ax_time.bar(sorted_k_values, computation_times, width=bar_width,
                    color='goldenrod', alpha=0.6)
        ax_time.set_ylabel('Runtime (s)', fontsize=34)
        ax_time.tick_params(labelsize=30)
        ax_time.grid(False)

        max_time = max(computation_times)
        ax_time.set_ylim(0, max_time * 1.15)
        for i, v in enumerate(computation_times):
            ax_time.text(sorted_k_values[i], v + max_time * 0.02,
                         f'{v:.2f}', ha='center', fontsize=20)

        ax2.plot(sorted_k_values, identified_taus, marker='s', markersize=10,
                 linewidth=2, color='royalblue', label='Identified τ')
        ax2.set_ylabel('Identified τ', fontsize=34)
        ax2.tick_params(labelsize=30)

        error_margin = 0.05 * TRUE_TAU
        ax2.axhspan(TRUE_TAU - error_margin, TRUE_TAU + error_margin,
                    color='gray', alpha=0.15, label='±5% Error Margin')
        ax2.axhline(y=TRUE_TAU, color='red', linestyle='--', linewidth=2)
        ax2.set_ylim(0.8, 1.2)
        ax2.set_xlabel('k', fontsize=34)

        lines, labels = ax2.get_legend_handles_labels()
        bar_hands, bar_labels = ax_time.get_legend_handles_labels()
        ax2.legend(lines + bar_hands, labels + bar_labels,
                    loc='upper left', fontsize=22, ncol=2, frameon=False)

        for spine in ax2.spines.values():
            spine.set_linewidth(2)
        for spine in ax_time.spines.values():
            spine.set_linewidth(2)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "figure_2D_stability_cost.png"),
                    dpi=600, bbox_inches="tight")
        plt.show()