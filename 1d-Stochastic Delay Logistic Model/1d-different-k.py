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
        x_matrix, t_values = data["x_matrix"], data["t_values"]
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

if __name__ == "__main__":
    INPUT_DATA_PATH = r"E:\1d\sdde_data.npz"
    OUTPUT_DIR = r"E:\1d
    TRUE_TAU = 1.0
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    K_VALUES_TO_TEST = [5, 10, 15, 20, 25, 30]
    N_SCAN_STEPS = 250
    SEARCH_SPACE = (0.1, 2.5)
    CACHE_FILE_PATH = os.path.join(OUTPUT_DIR, "k_sensitivity_cache_1D.npz")

    trajectories, t_values, dt = load_sdde_data_from_npz(INPUT_DATA_PATH)

    if trajectories is not None:
        print("\n--- Starting k-sensitivity analysis (1D System) ---")

        all_results = {}
        computation_times = {}

        # Load cache if exists
        if os.path.exists(CACHE_FILE_PATH):
            try:
                cached_data = np.load(CACHE_FILE_PATH, allow_pickle=True)
                all_results = cached_data['results'].item()
                if 'runtime' in cached_data:
                    computation_times = cached_data['runtime'].item()
                print(f"Loaded cached results for k={list(all_results.keys())}")
            except Exception as e:
                print(f"Cache load failed: {e}")

        existing_keys = {int(k) for k in all_results.keys()}
        k_to_run = [k for k in K_VALUES_TO_TEST if k not in existing_keys]

        # Calculate missing k values
        if k_to_run:
            print(f"Calculating k values: {k_to_run}")
            for k in k_to_run:
                print(f"Processing k = {k}...")
                start_time = time.time()

                taus_scan = np.linspace(SEARCH_SPACE[0], SEARCH_SPACE[1], N_SCAN_STEPS)
                mi_scores = []
                for tau in taus_scan:
                    Z, dX = prepare_data_for_tau(tau, trajectories, t_values, dt)
                    if Z is None:
                        mi_scores.append(np.nan)
                        continue
                    mi_val = mutual_info_regression(Z, dX[:, 0], n_neighbors=k, random_state=42)[0]
                    mi_scores.append(mi_val)

                all_results[k] = {'taus': taus_scan, 'mis': np.array(mi_scores)}
                elapsed = time.time() - start_time
                computation_times[k] = round(elapsed, 2)
                print(f"k={k} finished in {elapsed:.2f}s")

            # Save cache with runtime
            np.savez(CACHE_FILE_PATH, results=all_results, runtime=computation_times)
            print(f"Results saved to cache: {CACHE_FILE_PATH}")

        # Prepare data for plotting
        sorted_k_values = sorted(all_results.keys())
        identified_taus = []
        summary_list = []
        runtime_list = []

        for k in sorted_k_values:
            data = all_results[k]
            best_tau = data['taus'][np.nanargmax(data['mis'])]
            identified_taus.append(best_tau)
            runtime_list.append(computation_times[k])
            summary_list.append({
                'k Value': k,
                'Identified τ': best_tau,
                'Error |τ - τ_true|': abs(best_tau - TRUE_TAU),
                'Runtime (s)': computation_times[k]
            })

        # Print table
        summary_df = pd.DataFrame(summary_list)
        print("\n" + "="*70)
        print("    Quantitative Stability Analysis (1D System)")
        print(summary_df.to_string(index=False))
        print("="*70)

        # ==================== AUTOMATIC RUNTIME PLOT ====================
        # 修改：画布尺寸改为(10,8)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})

        # Panel a: MI curves
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted_k_values)))
        linestyles = ['--'] * len(sorted_k_values)
        for i, k in enumerate(sorted_k_values):
            data = all_results[k]
            ax1.plot(data['taus'], data['mis'], label=f'k = {k}', color=colors[i],
                     linestyle=linestyles[i], linewidth=2)
        ax1.axvline(x=TRUE_TAU, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('τ', fontsize=20)
        ax1.set_ylabel('J(τ)', fontsize=20)
        ax1.tick_params(labelsize=18)
        ax1.legend(loc='upper right', fontsize=20, frameon=False, ncol=2)

        for spine in ax1.spines.values():
            spine.set_linewidth(2)

        # Panel b: Stability + AUTOMATIC Runtime
        ax_time = ax2.twinx()
        bar_width = 1
        bars = ax_time.bar(sorted_k_values, runtime_list, width=bar_width,
                           color='goldenrod', alpha=0.6)
        ax_time.set_ylabel('Runtime (s)', fontsize=20)
        ax_time.tick_params(labelsize=18)
        ax_time.grid(False)
        ax_time.set_ylim(0, max(runtime_list) * 1.2)

        for i, v in enumerate(runtime_list):
            ax_time.text(sorted_k_values[i], v + max(runtime_list)*0.03, f'{v:.2f}', ha='center', fontsize=16)

        ax2.plot(sorted_k_values, identified_taus, marker='o', markersize=10,
                 linewidth=2, color='royalblue')
        ax2.set_ylabel('Identified τ', fontsize=20)
        ax2.tick_params(labelsize=18)

        error_margin = 0.05 * TRUE_TAU
        ax2.axhspan(TRUE_TAU - error_margin, TRUE_TAU + error_margin, color='gray', alpha=0.15)
        ax2.axhline(y=TRUE_TAU, color='red', linestyle='--', linewidth=2)
        ax2.set_ylim(0.8, 1.2)
        ax2.set_xlabel('k', fontsize=20)

        lines, labels = ax2.get_legend_handles_labels()
        bars, bar_labels = ax_time.get_legend_handles_labels()
        ax2.legend(lines + bars, labels + bar_labels, loc='upper left', fontsize=16, ncol=3, frameon=False)

        for spine in ax2.spines.values():
            spine.set_linewidth(2)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "figure_3_appendix_k_sensitivity_1D_cost_enhanced.png"),
                    dpi=600, bbox_inches="tight")
        plt.show()



