import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import gaussian_kde
from skopt import gp_minimize
from scipy.optimize import minimize
import time, warnings, os
import pickle

# ==============================================================================
# Global Settings
# ==============================================================================
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

RE_RUN_OPTIMIZATIONS = True

INPUT_DATA_PATH = r"E:\1d\sdde_data.npz"
OUTPUT_DIR = r"E:\1d"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "comparison_results_1d.pkl")

MI_NEIGHBORS = 10
TOTAL_EVALUATIONS = 30
N_RANDOM_STARTS_BO = 10
SEARCH_SPACE = [(0.1, 2.5)]
RANDOM_SEED = 42

evaluation_counter, evaluation_history = 0, []

class BudgetExceededError(Exception):
    pass

def reset_evaluation_tracker():
    global evaluation_counter, evaluation_history
    evaluation_counter, evaluation_history = 0, []

def generate_samples(tau):
    global trajectories, t_values, dt
    augmented_states, increments = [], []
    dim = trajectories.shape[2]
    if dim == 0:
        return None, None
    for traj in trajectories:
        interp_funcs = [interp1d(t_values, traj[:, d], kind='cubic', fill_value="extrapolate") for d in range(dim)]
        start_idx = int(np.ceil(tau / dt))
        end_idx = len(t_values) - 2
        for i in range(start_idx, end_idx + 1):
            t_current = t_values[i]
            x_current = traj[i]
            x_delayed = np.array([float(f(t_current - tau)) for f in interp_funcs])
            augmented_states.append(np.concatenate([x_current, x_delayed]))
            increments.append(traj[i+1] - traj[i])
    if not augmented_states:
        return None, None
    return np.array(augmented_states), np.array(increments)

def objective_function_mi(tau_list):
    global evaluation_counter, evaluation_history, dt, trajectories
    if evaluation_counter >= TOTAL_EVALUATIONS:
        raise BudgetExceededError()
    evaluation_counter += 1
    tau = float(tau_list[0])
    if tau < dt or tau > t_values[-1] / 2:
        return 1e10
    Z, dX = generate_samples(tau)
    if Z is None or len(Z) < 50:
        return 1e10
    total_mi = 0.0
    for d in range(dX.shape[1]):
        mi = mutual_info_regression(Z, dX[:, d], n_neighbors=MI_NEIGHBORS, random_state=42)[0]
        total_mi += mi
    objective_value = -total_mi
    evaluation_history.append((tau, objective_value))
    return objective_value

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
        print(f"Data loading failed: {e}")
        return None, None, None

def calculate_sfd_gradient(tau, h=1e-5):
    Z, _ = generate_samples(tau)
    if Z is None or Z.shape[0] < 50:
        return 0.0
    n_samples, z_dim = Z.shape
    try:
        joint_kde = gaussian_kde(Z.T)
        marginal_kdes = [gaussian_kde(Z[:, i]) for i in range(z_dim)]
    except:
        return 0.0
    jsf = np.zeros_like(Z)
    msf = np.zeros_like(Z)
    for i in range(z_dim):
        Z_plus_h = Z.copy()
        Z_minus_h = Z.copy()
        Z_plus_h[:, i] += h
        Z_minus_h[:, i] -= h
        jsf[:, i] = -(joint_kde.logpdf(Z_plus_h.T) - joint_kde.logpdf(Z_minus_h.T)) / (2 * h)
        msf[:, i] = -(marginal_kdes[i].logpdf(Z[:, i] + h) - marginal_kdes[i].logpdf(Z[:, i] - h)) / (2 * h)
    sfd = msf - jsf
    data_dim = trajectories.shape[2]
    dZ_dtau = np.zeros_like(Z)
    sample_count = 0
    for traj in trajectories:
        interp_funcs = [CubicSpline(t_values, traj[:, d], extrapolate=True) for d in range(data_dim)]
        interp_derivs = [f.derivative() for f in interp_funcs]
        start_idx = int(np.ceil(tau / dt))
        end_idx = len(t_values) - 2
        for t_idx in range(start_idx, end_idx + 1):
            if sample_count < n_samples:
                t_current = t_values[t_idx]
                delayed_derivs = np.array([-f_deriv(t_current - tau) for f_deriv in interp_derivs])
                dZ_dtau[sample_count, data_dim:] = delayed_derivs
                sample_count += 1
    grad = np.mean(np.sum(sfd * dZ_dtau, axis=1))
    return grad

def run_sfd_gradient_descent(start_tau, learning_rate):
    global evaluation_counter, evaluation_history
    tau = start_tau
    try:
        while evaluation_counter < TOTAL_EVALUATIONS:
            grad = calculate_sfd_gradient(tau)
            evaluation_counter += 1
            evaluation_history.append((tau, np.nan))
            tau = np.clip(tau - learning_rate * grad, SEARCH_SPACE[0][0], SEARCH_SPACE[0][1])
            if evaluation_counter < TOTAL_EVALUATIONS:
                objective_function_mi([tau])
            else:
                break
    except BudgetExceededError:
        pass
    temp_hist = list(evaluation_history)
    for i in range(len(temp_hist) - 1):
        if np.isnan(temp_hist[i][1]):
            temp_hist[i] = (temp_hist[i][0], temp_hist[i+1][1])
    evaluation_history[:] = [item for item in temp_hist if not np.isnan(item[1])][:TOTAL_EVALUATIONS]
    if evaluation_history:
        best_pt = min(evaluation_history, key=lambda x: x[1])
        return best_pt[0], best_pt[1]
    return start_tau, float('inf')


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trajectories, t_values, dt = load_sdde_data_from_npz(INPUT_DATA_PATH)
    if trajectories is None:
        exit()

    results = {}

    if RE_RUN_OPTIMIZATIONS:
        print(f"Running optimizations...")

        # Bayesian Optimization
        print("Running Bayesian Optimization (BO)")
        reset_evaluation_tracker()
        st = time.time()
        bo_res = gp_minimize(objective_function_mi, SEARCH_SPACE, acq_func="EI",
                            n_calls=TOTAL_EVALUATIONS, n_random_starts=N_RANDOM_STARTS_BO,
                            random_state=RANDOM_SEED, verbose=False)
        dur = time.time() - st
        results['Bayesian Optimization (BO)'] = {
            'tau': bo_res.x[0], 'min_val': bo_res.fun,
            'eval_history': evaluation_history, 'duration': dur
        }

        # L-BFGS-B
        print("Running L-BFGS-B")
        reset_evaluation_tracker()
        st = time.time()
        try:
            minimize(objective_function_mi, x0=[1.5], method='L-BFGS-B', bounds=SEARCH_SPACE)
        except BudgetExceededError:
            pass
        dur = time.time() - st
        if evaluation_history:
            best = min(evaluation_history, key=lambda x: x[1])
            results['L-BFGS-B'] = {
                'tau': best[0], 'min_val': best[1],
                'eval_history': evaluation_history, 'duration': dur
            }

        # SFD Gradient Descent
        print("Running SFD Gradient (SFD-GD)")
        reset_evaluation_tracker()
        st = time.time()
        sfd_tau, sfd_val = run_sfd_gradient_descent(start_tau=1.5, learning_rate=0.01)
        dur = time.time() - st
        results['SFD Gradient (SFD-GD)'] = {
            'tau': sfd_tau, 'min_val': sfd_val,
            'eval_history': evaluation_history, 'duration': dur
        }

        with open(RESULTS_FILE, 'wb') as f:
            pickle.dump(results, f)
        print("Results saved.")
    else:
        print("Loading saved results...")
        with open(RESULTS_FILE, 'rb') as f:
            results = pickle.load(f)


    colors = {'BO': 'blue', 'SFD': 'darkorange', 'L-BFGS-B': 'forestgreen'}
    markers = {'BO': '*', 'SFD': 'P', 'L-BFGS-B': 'x'}
    plot_order = ['L-BFGS-B', 'SFD', 'BO']
    method_map = {
        'BO': 'Bayesian Optimization (BO)',
        'SFD': 'SFD Gradient (SFD-GD)',
        'L-BFGS-B': 'L-BFGS-B'
    }

    # Convergence Curve
    plt.figure(figsize=(10, 8))
    ax_conv = plt.gca()
    for name in plot_order:
        fn = method_map[name]
        if fn in results:
            res = results[fn]
            hist = res.get('eval_history', [])[:TOTAL_EVALUATIONS]
            vals = [h[1] for h in hist if h[1] is not None and not np.isnan(h[1])]
            if vals:
                best = np.minimum.accumulate(vals)
                plt.plot(np.arange(1, len(best)+1), best, marker='.', lw=2,
                        color=colors[name], label=name)
    for spine in ax_conv.spines.values():
        spine.set_linewidth(2)
    plt.xlabel('Number of Evaluations', fontsize=30)
    plt.ylabel('-J(τ)', fontsize=34)
    plt.tick_params(labelsize=34)
    plt.legend(fontsize=34, frameon=False, loc='upper right')
    plt.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, "convergence_curve_3methods.png")
    plt.savefig(p1, dpi=600, bbox_inches='tight')
    plt.show()

    # Composite Plot
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,8), sharey=True,
                                 gridspec_kw={'width_ratios':[3,1.5]})
    fig.subplots_adjust(wspace=0.05)
    y_pos = np.arange(1, len(plot_order)+1)
    runtimes = []

    for i, name in enumerate(plot_order):
        fn = method_map[name]
        if fn in results:
            res = results[fn]
            runtimes.append(res.get('duration',0))
            hist = res.get('eval_history', [])[:TOTAL_EVALUATIONS]
            taus = [p[0] for p in hist]
            ax1.scatter(taus, np.ones_like(taus)*y_pos[i], c=colors[name],
                      marker=markers[name], s=100, alpha=0.7, ec='black', lw=0.5)
            ft = res.get('tau')
            if ft is not None:
                ax1.text(ft, y_pos[i]+0.2, f'τ={ft:.3f}', ha='center', va='bottom',
                        fontsize=25, fontweight='bold', color=colors[name])

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(plot_order, fontsize=25)
    ax1.set_xlabel('τ', fontsize=34)
    ax1.set_xlim(SEARCH_SPACE[0][0], SEARCH_SPACE[0][1])
    ax1.set_ylim(0.5, len(plot_order)+0.5)
    ax1.tick_params(axis='x', labelsize=25)
    ax1.grid(True, axis='x', ls=':', alpha=0.5)
    ax1.spines['right'].set_visible(False)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)

    ax2.barh(y_pos, runtimes, color=[colors[n] for n in plot_order], alpha=0.8, ec='black')
    ax2.set_xscale('log')
    for i, t in enumerate(runtimes):
        ax2.text(t*1.2, y_pos[i], f'{t:.2f} s', va='center', ha='left', fontsize=15)
    ax2.set_xlabel('Runtime', fontsize=30)
    ax2.tick_params(axis='x', labelsize=25)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, axis='x', ls=':', alpha=0.5)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)

    p2 = os.path.join(OUTPUT_DIR, "composite_samples_runtime_3methods.png")
    plt.savefig(p2, dpi=600, bbox_inches='tight')
    plt.show()

   