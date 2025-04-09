import numpy as np
from simulation.utils import objective_function

def run_gwo_algorithm(pop_size, max_iter, G_init, T):
    positions = np.random.uniform(0, 100, (pop_size, 1))
    fitness = np.array([objective_function(p[0], G_init, T) for p in positions])

    alpha_pos = positions[np.argmax(fitness)].copy()
    alpha_fit = max(fitness)
    beta_pos = alpha_pos.copy()
    delta_pos = alpha_pos.copy()

    convergence = []
    V_history = []

    G = G_init

    for iteration in range(max_iter):
        a = 2 - iteration * (2 / max_iter)
        G = G * (0.9 + 0.2 * np.random.rand())

        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            A1, C1 = 2 * a * r1 - a, 2 * r2
            D_alpha = abs(C1 * alpha_pos - positions[i])
            X1 = alpha_pos - A1 * D_alpha

            r1, r2 = np.random.rand(), np.random.rand()
            A2, C2 = 2 * a * r1 - a, 2 * r2
            D_beta = abs(C2 * beta_pos - positions[i])
            X2 = beta_pos - A2 * D_beta

            r1, r2 = np.random.rand(), np.random.rand()
            A3, C3 = 2 * a * r1 - a, 2 * r2
            D_delta = abs(C3 * delta_pos - positions[i])
            X3 = delta_pos - A3 * D_delta

            new_pos = (X1 + X2 + X3) / 3
            positions[i] = np.clip(new_pos, 0, 100)

        fitness = np.array([objective_function(p[0], G, T) for p in positions])
        sorted_idx = np.argsort(fitness)[::-1]
        alpha_pos = positions[sorted_idx[0]].copy()
        alpha_fit = fitness[sorted_idx[0]]
        beta_pos = positions[sorted_idx[1]].copy()
        delta_pos = positions[sorted_idx[2]].copy()

        convergence.append(alpha_fit)
        V_history.append(alpha_pos[0])

    final_power = float(alpha_fit)
    final_voltage = float(alpha_pos[0])
    threshold = 0.98 * final_power
    conv_time = next((i + 1 for i, p in enumerate(convergence) if p >= threshold), max_iter)

    return {
        "best_power": final_power,
        "final_voltage": final_voltage,
        "convergence": convergence,
        "voltage_history": V_history,
        "convergence_time": conv_time
    }
