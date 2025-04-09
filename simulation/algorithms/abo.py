import numpy as np
from simulation.utils import objective_function

def run_artificial_butterfly(pop_size, max_iter, G_init, T, p=0.8, c=0.01, a_exp=0.1):
    positions = np.random.uniform(0, 100, (pop_size, 1))
    fitness = np.array([objective_function(x[0], G_init, T) for x in positions])

    best_idx = np.argmax(fitness)
    global_best = positions[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence = []
    V_history = []
    G = G_init

    for _ in range(max_iter):
        G = G * (0.9 + 0.2 * np.random.rand())
        for i in range(pop_size):
            f_i = c * (fitness[i]**a_exp) if fitness[i] != -np.inf else 0
            r = np.random.rand()

            if r < p:
                new_position = positions[i] + r * (global_best - positions[i]) * f_i
            else:
                idxs = np.random.choice(range(pop_size), 2, replace=False)
                new_position = positions[i] + r * (positions[idxs[0]] - positions[idxs[1]]) * f_i

            positions[i] = np.clip(new_position, 0, 100)

        fitness = np.array([objective_function(x[0], G, T) for x in positions])
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            global_best = positions[current_best_idx].copy()
            best_fitness = fitness[current_best_idx]

        convergence.append(best_fitness)
        V_history.append(global_best[0])

    final_power = float(best_fitness)
    final_voltage = float(global_best[0])
    threshold = 0.98 * final_power
    conv_time = next((i + 1 for i, p in enumerate(convergence) if p >= threshold), max_iter)

    return {
        "best_power": final_power,
        "final_voltage": final_voltage,
        "convergence": convergence,
        "voltage_history": V_history,
        "convergence_time": conv_time
    }
