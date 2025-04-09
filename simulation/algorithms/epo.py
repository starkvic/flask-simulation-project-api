import numpy as np
from simulation.utils import objective_function

def run_epo_algorithm(pop_size, max_iter, G_init, T):
    positions = np.random.uniform(0, 100, (pop_size, 1))
    fitness = np.array([objective_function(pos[0], G_init, T) for pos in positions])

    best_idx = np.argmax(fitness)
    global_best = positions[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence = []
    V_history = []
    G = G_init

    for it in range(max_iter):
        G = G * (0.9 + 0.2 * np.random.rand())
        a = 2 - it * (2 / max_iter)
        swarm_mean = np.mean(positions)

        for i in range(pop_size):
            r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
            new_position = positions[i] \
                + a * r1 * (global_best - positions[i]) \
                + 0.5 * r2 * (swarm_mean - positions[i]) \
                + 0.1 * r3 * 100 * (np.random.rand() - 0.5)

            positions[i] = np.clip(new_position, 0, 100)

        fitness = np.array([objective_function(pos[0], G, T) for pos in positions])
        best_idx = np.argmax(fitness)

        if fitness[best_idx] > best_fitness:
            global_best = positions[best_idx].copy()
            best_fitness = fitness[best_idx]

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
#done