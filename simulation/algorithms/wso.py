import numpy as np
from simulation.utils import objective_function

def run_wso_algorithm(pop_size, max_iter, G_init, T):
    positions = np.random.uniform(0, 100, (pop_size, 1))
    current_G = G_init
    convergence = []
    voltage_history = []

    fitness = np.array([objective_function(ind[0], current_G, T) for ind in positions])
    best_index = np.argmax(fitness)
    global_best = positions[best_index].copy()
    best_fitness = fitness[best_index]

    for iteration in range(max_iter):
        current_G *= (0.9 + 0.2 * np.random.rand())
        hunting_factor = np.exp(-iteration / max_iter)

        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            new_position = positions[i] + r1 * hunting_factor * (global_best - positions[i]) \
                           + r2 * (np.random.rand() - 0.5) * 10
            positions[i] = np.clip(new_position, 0, 100)

        fitness = np.array([objective_function(ind[0], current_G, T) for ind in positions])
        current_best_index = np.argmax(fitness)

        if fitness[current_best_index] > best_fitness:
            global_best = positions[current_best_index].copy()
            best_fitness = fitness[current_best_index]

        convergence.append(best_fitness)
        voltage_history.append(global_best[0])

    final_power = float(max(convergence))
    final_voltage = float(voltage_history[-1])
    threshold = 0.98 * final_power
    conv_time = next((i + 1 for i, p in enumerate(convergence) if p >= threshold), max_iter)

    return {
        "best_power": final_power,
        "final_voltage": final_voltage,
        "convergence": convergence,
        "voltage_history": voltage_history,
        "convergence_time": conv_time
    }
