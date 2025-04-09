import numpy as np
from simulation.utils import objective_function

def run_pso_algorithm(pop_size, max_iter, G_init, T):
    w, c1, c2 = 0.7, 1.5, 1.5
    positions = np.random.uniform(0, 100, (pop_size, 1))
    velocities = np.random.uniform(-10, 10, (pop_size, 1))

    pbest_positions = positions.copy()
    pbest_fitness = np.array([objective_function(v[0], G_init, T) for v in positions])
    gbest_idx = np.argmax(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    convergence = []
    V_history = []

    G = G_init
    for _ in range(max_iter):
        G = G * (0.9 + 0.2 * np.random.rand())

        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest_position - positions[i])
            )
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], 0, 100)

            fitness_val = objective_function(positions[i][0], G, T)
            if fitness_val > pbest_fitness[i]:
                pbest_positions[i] = positions[i].copy()
                pbest_fitness[i] = fitness_val

        current_gbest_idx = np.argmax(pbest_fitness)
        if pbest_fitness[current_gbest_idx] > gbest_fitness:
            gbest_position = pbest_positions[current_gbest_idx].copy()
            gbest_fitness = pbest_fitness[current_gbest_idx]

        convergence.append(gbest_fitness)
        V_history.append(gbest_position[0])

    final_power = float(max(convergence))
    final_voltage = float(V_history[-1])
    threshold = 0.98 * final_power
    conv_time = next((i + 1 for i, p in enumerate(convergence) if p >= threshold), max_iter)

    return {
        "best_power": final_power,
        "final_voltage": final_voltage,
        "convergence": convergence,
        "voltage_history": V_history,
        "convergence_time": conv_time
    }
