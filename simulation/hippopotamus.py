import numpy as np
from simulation.utils import objective_function

def run_hippo_algorithm(pop_size, max_iter, G, T):
    population = np.random.uniform(0, 100, (pop_size, 1))
    convergence = []
    V_history = []
    best_power = -np.inf

    for _ in range(max_iter):
        G = G * (0.9 + 0.2 * np.random.rand())
        fitness = np.array([objective_function(v[0], G, T) for v in population])
        best_idx = np.argmax(fitness)
        best_now = fitness[best_idx]
        best_voltage = population[best_idx][0]
        convergence.append(best_now)
        V_history.append(best_voltage)

        for i in range(pop_size):
            partner = population[np.random.randint(pop_size)]
            population[i] += np.random.uniform(-1, 1) * (partner - population[i])
            population[i] = np.clip(population[i], 0, 100)

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
