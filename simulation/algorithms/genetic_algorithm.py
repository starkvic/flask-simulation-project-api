import numpy as np
from simulation.utils import objective_function

def run_genetic_algorithm(pop_size, max_iter, G, T, mutation_rate=0.1, crossover_rate=0.8):
    population = np.random.uniform(0, 100, (pop_size, 1))
    convergence = []
    V_history = []
    best_power = -np.inf

    for _ in range(max_iter):
        G = G * (0.9 + 0.2 * np.random.rand())
        fitness = np.array([objective_function(v[0], G, T) for v in population])
        sorted_indices = np.argsort(fitness)[::-1]
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]

        new_population = []
        while len(new_population) < pop_size:
            if np.random.rand() < crossover_rate:
                parents = population[np.random.choice(pop_size, 2, replace=False)]
                alpha = np.random.rand()
                child1 = alpha * parents[0] + (1 - alpha) * parents[1]
                child2 = (1 - alpha) * parents[0] + alpha * parents[1]
                new_population.extend([child1, child2])

        new_population = np.array(new_population)[:pop_size]

        mutation_mask = np.random.rand(*new_population.shape) < mutation_rate
        mutation_values = np.random.uniform(-2, 2, size=new_population.shape) * mutation_mask
        new_population = np.clip(new_population + mutation_values, 0, 100)

        population = new_population
        fitness = np.array([objective_function(v[0], G, T) for v in population])

        best_idx = np.argmax(fitness)
        best_now = fitness[best_idx]
        best_voltage = population[best_idx][0]
        convergence.append(best_now)
        V_history.append(best_voltage)

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
