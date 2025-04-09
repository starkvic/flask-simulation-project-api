import numpy as np
from simulation.utils import objective_function

def run_tlbo_algorithm(pop_size, max_iter, G, T):
    population = np.random.uniform(0, 100, (pop_size, 1))
    convergence = []
    V_history = []
    best_power = -np.inf

    for _ in range(max_iter):
        G = G * (0.9 + 0.2 * np.random.rand())
        fitness = np.array([objective_function(v, G, T) for v in population])
        teacher = population[np.argmax(fitness)]
        mean = np.mean(population, axis=0)
        TF = np.random.randint(1, 3)

        for i in range(pop_size):
            new_solution = population[i] + np.random.rand() * (teacher - TF * mean)
            new_solution = np.clip(new_solution, 0, 100)
            new_fitness = objective_function([new_solution[0]], G, T)
            current_fitness = objective_function([population[i][0]], G, T)
            if new_fitness > current_fitness:
                population[i] = new_solution

        best_idx = np.argmax([objective_function(v[0], G, T) for v in population])
        best_now = objective_function(population[best_idx][0], G, T)
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
#done