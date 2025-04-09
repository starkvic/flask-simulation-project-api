import numpy as np
from simulation.utils import objective_function

def run_de_algorithm(pop_size, max_iter, G_init, T, F=0.8, CR=0.9):
    population = np.random.uniform(0, 100, (pop_size, 1))
    current_G = G_init
    convergence = []
    voltage_history = []

    fitness = np.array([objective_function(ind[0], current_G, T) for ind in population])
    best_index = np.argmax(fitness)
    global_best = population[best_index].copy()
    best_fitness = fitness[best_index]

    for iteration in range(max_iter):
        current_G *= (0.9 + 0.2 * np.random.rand())
        new_population = np.empty_like(population)

        for i in range(pop_size):
            indices = list(range(pop_size))
            indices.remove(i)
            r1, r2, r3 = population[np.random.choice(indices, 3, replace=False)]

            mutant = r1 + F * (r2 - r3)
            trial = mutant if np.random.rand() < CR else population[i]
            new_population[i] = np.clip(trial, 0, 100)

        new_fitness = np.array([objective_function(ind[0], current_G, T) for ind in new_population])

        for i in range(pop_size):
            if new_fitness[i] > fitness[i]:
                population[i] = new_population[i]
                fitness[i] = new_fitness[i]

        best_index = np.argmax(fitness)
        if fitness[best_index] > best_fitness:
            global_best = population[best_index].copy()
            best_fitness = fitness[best_index]

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
