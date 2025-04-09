import numpy as np
from simulation.utils import objective_function

def run_csa_algorithm(pop_size, max_iter, G_init, T, clone_factor=5, mutation_rate=0.2, replacement_rate=0.2):
    population = np.random.uniform(0, 100, (pop_size, 1))
    fitness = np.array([objective_function(v[0], G_init, T) for v in population])

    best_idx = np.argmax(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence = []
    V_history = []

    G = G_init

    for _ in range(max_iter):
        G = G * (0.9 + 0.2 * np.random.rand())

        # Clone phase
        clones = []
        for rank, individual in enumerate(population):
            num_clones = int(np.ceil(clone_factor * (pop_size - rank) / pop_size))
            clones.extend([individual.copy() for _ in range(num_clones)])

        clones = np.array(clones)
        mutated_clones = clones + np.random.uniform(-mutation_rate * 100, mutation_rate * 100, clones.shape)
        mutated_clones = np.clip(mutated_clones, 0, 100)

        clone_fitness = np.array([objective_function(v[0], G, T) for v in mutated_clones])

        # Combine population and mutated clones
        combined = np.vstack((population, mutated_clones))
        combined_fitness = np.concatenate((fitness, clone_fitness))

        # Select best individuals
        sorted_indices = np.argsort(combined_fitness)[::-1]
        population = combined[sorted_indices[:pop_size]]
        fitness = combined_fitness[sorted_indices[:pop_size]]

        # Replace worst individuals with random ones
        n_replace = int(np.ceil(replacement_rate * pop_size))
        if n_replace > 0:
            new_individuals = np.random.uniform(0, 100, (n_replace, 1))
            new_fitness = np.array([objective_function(v[0], G, T) for v in new_individuals])
            replace_indices = np.argsort(fitness)[:n_replace]
            population[replace_indices] = new_individuals
            fitness[replace_indices] = new_fitness

        best_idx = np.argmax(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        convergence.append(best_fitness)
        V_history.append(best_solution[0])

    final_power = float(best_fitness)
    final_voltage = float(best_solution[0])
    threshold = 0.98 * final_power
    conv_time = next((i + 1 for i, p in enumerate(convergence) if p >= threshold), max_iter)

    return {
        "best_power": final_power,
        "final_voltage": final_voltage,
        "convergence": convergence,
        "voltage_history": V_history,
        "convergence_time": conv_time
    }
