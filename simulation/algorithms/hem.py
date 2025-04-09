import numpy as np
from simulation.utils import objective_function

def run_human_evolutionary(pop_size, max_iter, G_init, T,
                           selection_rate=0.3, mutation_rate=0.1,
                           crossover_rate=0.7, diversity_rate=0.1):
    
    population = np.random.uniform(0, 100, (pop_size, 1))
    G = G_init
    fitness = np.array([objective_function(ind[0], G, T) for ind in population])

    best_idx = np.argmax(fitness)
    global_best = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence = []
    V_history = []

    for _ in range(max_iter):
        G = G * (0.9 + 0.2 * np.random.rand())

        elite_count = int(np.ceil(selection_rate * pop_size))
        sorted_idx = np.argsort(fitness)[::-1]
        elites = population[sorted_idx[:elite_count]]

        children = []
        while len(children) < pop_size - elite_count:
            if np.random.rand() < crossover_rate:
                p1, p2 = elites[np.random.choice(elite_count, 2, replace=False)]
                weight = np.random.rand()
                child = weight * p1 + (1 - weight) * p2
            else:
                child = elites[np.random.randint(elite_count)]
            child += np.random.uniform(-mutation_rate * 100, mutation_rate * 100, size=child.shape)
            child = np.clip(child, 0, 100)
            children.append(child)

        children = np.array(children)
        r = np.random.rand(*children.shape)
        children = children + r * (global_best - children)
        children = np.clip(children, 0, 100)

        diversity_count = int(np.ceil(diversity_rate * pop_size))
        random_individuals = np.random.uniform(0, 100, (diversity_count, 1))
        
        new_pop = np.vstack((elites, children))
        if new_pop.shape[0] > pop_size:
            new_pop = new_pop[:pop_size]
        else:
            new_pop[-diversity_count:] = random_individuals

        population = new_pop
        fitness = np.array([objective_function(ind[0], G, T) for ind in population])
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_fitness:
            global_best = population[best_idx].copy()
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
