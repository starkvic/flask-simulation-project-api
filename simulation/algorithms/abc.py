import numpy as np
from simulation.utils import objective_function

def run_abc_algorithm(num_sources, max_iter, G_init, T, limit=20):
    food_sources = np.random.uniform(0, 100, (num_sources, 1))
    fitness = np.array([objective_function(v[0], G_init, T) for v in food_sources])
    trial = np.zeros(num_sources)
    best_index = np.argmax(fitness)
    global_best = food_sources[best_index].copy()
    best_fitness = fitness[best_index]

    convergence = []
    voltage_history = []
    current_G = G_init

    for _ in range(max_iter):
        # Employed Bees Phase
        for i in range(num_sources):
            k = np.random.choice([j for j in range(num_sources) if j != i])
            phi = np.random.uniform(-1, 1)
            new_solution = food_sources[i] + phi * (food_sources[i] - food_sources[k])
            new_solution = np.clip(new_solution, 0, 100)
            new_fitness = objective_function(new_solution[0], current_G, T)

            if new_fitness > fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness
                trial[i] = 0
            else:
                trial[i] += 1

        # Calculate probabilities
        total_fit = np.sum(fitness[fitness > -np.inf])
        probabilities = fitness / total_fit if total_fit > 0 else np.ones(num_sources) / num_sources

        # Onlooker Bees Phase
        for i in range(num_sources):
            if np.random.rand() < probabilities[i]:
                k = np.random.choice([j for j in range(num_sources) if j != i])
                phi = np.random.uniform(-1, 1)
                new_solution = food_sources[i] + phi * (food_sources[i] - food_sources[k])
                new_solution = np.clip(new_solution, 0, 100)
                new_fitness = objective_function(new_solution[0], current_G, T)

                if new_fitness > fitness[i]:
                    food_sources[i] = new_solution
                    fitness[i] = new_fitness
                    trial[i] = 0
                else:
                    trial[i] += 1

        # Scout Bee Phase
        for i in range(num_sources):
            if trial[i] > limit:
                food_sources[i] = np.random.uniform(0, 100, (1,))
                fitness[i] = objective_function(food_sources[i][0], current_G, T)
                trial[i] = 0

        best_index = np.argmax(fitness)
        if fitness[best_index] > best_fitness:
            global_best = food_sources[best_index].copy()
            best_fitness = fitness[best_index]

        current_G *= (0.9 + 0.2 * np.random.rand())
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
