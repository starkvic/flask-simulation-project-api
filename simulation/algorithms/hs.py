import numpy as np
from simulation.utils import objective_function

def run_hs_algorithm(hm_size, max_iter, G_init, T, HMCR=0.9, PAR=0.3, bw=5):
    harmony_memory = np.random.uniform(0, 100, (hm_size, 1))
    fitness_memory = np.array([objective_function(v[0], G_init, T) for v in harmony_memory])
    best_idx = np.argmax(fitness_memory)
    best_harmony = harmony_memory[best_idx].copy()
    best_fitness = fitness_memory[best_idx]

    convergence = []
    V_history = []

    G = G_init

    for _ in range(max_iter):
        G = G * (0.9 + 0.2 * np.random.rand())

        if np.random.rand() < HMCR:
            new_value = np.random.choice(harmony_memory.flatten())
            if np.random.rand() < PAR:
                new_value += np.random.uniform(-bw, bw)
        else:
            new_value = np.random.uniform(0, 100)

        new_value = np.clip(new_value, 0, 100)
        new_fitness = objective_function(new_value, G, T)

        worst_idx = np.argmin(fitness_memory)
        if new_fitness > fitness_memory[worst_idx]:
            harmony_memory[worst_idx] = new_value
            fitness_memory[worst_idx] = new_fitness

            best_idx = np.argmax(fitness_memory)
            best_harmony = harmony_memory[best_idx].copy()
            best_fitness = fitness_memory[best_idx]

        convergence.append(best_fitness)
        V_history.append(best_harmony[0])

    final_power = float(best_fitness)
    final_voltage = float(best_harmony[0])
    threshold = 0.98 * final_power
    conv_time = next((i + 1 for i, p in enumerate(convergence) if p >= threshold), max_iter)

    return {
        "best_power": final_power,
        "final_voltage": final_voltage,
        "convergence": convergence,
        "voltage_history": V_history,
        "convergence_time": conv_time
    }
