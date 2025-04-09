import numpy as np
from math import gamma, sin, pi
from simulation.utils import objective_function

def levy_flight(beta):
    sigma_u = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    u = np.random.randn() * sigma_u
    v = np.random.randn()
    step = u / (abs(v) ** (1 / beta))
    return step

def run_cuckoo_search(n_nests, max_iter, G_init, T, pa=0.25, alpha=0.01, beta=1.5):
    nests = np.random.uniform(0, 100, (n_nests, 1))
    fitness = np.array([objective_function(n[0], G_init, T) for n in nests])

    best_idx = np.argmax(fitness)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence = []
    V_history = []
    G = G_init

    for _ in range(max_iter):
        G = G * (0.9 + 0.2 * np.random.rand())

        # Levy Flights
        new_nests = np.empty_like(nests)
        for i in range(n_nests):
            step = levy_flight(beta)
            new_nest = nests[i] + alpha * step
            new_nests[i] = np.clip(new_nest, 0, 100)

        new_fitness = np.array([objective_function(n[0], G, T) for n in new_nests])

        # Replace with better nests
        for i in range(n_nests):
            j = np.random.randint(n_nests)
            if new_fitness[i] > fitness[j]:
                nests[j] = new_nests[i]
                fitness[j] = new_fitness[i]

        # Abandon some nests
        num_abandoned = int(pa * n_nests)
        worst_indices = np.argsort(fitness)[:num_abandoned]
        nests[worst_indices] = np.random.uniform(0, 100, (num_abandoned, 1))
        fitness[worst_indices] = np.array([objective_function(n[0], G, T) for n in nests[worst_indices]])

        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            best_nest = nests[current_best_idx].copy()
            best_fitness = fitness[current_best_idx]

        convergence.append(best_fitness)
        V_history.append(best_nest[0])

    final_power = float(best_fitness)
    final_voltage = float(best_nest[0])
    threshold = 0.98 * final_power
    conv_time = next((i + 1 for i, p in enumerate(convergence) if p >= threshold), max_iter)

    return {
        "best_power": final_power,
        "final_voltage": final_voltage,
        "convergence": convergence,
        "voltage_history": V_history,
        "convergence_time": conv_time
    }
