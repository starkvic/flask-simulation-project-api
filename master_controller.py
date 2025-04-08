import numpy as np
np.random.seed(49)
from simulation.hippopotamus import run_hippo_algorithm
from simulation.pso import run_pso_algorithm
from simulation.genetic_algorithm import run_genetic_algorithm
# import all others...

def run_algorithm(algorithm, pop_size, max_iter, G, T):
    if algorithm == "hippo":
        return run_hippo_algorithm(pop_size, max_iter, G, T)
    elif algorithm == "pso":
        return run_pso_algorithm(pop_size, max_iter, G, T)
    elif algorithm == "ga":
        return  run_genetic_algorithm(pop_size, max_iter, G, T)
    # ...
    else:
        return {"error": f"Algorithm '{algorithm}' is not supported."}
