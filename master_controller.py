import numpy as np
np.random.seed(49)
from simulation.algorithms.hippopotamus import run_hippo_algorithm
from simulation.algorithms.tlbo import run_tlbo_algorithm
from simulation.algorithms.genetic import run_genetic_algorithm
from simulation.algorithms.pso import run_pso_algorithm
from simulation.algorithms.sa import run_sa_algorithm
from simulation.algorithms.gwo import run_gwo_algorithm
from simulation.algorithms.hs import run_hs_algorithm
from simulation.algorithms.csa import run_csa_algorithm
from simulation.algorithms.epo import run_epo_algorithm
from simulation.algorithms.cuckoo import run_cs_algorithm
from simulation.algorithms.abo import run_abo_algorithm
from simulation.algorithms.hem import run_hem_algorithm
from simulation.algorithms.wso import run_wso_algorithm
from simulation.algorithms.abc import run_abc_algorithm

# Optional: Add these once you've created them
# from simulation.algorithms.lsa import run_lsa_algorithm
# from simulation.algorithms.de import run_de_algorithm

ALGORITHM_MAP = {
    "hippo": run_hippo_algorithm,
    "tlbo": run_tlbo_algorithm,
    "ga": run_genetic_algorithm,
    "pso": run_pso_algorithm,
    "sa": run_sa_algorithm,
    "gwo": run_gwo_algorithm,
    "hs": run_hs_algorithm,
    "csa": run_csa_algorithm,
    "epo": run_epo_algorithm,
    "cs": run_cs_algorithm,
    "abo": run_abo_algorithm,
    "hem": run_hem_algorithm,
    "wso": run_wso_algorithm,
    "abc": run_abc_algorithm,
    # "lsa": run_lsa_algorithm,
    # "de": run_de_algorithm,
}

def run_algorithm(algorithm_name, pop_size, max_iter, irradiance, temperature):
    algorithm = ALGORITHM_MAP.get(algorithm_name)
    if algorithm is None:
        raise ValueError(f"Algorithm '{algorithm_name}' not supported.")
    return algorithm(pop_size, max_iter, irradiance, temperature)
