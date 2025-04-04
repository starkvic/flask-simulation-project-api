import numpy as np

def objective_function(V, G, T):
    return V * (G / 1000) * (1 - np.exp(-V / 100)) * (1 - 0.005 * (T - 25))
