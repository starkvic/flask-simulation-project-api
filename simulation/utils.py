import numpy as np

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
initial_irradiance = 800  # [W/m²]
T_value = 25              # [°C]

# ------------------------------------------------------------------------------
# PV System Model and Objective Function (Shared by All Algorithms)
# ------------------------------------------------------------------------------
def pv_system_model(V, G, T):
    I_sc = 10              # [A]
    V_oc = 100             # [V]
    Temp_coeff_V = -0.005  # [-/°C]
    T_ref = 25             # [°C]
    Max_efficiency = 0.85  # [unitless]
    V_oc_adjusted = V_oc * (1 + Temp_coeff_V * (T - T_ref))
    I = I_sc * (1 - np.exp(-V / V_oc_adjusted)) * (G / 1000)
    P = V * I
    P_max = Max_efficiency * V_oc_adjusted * I_sc
    if P > P_max:
        return -np.inf
    return P

def objective_function(params, G, T):
    V = params[0]
    if V < 0 or V > 100:
        return -np.inf
    return pv_system_model(V, G, T)

def compute_power_history(V_history, G_history, T):
    return [pv_system_model(V, G, T) for V, G in zip(V_history, G_history)]
