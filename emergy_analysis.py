# emergy_analysis.py
import numpy as np

def calculate_ESI(x, P_pv, P_wind, P_el, Q_cool, Q_heat, config):
    T = config['T']
    X = np.array(x).reshape(9, T)
    
    # 输入能值 (seJ)
    F = np.sum(X[0]) * config['EMR_GRID'] * 3600  # 电网
    N = np.sum(X[2]) * config['EMR_GAS'] * 3600   # 天然气
    R_solar = np.sum(P_pv) * config['EMR_SOLAR'] * 3600
    R_wind = np.sum(P_wind) * config['EMR_WIND'] * 3600 if 'EMR_WIND' in config else 0
    
    R = R_solar + R_wind
    total_input = F + N + R
    
    Y = np.sum(P_el - 20) * config['EMR_GRID'] * 3600
    
    if total_input == 0 or (F + N) == 0:
        return 0.0, 0.0, float('inf')
    
    EYR = Y / (F + N)
    ELR = (F + N) / R if R > 0 else float('inf')
    ESI = EYR / ELR if ELR > 0 else 0.0
    
    return ESI, EYR, ELR