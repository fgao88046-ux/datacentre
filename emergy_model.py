# emergy_model.py
from config import Config

def calculate_emergy_indicators(
    pv_energy, wind_energy,
    grid_energy, ng_energy, h2_energy
):
    """计算能值指标"""
    R = pv_energy * Config.EMERGY_SOLAR + wind_energy * Config.EMERGY_WIND
    N = ng_energy * Config.EMERGY_NG + h2_energy * Config.EMERGY_H2
    F = grid_energy * Config.EMERGY_GRID
    
    Y = R + N + F
    if F == 0 or R == 0:
        return {"EYR": 0, "ELR": float('inf'), "ESI": 0}
    
    EYR = Y / F
    ELR = (N + F) / R
    ESI = EYR / ELR if ELR > 0 else 0
    
    return {"EYR": EYR, "ELR": ELR, "ESI": ESI}