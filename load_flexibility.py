# load_flexibility.py
import numpy as np

def generate_computing_load(
    base_servers: int,
    utilization_profile: list,
    pue: float = 1.5
) -> tuple:
    """
    生成数据中心负荷（含冷却）
    :param base_servers: 服务器数量
    :param utilization_profile: 利用率 [0,1]
    :param pue: 电能使用效率
    :return: (总功率, IT功率, 冷却功率)
    """
    P_idle = 0.5   # kW/server
    P_peak = 1.0   # kW/server
    
    it_power = [
        base_servers * (P_idle + (P_peak - P_idle) * u)
        for u in utilization_profile
    ]
    total_power = [p * pue for p in it_power]
    cooling_power = [p * (pue - 1) for p in it_power]
    
    return total_power, it_power, cooling_power

def get_flexible_windows(load_profile, threshold=0.8):
    """识别可迁移算力时段（低谷期）"""
    avg = np.mean(load_profile)
    flexible = [i for i, l in enumerate(load_profile) if l < avg * threshold]
    return flexible