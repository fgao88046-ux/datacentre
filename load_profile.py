# load_profile.py
from load_flexibility import generate_computing_load
import numpy as np

def generate_load_profile(hours: int = 24) -> list:
    """生成典型数据中心负荷"""
    # 模拟利用率（白天高，夜间低）
    utilization = [0.7 + 0.3 * np.sin(2*np.pi*i/24 - np.pi/2) for i in range(hours)]
    total_power, _, _ = generate_computing_load(
        base_servers=200,
        utilization_profile=utilization,
        pue=1.5
    )
    return total_power