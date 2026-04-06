# device_dynamic_efficiency.py
import numpy as np

def gt_efficiency(load_ratio):
    """燃气轮机效率-负载率拟合（三次函数）"""
    if load_ratio < 0.1:
        return 0.0
    x = min(1.0, max(0.1, load_ratio))
    return 0.4102*x**3 - 0.9356*x**2 + 0.8274*x + 0.0192

def h2fc_efficiency(load_ratio):
    """氢燃料电池效率-负载率（五次函数）"""
    if load_ratio < 0.1:
        return 0.0
    x = min(1.0, max(0.1, load_ratio))
    eff = (26.067*x**5 + 18.01*x**4 + 60.894*x**3 +
           88.849*x**2 + 21.788*x + 55.916) / 100
    return min(0.6, eff)  # 上限60%