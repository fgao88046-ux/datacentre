# objectives.py
import numpy as np
from exergy_model import calculate_exergy_loss
from emergy_model import calculate_emergy_indicators

def multi_objective_function(individual, context):
    """
    多目标：经济成本 + 㶲损失 + 能值可持续性
    individual: [grid, battery, gt, h2fc] * horizon
    """
    horizon = len(context["load"])
    grid = individual[0::4]
    battery = individual[1::4]
    gt = individual[2::4]
    h2fc = individual[3::4]
    
    # 经济成本
    cost = sum(g * p for g, p in zip(grid, context["price"]))
    
    # 㶲损失
    ex_loss = calculate_exergy_loss(grid, gt, h2fc, 
                                   context["pv"], context["wind"], context["load"])
    
    # 能值指标（惩罚低 ESI）
    emergy = calculate_emergy_indicators(
        pv_energy=sum(context["pv"]),
        wind_energy=sum(context["wind"]),
        grid_energy=sum(grid),
        ng_energy=sum(gt) * 0.3,  # m³
        h2_energy=sum(h2fc) * 0.4 / 33.3  # kg
    )
    esi_penalty = 1.0 / (emergy["ESI"] + 1e-5)  # 鼓励高 ESI
    
    return cost, ex_loss, esi_penalty