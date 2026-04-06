# exergy_model.py
from config import Config

def calculate_exergy_loss(
    grid_import, gt_power, h2fc_power,
    pv_power, wind_power, load
):
    """计算系统㶲损失率"""
    lambda_grid = Config.EXERGY_GRID
    lambda_ng = Config.EXERGY_NG
    lambda_h2 = Config.EXERGY_H2
    
    # 输入㶲
    ex_in = (
        sum(g * lambda_grid for g in grid_import) +
        sum(gt / 0.35 * lambda_ng for gt in gt_power) +
        sum(h2 / 0.4 * lambda_h2 for h2 in h2fc_power)
    )
    
    # 输出㶲（电能=1）
    ex_out = sum(load)
    
    if ex_in == 0:
        return 0.0
    return (ex_in - ex_out) / ex_in