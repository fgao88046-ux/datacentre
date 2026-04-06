# config.py
class Config:
    # DEAP 参数
    POP_SIZE = 80
    N_GEN = 60
    MUT_PB = 0.2
    CX_PB = 0.7

    # 系统约束
    BATTERY_MIN_SOC = 0.05
    BATTERY_MAX_SOC = 0.95
    H2_TANK_MIN = 1.0      # kg
    H2_TANK_MAX = 200.0    # kg

    # 能值系数 (seJ/J)
    EMERGY_SOLAR = 1.0
    EMERGY_WIND = 1.0
    EMERGY_GRID = 3.0      # 中国电网平均
    EMERGY_NG = 5.0        # 天然气
    EMERGY_H2 = 8.0        # 电解氢

    # 㶲系数
    EXERGY_GRID = 0.75
    EXERGY_NG = 0.86
    EXERGY_H2 = 1.20