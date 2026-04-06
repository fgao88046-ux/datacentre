# optimizer.py
import random
from deap import base, creator, tools, algorithms
import numpy as np
from objectives import multi_objective_function

def optimize_energy_schedule(
    load_profile, pv_power, wind_power, price_profile,
    battery_capacity=500.0,
    include_gas_turbine=True,
    include_hydrogen=True
):
    horizon = len(load_profile)
    context = {
        "load": load_profile,
        "pv": pv_power,
        "wind": wind_power,
        "price": price_profile
    }
    
    # 创建多目标个体
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    
    def init_individual():
        ind = []
        for _ in range(horizon):
            grid = random.uniform(0, 300)
            battery = random.uniform(-100, 100)
            gt = random.uniform(0, 300) if include_gas_turbine else 0
            h2fc = random.uniform(0, 100) if include_hydrogen else 0
            ind.extend([grid, battery, gt, h2fc])
        return creator.Individual(ind)
    
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", multi_objective_function, context=context)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=300, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=300, eta=20.0, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    
    pop = toolbox.population(n=80)
    hof = tools.ParetoFront()
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    
    algorithms.eaMuPlusLambda(pop, toolbox, mu=80, lambda_=120, cxpb=0.7, mutpb=0.2,
                              ngen=60, stats=stats, halloffame=hof, verbose=False)
    
    # 取 Pareto 前沿第一个解（最经济）
    best = hof[0] if hof else [0]*(4*horizon)
    
    # 解析结果
    result = {
        "time": [i for i in range(horizon)],
        "pv": pv_power,
        "wind": wind_power,
        "load": load_profile,
        "grid": best[0::4],
        "battery": best[1::4],
        "gas_turbine": best[2::4],
        "h2_fuelcell": best[3::4],
        "price": price_profile
    }
    return result