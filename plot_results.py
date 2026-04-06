# plot_results.py
import matplotlib.pyplot as plt
import numpy as np

def plot_scheduling(x_opt, P_pv, P_wind, P_el, Q_cool, Q_heat, title, config=None):
    x_opt = np.array(x_opt)
    X = x_opt.reshape(9, 24)
    hours = np.arange(24)
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
    # 电（含风电）
    axs[0].plot(hours, P_el, 'k-', linewidth=2, label='电负荷')
    axs[0].plot(hours, P_pv + P_wind + X[0] - X[1] + X[6] - X[5], 'b--', label='供电')
    axs[0].fill_between(hours, 0, P_pv, color='gold', alpha=0.4, label='光伏')
    axs[0].fill_between(hours, P_pv, P_pv + P_wind, color='skyblue', alpha=0.4, label='风电')  # 新增
    axs[0].set_ylabel('电功率 (kW)')
    axs[0].legend()
    axs[0].grid(True)
    
    # 冷
    axs[1].plot(hours, Q_cool, 'k-', linewidth=2, label='冷负荷')
    axs[1].plot(hours, X[3]*3.5 + X[4] + X[8] - X[7], 'g--', label='制冷')
    axs[1].set_ylabel('冷功率 (kW)')
    axs[1].legend()
    axs[1].grid(True)
    
    # 热
    axs[2].plot(hours, Q_heat, 'k-', linewidth=2, label='热负荷')
    axs[2].plot(hours, X[2], 'r--', label='锅炉')
    axs[2].set_ylabel('热功率 (kW)')
    axs[2].legend()
    axs[2].grid(True)
    
    # 储能
    soc = np.zeros(25); soc[0] = 0.5
    tes = np.zeros(25); tes[0] = 0.5
    be_cap = config['BESS_CAPACITY'] if config else 500
    tes_cap = config['TES_CAPACITY'] if config else 2000
    for t in range(24):
        soc[t+1] = soc[t] + (X[5,t]*0.9 - X[6,t]/0.9) / be_cap
        tes[t+1] = tes[t] + (X[7,t]*0.95 - X[8,t]/0.95) / tes_cap
    axs[3].plot(hours, soc[1:], 'b-', label='电池 SOC')
    axs[3].plot(hours, tes[1:], 'g-', label='蓄冷罐温度')
    axs[3].set_ylabel('状态')
    axs[3].set_xlabel('小时')
    axs[3].legend()
    axs[3].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    #plt.show()