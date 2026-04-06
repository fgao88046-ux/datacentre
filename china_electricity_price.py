# china_electricity_price.py
from datetime import datetime

# =============== 全国分时电价配置 ===============
ELECTRICITY_PRICE = {
    # 华北
    "北京": {
        "大工业": {
            "summer": {"peak": 1.35, "shoulder": 0.85, "valley": 0.35},
            "winter": {"peak": 1.25, "shoulder": 0.80, "valley": 0.30}
        }
    },
    "河北": {
        "大工业": {
            "summer": {"peak": 1.20, "shoulder": 0.75, "valley": 0.32},
            "winter": {"peak": 1.15, "shoulder": 0.70, "valley": 0.28}
        }
    },

    # 华东
    "上海": {
        "大工业": {
            "summer": {"peak": 1.30, "shoulder": 0.80, "valley": 0.40},
            "winter": {"peak": 1.20, "shoulder": 0.75, "valley": 0.35}
        }
    },
    "江苏": {
        "大工业": {
            "summer": {"peak": 1.25, "shoulder": 0.78, "valley": 0.38},
            "winter": {"peak": 1.18, "shoulder": 0.72, "valley": 0.33}
        }
    },
    "浙江": {
        "大工业": {
            "summer": {"peak": 1.28, "shoulder": 0.80, "valley": 0.36},
            "winter": {"peak": 1.20, "shoulder": 0.75, "valley": 0.32}
        }
    },
    "山东": {
        "大工业": {
            "summer": {"peak": 1.15, "shoulder": 0.70, "valley": 0.30},
            "winter": {"peak": 1.10, "shoulder": 0.68, "valley": 0.28}
        }
    },

    # 华南
    "广东": {
        "大工业": {
            "summer": {"peak": 1.50, "shoulder": 0.90, "valley": 0.30},
            "winter": {"peak": 1.40, "shoulder": 0.85, "valley": 0.28}
        }
    },
    "广西": {
        "大工业": {
            "summer": {"peak": 1.35, "shoulder": 0.82, "valley": 0.32},
            "winter": {"peak": 1.25, "shoulder": 0.78, "valley": 0.30}
        }
    },

    # 华中
    "湖北": {
        "大工业": {
            "summer": {"peak": 1.25, "shoulder": 0.75, "valley": 0.35},
            "winter": {"peak": 1.20, "shoulder": 0.72, "valley": 0.32}
        }
    },
    "湖南": {
        "大工业": {
            "summer": {"peak": 1.22, "shoulder": 0.74, "valley": 0.34},
            "winter": {"peak": 1.18, "shoulder": 0.70, "valley": 0.30}
        }
    },
    "河南": {
        "大工业": {
            "summer": {"peak": 1.18, "shoulder": 0.72, "valley": 0.30},
            "winter": {"peak": 1.12, "shoulder": 0.68, "valley": 0.28}
        }
    },

    # 西南
    "四川": {
        "大工业": {
            "summer": {"peak": 1.10, "shoulder": 0.65, "valley": 0.28},
            "winter": {"peak": 1.05, "shoulder": 0.62, "valley": 0.25}
        }
    },
    "重庆": {
        "大工业": {
            "summer": {"peak": 1.15, "shoulder": 0.70, "valley": 0.30},
            "winter": {"peak": 1.10, "shoulder": 0.68, "valley": 0.28}
        }
    },

    # 西北
    "陕西": {
        "大工业": {
            "summer": {"peak": 1.12, "shoulder": 0.68, "valley": 0.28},
            "winter": {"peak": 1.08, "shoulder": 0.65, "valley": 0.25}
        }
    },
    "新疆": {
        "大工业": {
            "summer": {"peak": 1.05, "shoulder": 0.62, "valley": 0.25},
            "winter": {"peak": 1.00, "shoulder": 0.60, "valley": 0.22}
        }
    }
}

# =============== 分时规则（峰谷时段）===============
TIME_OF_USE_RULES = {
    "北京": {"summer_months": [6, 7, 8], "peak_hours": [(10, 15), (18, 21)], "valley_hours": [(0, 8), (23, 24)]},
    "河北": {"summer_months": [6, 7, 8], "peak_hours": [(10, 15), (18, 21)], "valley_hours": [(0, 8)]},
    
    "上海": {"summer_months": [6, 7, 8, 9], "peak_hours": [(8, 11), (18, 21)], "valley_hours": [(23, 7)]},
    "江苏": {"summer_months": [7, 8], "peak_hours": [(9, 11), (19, 21)], "valley_hours": [(23, 8)]},
    "浙江": {"summer_months": [7, 8], "peak_hours": [(9, 11), (19, 21)], "valley_hours": [(23, 7)]},
    "山东": {"summer_months": [6, 7, 8], "peak_hours": [(8, 11), (18, 21)], "valley_hours": [(23, 7)]},
    
    "广东": {"summer_months": [5, 6, 7, 8, 9], "peak_hours": [(10, 12), (14, 19)], "valley_hours": [(0, 8)]},
    "广西": {"summer_months": [5, 6, 7, 8, 9], "peak_hours": [(10, 12), (15, 20)], "valley_hours": [(0, 8)]},
    
    "湖北": {"summer_months": [6, 7, 8], "peak_hours": [(9, 12), (16, 21)], "valley_hours": [(23, 7)]},
    "湖南": {"summer_months": [6, 7, 8], "peak_hours": [(9, 12), (16, 21)], "valley_hours": [(23, 7)]},
    "河南": {"summer_months": [6, 7, 8], "peak_hours": [(8, 11), (18, 21)], "valley_hours": [(23, 7)]},
    
    "四川": {"summer_months": [6, 7, 8], "peak_hours": [(9, 12), (15, 18)], "valley_hours": [(23, 7)]},
    "重庆": {"summer_months": [6, 7, 8], "peak_hours": [(9, 12), (16, 20)], "valley_hours": [(23, 7)]},
    
    "陕西": {"summer_months": [6, 7, 8], "peak_hours": [(9, 12), (17, 21)], "valley_hours": [(23, 7)]},
    "新疆": {"summer_months": [6, 7, 8], "peak_hours": [(10, 14), (19, 23)], "valley_hours": [(0, 8)]}
}


def get_hourly_price(province: str, user_type: str = "大工业", hours: int = 24) -> list:
    """获取指定省份24小时分时电价"""
    if province == "自定义":
        # 自定义模式由前端传入完整价格列表
        return [0.8] * hours  # 默认值，实际由前端覆盖

    now = datetime.now()
    rules = TIME_OF_USE_RULES.get(province, TIME_OF_USE_RULES["北京"])
    is_summer = now.month in rules["summer_months"]
    
    price_map = ELECTRICITY_PRICE.get(province, ELECTRICITY_PRICE["北京"])
    season_key = "summer" if is_summer else "winter"
    prices = price_map[user_type].get(season_key, price_map[user_type]["summer"])
    
    def in_peak(hour):
        return any(s <= hour < e for s, e in rules["peak_hours"])
    def in_valley(hour):
        return any(s <= hour < e for s, e in rules["valley_hours"])
    
    hourly = []
    for i in range(hours):
        hour = (now.hour + i) % 24
        if in_peak(hour):
            p = prices["peak"]
        elif in_valley(hour):
            p = prices["valley"]
        else:
            p = prices.get("shoulder", (prices["peak"] + prices["valley"]) / 2)
        hourly.append(p)
    return hourly


def get_all_provinces():
    """返回所有支持的省份列表"""
    provinces = sorted(ELECTRICITY_PRICE.keys())
    provinces.insert(0, "自定义")  # 把“自定义”放在第一位
    return provinces