# device_config.py
DEVICE_CONFIG = {
    "battery": {
        "capacity_kwh": 500.0,
        "efficiency": 0.95,
        "max_power_kw": 100.0
    },
    "hydrogen": {
        "tank_capacity_kg": 100.0,
        "electrolysis_max_kw": 100.0,
        "fuelcell_max_kw": 100.0,
        "h2_lhv_kwh_per_kg": 33.3
    },
    "gas_turbine": {
        "max_power_kw": 300.0,
        "ng_consumption_m3_per_kwh": 0.3,
        "ng_price_yuan_per_m3": 3.5
    },
    "pv": {"capacity_kw": 100.0},
    "wind": {"capacity_kw": 150.0}
}