# energy_system.py
class EnergySystem:
    def __init__(self, config):
        self.soc = 0.5
        self.h2_level = 10.0
        self.battery_cap = config["battery"]["capacity_kwh"]
        self.h2_tank_cap = config["hydrogen"]["tank_capacity_kg"]
    
    def update(self, action):
        # 电池
        charge = -action["battery"] if action["battery"] < 0 else 0
        discharge = action["battery"] if action["battery"] > 0 else 0
        self.soc += (charge * 0.95 - discharge * 0.95) / self.battery_cap
        self.soc = max(0.05, min(0.95, self.soc))
        
        # 氢能
        h2_prod = action["h2_electrolysis"] * 0.6 / 33.3 if "h2_electrolysis" in action else 0
        h2_cons = action["h2_fuelcell"] * 0.4 / 33.3 if "h2_fuelcell" in action else 0
        self.h2_level += h2_prod - h2_cons
        self.h2_level = max(1.0, min(self.h2_tank_cap, self.h2_level))