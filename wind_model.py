# wind_model.py
import numpy as np

def wind_power_forecast(wind_speed, turbine_type="Vestas V90 2MW", hub_height=80):
    """使用 windpowerlib 计算风电出力"""
    try:
        from windpowerlib import WindTurbine, ModelChain
        import pandas as pd
        
        turbine = WindTurbine(turbine_type=turbine_type, hub_height=hub_height)
        time_index = pd.date_range('2023-07-15', periods=24, freq='H')
        weather_df = pd.DataFrame({
            'wind_speed': wind_speed,
            'temperature': 25,
            'pressure': 101325
        }, index=time_index)
        
        mc = ModelChain(turbine).run_model(weather_df)
        power_w = mc.power_output.values
        return np.maximum(power_w / 1000, 0)
    except Exception as e:
        print(f"windpowerlib 失败，使用默认模型: {e}")
        return wind_forecast_default()

def wind_forecast_default():
    """默认风电曲线（沿海地区）"""
    return np.array([
        800, 850, 900, 920, 900, 850,
        750, 650, 550, 500, 480, 470,
        460, 480, 520, 580, 650, 730,
        800, 860, 900, 920, 930, 920
    ]) / 1000  # kW