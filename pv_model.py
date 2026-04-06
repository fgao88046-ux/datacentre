# pv_model.py
import pvlib
import pandas as pd
import numpy as np

def pv_forecast_from_location(lat, lon, pv_area=400, pv_eff=0.175, 
                              tz='Asia/Shanghai', surface_tilt=20, 
                              surface_azimuth=180, date='2023-07-15'):
    loc = pvlib.location.Location(lat, lon, tz=tz)
    # ⬇️ 修复：'H' → 'h'
    times = pd.date_range(date, periods=24, freq='h', tz=tz)
    cs = loc.get_clearsky(times)
    solpos = loc.get_solarposition(times)
    poa_irrad = pvlib.irradiance.get_total_irradiance(
        surface_tilt, surface_azimuth,
        solpos['apparent_zenith'], solpos['azimuth'],
        cs['dni'], cs['ghi'], cs['dhi']
    )
    dc_power = poa_irrad['poa_global'] * pv_area * pv_eff / 1000
    return np.maximum(dc_power.values, 0)

def pv_forecast_default(config):
    solar_rad = np.concatenate([np.zeros(6), np.linspace(0,800,6), np.linspace(800,0,6), np.zeros(6)])
    return solar_rad * config['PV_AREA'] * config['PV_EFF'] / 1000