import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SCENARIOS = {
    "mild": {
        "T_avg": 20, "T_amp": 10,
        "RH_avg": 55, "RH_amp": 15,
        "cycles_base": 20, "cycles_amp": 10,
        "dT_min": 20, "dT_max": 35
    },
    "moderate": {
        "T_avg": 30, "T_amp": 15,
        "RH_avg": 60, "RH_amp": 20,
        "cycles_base": 80, "cycles_amp": 30,
        "dT_min": 30, "dT_max": 50
    },
    "harsh": {
        "T_avg": 55, "T_amp": 20,
        "RH_avg": 45, "RH_amp": 25,
        "cycles_base": 200, "cycles_amp": 80,
        "dT_min": 40, "dT_max": 70
    }
}


def seasonal_temperature(month, scenario):
    cfg = SCENARIOS[scenario]
    phi = 7  # peak in July
    noise = np.random.uniform(-2.0, 2.0)
    T = cfg["T_avg"] + cfg["T_amp"] * np.sin((2 * np.pi / 12.0) * (month - phi)) + noise
    return round(float(T), 2)


def seasonal_humidity(month, scenario, T):
    cfg = SCENARIOS[scenario]
    phi = 1
    RH = cfg["RH_avg"] + cfg["RH_amp"] * np.sin((2 * np.pi / 12.0) * (month - phi) + np.pi)
    RH -= 0.15 * (T - cfg["T_avg"])  # inverse relation
    RH += np.random.uniform(-3.0, 3.0)
    RH = float(np.clip(RH, 5.0, 95.0))
    return round(RH, 1)


def seasonal_cycles(month, scenario):
    cfg = SCENARIOS[scenario]
    cycles = cfg["cycles_base"] + cfg["cycles_amp"] * abs(np.sin((2 * np.pi / 12.0) * (month - 6)))
    cycles += np.random.uniform(-3.0, 3.0)
    return int(max(0, round(cycles)))


def seasonal_delta_T(scenario):
    cfg = SCENARIOS[scenario]
    return round(np.random.uniform(cfg["dT_min"], cfg["dT_max"]), 2)


def generate_input_data(n_samples=200, scenario="moderate", start_date="2020-01-01"):
    scenario = scenario.lower()
    if scenario not in SCENARIOS:
        raise ValueError("scenario must be one of: mild, moderate, harsh")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    rows = []

    for i in range(n_samples):
        date = start + timedelta(days=30 * i)
        month = date.month
        year = date.year
        aging_month = i

        T = seasonal_temperature(month, scenario)
        RH = seasonal_humidity(month, scenario, T)
        cycles = seasonal_cycles(month, scenario)
        dT = seasonal_delta_T(scenario) 

        rows.append({
            "scenario": scenario,
            "temperature": T,
            "humidity": RH,
            "month": month,
            "aging_month": aging_month,
            "year": year,
            "date": date.strftime("%Y-%m-%d"),
            "thermal_cycles": cycles,
            "spice_simulated": False,
            "gain_spice": 10.0,
            "_dT_cycle": dT 
        })

    df = pd.DataFrame(rows)
    return df[[
        "scenario", "temperature", "humidity", "month",
        "aging_month", "year", "date", "thermal_cycles",
        "spice_simulated", "gain_spice"
    ]]
