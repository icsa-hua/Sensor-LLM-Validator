"""
gen_calculated_data.py
Calculates sensor outputs from environmental input DataFrame.
"""

import math
import random
import numpy as np
import pandas as pd
from gen_input_data import generate_input_data 

# -------------------------
# Constants
# -------------------------
k_B = 8.617e-5  # eV/K
T_ref = 298.15  # K

Ea_sensitivity = 0.8
Ea_ionic = 0.6
Ea_humidity = 0.5

k_sensitivity_base = 0.05 / 120.0
k_ionic_base = 0.00021
k_cycling = 0.00005
k_humidity_base = 0.002

SETTLING_BASE = 0.005
TAU_SETTLING = 3.0
TAU_HUMIDITY = 6.0

FLICKER_BASE = 0.00005
WHITE_BASE = 0.00002
BURST_BASE = 0.0001


# -------------------------
# Helper functions
# -------------------------
def arrhenius(Ea, T_oper_K):
    return math.exp((Ea / k_B) * (1.0 / T_ref - 1.0 / T_oper_K))


def compute_sensitivity_factor(aging_months, T_oper_K):
    if aging_months <= 0:
        return 1.0
    AF_temp = arrhenius(Ea_sensitivity, T_oper_K)
    time_factor = (aging_months / 12.0) ** (1.0 / 3.0)
    degradation = k_sensitivity_base * time_factor * AF_temp * 12.0
    unit_variation = random.uniform(0.95, 1.05)
    sensitivity = 1.0 - degradation * unit_variation
    sensitivity = sensitivity * random.uniform(0.995, 1.005)
    return max(sensitivity, 0.8)


def settling_drift(aging_months):
    if aging_months <= 0:
        return 0.0
    s = SETTLING_BASE * (1.0 - math.exp(-aging_months / TAU_SETTLING))
    s *= (1.0 + 0.1 * math.log(1.0 + aging_months / 6.0))
    return s


def ionic_drift(aging_months, T_oper_K):
    if aging_months <= 0:
        return 0.0
    AF = arrhenius(Ea_ionic, T_oper_K)
    return k_ionic_base * math.sqrt(aging_months) * AF


def thermal_drift(thermal_cycles, dT_cycle=50.0):
    if thermal_cycles <= 0:
        return 0.0
    cycle_factor = (thermal_cycles / 1000.0) ** (1.0 / 3.0)
    temp_swing_factor = (dT_cycle / 50.0) ** 2
    return k_cycling * cycle_factor * temp_swing_factor * 1000.0


def humidity_drift(aging_months, RH_percent, T_oper_K):
    if aging_months <= 0:
        return 0.0
    rh_factor = (RH_percent / 85.0) ** 2
    AF = arrhenius(Ea_humidity, T_oper_K)
    base = k_humidity_base * rh_factor * AF
    return base * (1.0 - math.exp(-aging_months / TAU_HUMIDITY))


def compute_offset_total(aging_months, T_oper_K, RH, thermal_cycles, dT_cycle):
    s = settling_drift(aging_months)
    i = ionic_drift(aging_months, T_oper_K)
    m = thermal_drift(thermal_cycles, dT_cycle)
    h = humidity_drift(aging_months, RH, T_oper_K)
    total = s + i + m + h
    total *= random.uniform(0.9, 1.1)
    total += random.uniform(-0.5e-6, 0.5e-6)
    return total, s, i, m, h


def compute_noise_components(aging_months, T_oper_K):
    if aging_months <= 0:
        flicker = 0.0
    else:
        flicker = FLICKER_BASE * (T_oper_K / T_ref) * ((aging_months / 12.0) ** 0.3)
    white = WHITE_BASE * math.sqrt(T_oper_K / T_ref) * (1.0 + 0.2 * ((aging_months / 120.0) ** 0.25))
    burst_prob = 0.0 if aging_months <= 0 else (1.0 - math.exp(-0.01 * aging_months))
    burst = BURST_BASE * burst_prob if random.random() < burst_prob else 0.0

    unit_var = random.uniform(0.8, 1.2)
    flicker *= unit_var
    white *= unit_var
    burst *= unit_var

    total_rms = math.sqrt(flicker**2 + white**2 + burst**2)
    return total_rms, flicker, white, burst, burst_prob


# -------------------------
# Main calculation function
# -------------------------
def gen_calculated_data(df_inputs):
    np.random.seed(0)
    random.seed(0)

    dT_map = {
        "mild": (20.0, 35.0),
        "moderate": (30.0, 50.0),
        "harsh": (40.0, 70.0)
    }

    rows = []
    for idx, row in df_inputs.iterrows():
        scenario = str(row["scenario"]).lower()
        temp_C = float(row["temperature"])
        RH = float(row["humidity"])
        month = int(row["month"])
        aging_month = int(row["aging_month"])
        year = int(row["year"])
        date = row["date"]
        thermal_cycles = int(row["thermal_cycles"])
        spice_simulated = bool(row.get("spice_simulated", False))
        gain_spice = float(row.get("gain_spice", 10.0))

        T_K = temp_C + 273.15
        ideal_voltage = 0.010 * temp_C
        sensitivity = compute_sensitivity_factor(aging_month, T_K)

        dT_min, dT_max = dT_map.get(scenario, (30.0, 50.0))
        dT_cycle = (dT_min + dT_max) / 2.0 + random.uniform(-2.0, 2.0)

        offset_total, s, i, m, h = compute_offset_total(aging_month, T_K, RH, thermal_cycles, dT_cycle)
        noise_rms, noise_flicker, noise_white, noise_burst, burst_prob = compute_noise_components(aging_month, T_K)

        output_voltage = ideal_voltage * sensitivity + offset_total + noise_rms
        voltage_error = output_voltage - ideal_voltage

        rows.append({
            "scenario": scenario,
            "temperature": round(temp_C, 2),
            "humidity": round(RH, 1),
            "month": month,
            "aging_month": aging_month,
            "year": year,
            "date": date,
            "output_voltage": round(output_voltage, 6),
            "ideal_voltage": round(ideal_voltage, 6),
            "voltage_error": round(voltage_error, 6),
            "thermal_cycles": thermal_cycles,
            "sensitivity_factor": round(sensitivity, 6),
            "offset_voltage": round(offset_total, 6),
            "noise_rms": round(noise_rms, 9),
            "spice_simulated": spice_simulated,
            "gain_spice": gain_spice,
            "noise_flicker": round(noise_flicker, 9),
            "noise_white": round(noise_white, 9),
            "burst_probability": round(burst_prob, 6)
        })

    return pd.DataFrame(rows, columns=[
        "scenario","temperature","humidity","month","aging_month","year","date",
        "output_voltage","ideal_voltage","voltage_error","thermal_cycles","sensitivity_factor",
        "offset_voltage","noise_rms","spice_simulated","gain_spice","noise_flicker","noise_white","burst_probability"
    ])


if __name__ == "__main__":

    df_inputs = generate_input_data(n_samples=200, scenario="moderate", start_date="2020-01-01")

    df_outputs = gen_calculated_data(df_inputs)

    df_outputs.to_csv("sensor_created.csv", index=False)
    print(f"Wrote {len(df_outputs)} rows to sensor_aging_output.csv")
