#!/usr/bin/env python3
import time
import math

# SparkFun Qwiic AS7265x driver
# pip install sparkfun-qwiic-as7265x
import qwiic_as7265x

# -------- Config (mirrors your Arduino constants) --------
NUM_FRAMES        = 3
SAT_CUTOFF        = 3000.0   # simple saturation gate
MIN_TOTAL_SIGNAL  = 60.0
INTEG_CYCLES      = 60       # exposure (cycles)
GAIN_SETTING      = 3        # AS7265x gain enum; 0=1x,1=3.7x,2=16x,3=64x (driver-dependent)
LED_CURRENT       = 2        # driver-dependent enum; typical 2=25mA

# ----- Apple-specific thresholds -----
GR_FRESH_MIN   = 0.37
GR_STALE_MAX   = 0.33

ARI_FRESH_MAX  = 0.002
ARI_STALE_MIN  = 0.008

NDWI_FRESH_MAX = 0.49
NDWI_STALE_MIN = 0.55

TOS_FRESH_MAX  = 0.455
TOS_STALE_MIN  = 0.470

UOS_FRESH_MAX  = 0.255
UOS_STALE_MIN  = 0.275

# ---- Weights (sum = 100) ----
W_GR   = 5
W_ARI  = 40
W_NDWI = 10
W_TOS  = 5
W_UOS  = 5
# NDVI weight 0 in your setup

# -------- Helpers --------
def safe_div(a, b):
    return (a / b) if b != 0.0 else 0.0

def score_index(val, fresh_min_or_max, stale_max_or_min, higher_is_better):
    """
    Returns 0..20 points
    """
    if higher_is_better:
        fresh_min = fresh_min_or_max
        stale_max = stale_max_or_min
        if val >= fresh_min:
            return 20
        if val <= stale_max:
            return 0
        return int(20.0 * (val - stale_max) / (fresh_min - stale_max))
    else:
        fresh_max = fresh_min_or_max
        stale_min = stale_max_or_min
        if val <= fresh_max:
            return 20
        if val >= stale_min:
            return 0
        return int(20.0 * (stale_min - val) / (stale_min - fresh_max))

def scale_pts(pts0to20, weight):
    return int((pts0to20 * weight) / 20)

# Small shim to tolerate different method names across driver versions
def call_any(obj, names, *args):
    for n in names:
        fn = getattr(obj, n, None)
        if callable(fn):
            return fn(*args)
    raise AttributeError(f"None of {names} exist on {obj}")

def get_calibrated_channel(sensor, ch):
    """
    ch in ['E','F','G','H','I','J','S','T','U','V','W','L']
    """
    # try e.g. get_calibrated_e() and getCalibratedE()
    lname = ch.lower()
    return call_any(sensor, [f"get_calibrated_{lname}", f"getCalibrated{ch}"])

def set_integration_cycles(sensor, cycles):
    call_any(sensor, ["set_integration_cycles", "setIntegrationCycles"], cycles)

def set_gain(sensor, gain_enum):
    call_any(sensor, ["set_gain", "setGain"], gain_enum)

def enable_bulb(sensor, which="white"):
    # Many drivers just enable the white bulb; some accept enums
    try:
        # new style: enable_bulb("white")
        call_any(sensor, ["enable_bulb", "enableBulb"], which)
    except Exception:
        # fallback: simple LED on
        call_any(sensor, ["enable_indicator", "enableIndicator"])

def set_bulb_current(sensor, current_enum, which="white"):
    try:
        call_any(sensor, ["set_bulb_current", "setBulbCurrent"], which, current_enum)
    except Exception:
        pass  # not critical; keep defaults

def take_measurements(sensor):
    call_any(sensor, ["take_measurements", "takeMeasurements"])

# -------- Main --------
def main():
    sensor = qwiic_as7265x.QwiicAS7265x()

    if not sensor.begin():
        print("Sensor not found!")
        return

    # Configure
    set_integration_cycles(sensor, INTEG_CYCLES)
    set_gain(sensor, GAIN_SETTING)
    sensor.enable_bulb(sensor.kLedWhite)
    sensor.set_bulb_current(sensor.kLedCurrentLimit25mA, sensor.kLedWhite)

    print("Fruit Freshness Analyzer – Index-based (Apple-tuned)")
    print("Keep 3–5 cm distance; use a dark shroud if possible.")
    time.sleep(1.2)

    while True:
        sum_vals = {k: 0.0 for k in list("EFGHIJSTU" "VWL")}
        valid = 0

        for _ in range(NUM_FRAMES):
            take_measurements(sensor)

            try:
                E = get_calibrated_channel(sensor, "E")  # 510
                F = get_calibrated_channel(sensor, "F")  # 535
                G = get_calibrated_channel(sensor, "G")  # 560
                H = get_calibrated_channel(sensor, "H")  # 585
                I = get_calibrated_channel(sensor, "I")  # 645
                J = get_calibrated_channel(sensor, "J")  # 705
                S = get_calibrated_channel(sensor, "S")  # 680
                T = get_calibrated_channel(sensor, "T")  # 730
                U = get_calibrated_channel(sensor, "U")  # 760
                V = get_calibrated_channel(sensor, "V")  # 810
                W = get_calibrated_channel(sensor, "W")  # 860
                L = get_calibrated_channel(sensor, "L")  # 940
            except Exception as e:
                print(f"Read error: {e}")
                time.sleep(1.5)
                continue

            # Simple saturation gate
            if all(x < SAT_CUTOFF for x in [E,F,G,H,I,J,S,T,U,V,W,L]):
                for k, val in zip(list("EFGHIJSTUVWL"), [E,F,G,H,I,J,S,T,U,V,W,L]):
                    sum_vals[k] += val
                valid += 1

            time.sleep(0.12)

        if valid < 2:
            print("MEASUREMENT ERROR – Adjust position/lighting")
            time.sleep(1.5)
            continue

        # Averages
        E = sum_vals["E"]/valid; F = sum_vals["F"]/valid; G = sum_vals["G"]/valid; H = sum_vals["H"]/valid
        I = sum_vals["I"]/valid; J = sum_vals["J"]/valid; S = sum_vals["S"]/valid
        T = sum_vals["T"]/valid; U = sum_vals["U"]/valid; V = sum_vals["V"]/valid
        W = sum_vals["W"]/valid; L = sum_vals["L"]/valid

        total_sig = E+F+G+H+I+J+S+T+U+V+W+L
        if total_sig < MIN_TOTAL_SIGNAL:
            print("NO FRUIT / Very low signal – move closer or increase illumination")
            time.sleep(1.2)
            continue

        # ---- Indices ----
        green_mean = (E + F + G + H) / 4.0
        GR   = safe_div(green_mean, S)

        ARI  = (safe_div(1.0, G) - safe_div(1.0, J))  # 1/560 - 1/705

        NIR  = (V + W) / 2.0
        NDVI = ((NIR - S) / (NIR + S)) if (NIR + S) > 0 else 0.0

        NDWI = ((W - L) / (W + L)) if (W + L) > 0 else 0.0

        T_over_S = safe_div(T, S)
        U_over_S = safe_div(U, S)

        # ---- Scoring ----
        pts_GR   = scale_pts(score_index(GR,   GR_FRESH_MIN,   GR_STALE_MAX,   True),  W_GR)
        pts_ARI  = scale_pts(score_index(ARI,  ARI_FRESH_MAX,  ARI_STALE_MIN,  False), W_ARI)
        pts_NDWI = scale_pts(score_index(NDWI, NDWI_FRESH_MAX, NDWI_STALE_MIN, False), W_NDWI)
        pts_TOS  = scale_pts(score_index(T_over_S, TOS_FRESH_MAX, TOS_STALE_MIN, False), W_TOS)
        pts_UOS  = scale_pts(score_index(U_over_S, UOS_FRESH_MAX, UOS_STALE_MIN, False), W_UOS)

        score = pts_GR + pts_ARI + pts_NDWI + pts_TOS + pts_UOS
        score = max(0, min(100, score))

        # Category & probability
        if score >= 60:
            category = "FRESH"
            probability = 70 + int((score - 60) * 1.0)
        elif score >= 45:
            category = "AVERAGE"
            probability = 45 + int((score - 45) * 1.67)
        else:
            category = "STALE"
            probability = int(score * 1.0)

        probability = max(0, min(100, probability))

        # Confidence by signal strength
        if   total_sig >= 2000: confidence = 95
        elif total_sig >= 1000: confidence = 85
        elif total_sig >=  400: confidence = 70
        elif total_sig >=  200: confidence = 55
        else:                   confidence = 40

        # ---- Print results ----
        print(f"FRESHNESS_SCORE:{score}")
        print(f"PROBABILITY:{probability}%")
        print(f"CATEGORY:{category}")
        print(f"CONFIDENCE:{confidence}%")

        print("--- Index Details ---")
        print(f"G/R: {GR:.3f}")
        print(f"ARI': {ARI:.6f}")
        print(f"NDVI: {NDVI:.3f}")
        print(f"NDWI': {NDWI:.3f}")
        print(f"T/S: {T_over_S:.3f}")
        print(f"U/S: {U_over_S:.3f}")

        print(f"Total signal: {total_sig:.0f}")
        print("========================")

        time.sleep(2.5)

if __name__ == "__main__":
    main()
