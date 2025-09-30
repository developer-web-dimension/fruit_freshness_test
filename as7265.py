import csv
from datetime import datetime
import qwiic_as7265x
import os
import time


INTEG_CYCLES      = 60
GAIN_SETTING      = 3
SAT_CUTOFF   = 3000.0     # saturation cutoff you referenced
LED_CURRENT  = 2          # 12.5/25/50/100 mA enum depending on driver
sdiv = lambda a, b: (a / b) if b else 0.0

def take_measurements(sensor):
    """Trigger a measurement (API name differs across versions)."""
    return call_any(sensor, ["take_measurements", "takeMeasurements"])

def get_calibrated_channel(sensor, channel):
    """
    Read one calibrated channel (E,F,G,...). Different libs expose
    either snake_case or CamelCase getters.
    """
    ch = channel.strip()
    return call_any(sensor, [f"get_calibrated_{ch.lower()}",
                             f"getCalibrated{ch}"])

def set_gain(sensor, gain_enum):
    """Set sensor gain with API compatibility."""
    call_any(sensor, ["set_gain", "setGain"], gain_enum)

def call_any(obj, names, *args, **kwargs):
    """
    Call the first available method from a list of method names.
    Useful for handling different API versions of the sensor library.
    """
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            return method(*args, **kwargs)
    raise AttributeError(f"None of {names} exist on {obj}")

def set_integration_cycles(sensor, cycles):
    """Set integration cycles with API compatibility."""
    call_any(sensor, ["set_integration_cycles", "setIntegrationCycles"], cycles)

def continuous_collect_as7265(
    sample_hz: float = 2.0,
    avg_frames: int = 3,          
    out_csv: str | None = "/home/Subral/python/fresh_fruit/as7265_stream.csv",
    enable_white_led: bool = True,
):
    """
    Continuously collect calibrated AS7265x data at ~sample_hz until Ctrl+C.
    - Averages 'avg_frames' frames per sample (skips saturated frames).
    - Writes CSV if 'out_csv' is provided (creates file + header if missing).
    - Prints one line per sample to the console.
    """

    # --- init sensor ---
    sensor = qwiic_as7265x.QwiicAS7265x()
    if not sensor.begin():
        print("AS7265x Sensor not found! Check wiring/I2C.")
        return

    # Configure integration/gain
    set_integration_cycles(sensor, INTEG_CYCLES)
    set_gain(sensor, GAIN_SETTING)

    # Optionally enable on-board white lamp
    if enable_white_led:
        try:
            sensor.enable_bulb(sensor.kLedWhite)
            try:
                sensor.set_bulb_current(sensor.kLedCurrentLimit25mA, sensor.kLedWhite)
            except Exception:
                pass
        except Exception:
            enable_bulb(sensor, "white")
            set_bulb_current(sensor, LED_CURRENT, "white")

    # Your driver channel order (keep consistent with your code)
    channels = list("EFGHIJSTUVWL")

    # Prepare CSV
    csv_file = None
    csv_writer = None
    if out_csv:
        # Create and write header if new file
        new_file = not os.path.exists(out_csv)
        csv_file  = open(out_csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if new_file:
            csv_writer.writerow(
                ["timestamp"] + channels +
                ["total_signal", "GR", "ARI", "NDVI", "NDWI", "T_over_S", "U_over_S"]
            )
            csv_file.flush()

    period = 1.0 / max(0.1, float(sample_hz))  # guard against div-by-zero or silly values
    print(f"Streaming AS7265x @ ~{sample_hz} Hz (avg_frames={avg_frames}). Ctrl+C to stop.")
    try:
        while True:
            t_start = time.time()

            # --- average a few frames ---
            sum_values = {ch: 0.0 for ch in channels}
            valid_frames = 0
            for _ in range(max(1, int(avg_frames))):
                try:
                    take_measurements(sensor)
                    readings = {ch: get_calibrated_channel(sensor, ch) for ch in channels}
                    # skip saturated frames
                    if all(v < SAT_CUTOFF for v in readings.values()):
                        for ch, val in readings.items():
                            sum_values[ch] += val
                        valid_frames += 1
                except Exception as e:
                    # soft-fail this frame
                    # print(f"Frame read error: {e}")
                    pass
                time.sleep(0.01)

            if valid_frames == 0:
                print("No valid frames this sample (saturation or read error).")
                time.sleep(max(0, period - (time.time() - t_start)))
                continue

            avg_values = {ch: sum_values[ch] / valid_frames for ch in channels}
            E, F, G, H = avg_values['E'], avg_values['F'], avg_values['G'], avg_values['H']
            I, J, S, T = avg_values['I'], avg_values['J'], avg_values['S'], avg_values['T']
            U, V, W, L = avg_values['U'], avg_values['V'], avg_values['W'], avg_values['L']

            total_signal = sum(avg_values.values())

            # Indices (same math as your analyzer)
            green_mean = (E + F + G + H) / 4.0
            GR        = sdiv(green_mean, S)
            ARI       = (sdiv(1.0, G) - sdiv(1.0, J))
            NIR       = (V + W) / 2.0
            NDVI      = sdiv(NIR - S, NIR + S)
            NDWI      = sdiv(W - L, W + L)
            T_over_S  = sdiv(T, S)
            U_over_S  = sdiv(U, S)

            timestamp = datetime.now().isoformat(timespec="seconds")

            # Console line
            print(
                f"+++++++++++++++++++++++++++++++++++++++++\n"
                f"[{timestamp}] total={total_signal:.0f} | \n "
                # f"E={E:.1f} F={F:.1f} G={G:.1f} H={H:.1f} I={I:.1f} J={J:.1f} "
                # f"S={S:.1f} T={T:.1f} U={U:.1f} V={V:.1f} W={W:.1f} L={L:.1f} | "
                f"GR={GR:.3f} \n ARI={ARI:.6f} \n NDVI={NDVI:.3f} \n NDWI={NDWI:.3f} \n " 
                f"T/S={T_over_S:.3f} \n U/S={U_over_S:.3f}"
            )

            # CSV line
            if csv_writer:
                csv_writer.writerow(
                    [timestamp] + [avg_values[ch] for ch in channels] +
                    [total_signal, GR, ARI, NDVI, NDWI, T_over_S, U_over_S]
                )
                csv_file.flush()

            # pacing
            elapsed = time.time() - t_start
            sleep_left = period - elapsed
            if sleep_left > 0:
                time.sleep(sleep_left)

    except KeyboardInterrupt:
        print("\nStopping stream…")

    finally:
        # turn off lamp if we turned it on
        if enable_white_led:
            try:
                try:
                    sensor.disable_bulb(sensor.kLedWhite)
                except Exception:
                    disable_bulb(sensor, "white")
            except Exception:
                pass
        try:
            if csv_file:
                csv_file.close()
        except Exception:
            pass
        print("AS7265x stream ended.")

# Example: stream at 2 Hz, average 3 frames per sample, write CSV
continuous_collect_as7265(sample_hz=2.0, avg_frames=3,out_csv="/home/Subral/python/fresh_fruit/as7265_stream.csv",enable_white_led=True)
