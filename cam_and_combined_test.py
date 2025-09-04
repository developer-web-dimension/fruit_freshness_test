#!/usr/bin/env python3
import sys, time, signal
import numpy as np
import cv2
from PIL import Image, ImageOps

# ---- TensorFlow/Keras (2.13.x) ----
import tensorflow as tf
from tensorflow.keras.models import load_model

def DepthwiseConv2D_compat(**kwargs):
    kwargs.pop("groups", None)
    return tf.keras.layers.DepthwiseConv2D(**kwargs)

CUSTOM_OBJECTS = {"DepthwiseConv2D": DepthwiseConv2D_compat}

MODEL_PATH  = "/home/Subral/python/arduino/model/keras_model.h5"
LABELS_PATH = "/home/Subral/python/arduino/model/labels.txt"
INPUT_SIZE  = (224, 224)
CONF_THRESHOLD = 0.70
WARMUP_FRAMES  = 5            # ignore first N frames for camera warmup
STABLE_HITS    = 5            # need this many consecutive frames ≥ threshold

# ---- AS7265X (SparkFun Qwiic) ----
import qwiic_as7265x

NUM_FRAMES        = 3
SAT_CUTOFF        = 3000.0
MIN_TOTAL_SIGNAL  = 60.0
INTEG_CYCLES      = 60
GAIN_SETTING      = 3     # 0=1x,1=3.7x,2=16x,3=64x (driver-dependent enum)
LED_CURRENT_CONST = None  # resolved after sensor.begin()

LED_STRIP_PIN = 18        # BCM numbering; change to your wiring
LED_ACTIVE_HIGH = True     # set False if your relay is active-LOW


# Thresholds (your tuned values)
GR_FRESH_MIN   = 0.37; GR_STALE_MAX   = 0.33
ARI_FRESH_MAX  = 0.002; ARI_STALE_MIN = 0.008
NDWI_FRESH_MAX = 0.49;  NDWI_STALE_MIN= 0.55
TOS_FRESH_MAX  = 0.455; TOS_STALE_MIN = 0.470
UOS_FRESH_MAX  = 0.255; UOS_STALE_MIN = 0.275

W_GR=5; W_ARI=40; W_NDWI=20; W_TOS=5; W_UOS=5

np.set_printoptions(suppress=True)

class LedStrip:
    def __init__(self, pin=LED_STRIP_PIN, active_high=LED_ACTIVE_HIGH):
        self.pin = pin
        self.active_high = active_high
        self._lib = None
        try:
            import RPi.GPIO as GPIO
            self._lib = 'RPi.GPIO'
            self.GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT,
                       initial=GPIO.LOW if active_high else GPIO.HIGH)
        except Exception:
            try:
                from gpiozero import LED
                self._lib = 'gpiozero'
                self._led = LED(pin, active_high=active_high)
            except Exception:
                self._lib = None
                print("WARNING: No GPIO lib found; LED control disabled.")

    def on(self):
        if self._lib == 'RPi.GPIO':
            self.GPIO.output(self.pin,
                             self.GPIO.HIGH if self.active_high else self.GPIO.LOW)
        elif self._lib == 'gpiozero':
            self._led.on()

    def off(self):
        if self._lib == 'RPi.GPIO':
            self.GPIO.output(self.pin,
                             self.GPIO.LOW if self.active_high else self.GPIO.HIGH)
        elif self._lib == 'gpiozero':
            self._led.off()

    def close(self):
        if self._lib == 'RPi.GPIO':
            try: self.GPIO.cleanup(self.pin)
            except: pass
        elif self._lib == 'gpiozero':
            try: self._led.close()
            except: pass

# ---------------- helpers ----------------
def safe_div(a, b): return (a / b) if b != 0.0 else 0.0

def score_index(val, fresh_min_or_max, stale_max_or_min, higher_is_better):
    if higher_is_better:
        fresh_min = fresh_min_or_max; stale_max = stale_max_or_min
        if val >= fresh_min: return 20
        if val <= stale_max: return 0
        return int(20.0 * (val - stale_max) / (fresh_min - stale_max))
    else:
        fresh_max = fresh_min_or_max; stale_min = stale_max_or_min
        if val <= fresh_max: return 20
        if val >= stale_min: return 0
        return int(20.0 * (stale_min - val) / (stale_min - fresh_max))

def scale_pts(pts0to20, weight): return int((pts0to20 * weight) / 20)

# --------------- stage 1: camera classify ---------------
def find_fruit_from_camera(led=None):
    model = load_model(MODEL_PATH, compile=False, custom_objects= CUSTOM_OBJECTS)
    with open(LABELS_PATH, "r") as f:
        class_names = [ln.strip() for ln in f.readlines()]

    data = np.empty((1, 224, 224, 3), dtype=np.float32)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        if led: 
            try: led.off(); led.close()
            except: pass
        sys.exit(1)
    
    if led:
        try: led.on()
        except: pass

    def cleanup(_s=None,_f=None):
        try: cap.release()
        except: pass
        cv2.destroyAllWindows()
        if led:
            try: led.off(); led.close()
            except: pass
        if _s is not None:
            sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    stable_count = 0
    best_label = "null"
    best_conf  = 0.0
    frame_idx  = 0

    print("📷 Looking for fruit in camera… (press q to cancel)")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame")
            break

        frame_idx += 1

        # preprocess
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.fit(image, INPUT_SIZE, Image.Resampling.LANCZOS)
        image_array = np.asarray(image, dtype=np.float32)
        data[0] = (image_array / 127.5) - 1.0

        # predict
        preds = model.predict(data, verbose=0)
        probs = preds[0]
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])
        label = class_names[idx] if conf >= CONF_THRESHOLD else "null"

        # overlay live view
        cv2.putText(frame, f"{label} ({conf:.2f})", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Fruit Finder", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cleanup()
            break

        # stabilize: ignore warmup, then require N consecutive hits
        if frame_idx > WARMUP_FRAMES and label != "null":
            if label == best_label:
                stable_count += 1
                best_conf = max(best_conf, conf)
            else:
                best_label = label
                best_conf  = conf
                stable_count = 1

            if stable_count >= STABLE_HITS:
                print(f"🍏 Detected fruit: {best_label} (conf ~{best_conf:.2f})")
                cleanup()  # closes camera window before spectral phase
                return best_label, best_conf

    cleanup()
    return "null", 0.0

# --------------- stage 2: spectral freshness ---------------
def measure_freshness_once():
    sensor = qwiic_as7265x.QwiicAS7265x()
    if not sensor.begin():
        print("AS7265X sensor not found!")
        return

    # configure sensor
    try: sensor.set_integration_cycles(INTEG_CYCLES)
    except AttributeError: sensor.setIntegrationCycles(INTEG_CYCLES)

    try: sensor.set_gain(GAIN_SETTING)
    except AttributeError: sensor.setGain(GAIN_SETTING)

    # LED current constant (resolve from driver)
    global LED_CURRENT_CONST
    if LED_CURRENT_CONST is None:
        LED_CURRENT_CONST = getattr(sensor, "kLedCurrentLimit25mA",
                             getattr(sensor, "kLED_CURRENT_LIMIT_25MA", None))

    led_white = getattr(sensor, "kLedWhite",
                 getattr(sensor, "kLED_WHITE", None))

    # set LED current & enable white bulb
    if LED_CURRENT_CONST is not None and led_white is not None:
        try:
            sensor.set_bulb_current(LED_CURRENT_CONST, led_white)  # (current, device)
        except AttributeError:
            sensor.setBulbCurrent(led_white, LED_CURRENT_CONST)    # some old libs swap order
        try:
            sensor.enable_bulb(led_white)
        except AttributeError:
            sensor.enableBulb(led_white)

    # average a few frames
    sumE=sumF=sumG=sumH=sumI=sumJ=sumS=sumT=sumU=sumV=sumW=sumL=0.0
    valid = 0
    for _ in range(NUM_FRAMES):
        try:
            if hasattr(sensor, "take_measurements_with_bulb"):
                sensor.take_measurements_with_bulb()
            else:
                try: sensor.take_measurements()
                except AttributeError: sensor.takeMeasurements()
        except Exception:
            time.sleep(0.15)
            continue

        # read channels helper
        def ch(name):
            lname=name.lower()
            for meth in (f"get_calibrated_{lname}", f"getCalibrated{name}"):
                fn = getattr(sensor, meth, None)
                if callable(fn): return fn()
            raise AttributeError(meth)

        try:
            E,F,G,H = ch("E"),ch("F"),ch("G"),ch("H")
            I,J,S   = ch("I"),ch("J"),ch("S")
            T,U,V   = ch("T"),ch("U"),ch("V")
            W,L     = ch("W"),ch("L")
        except Exception:
            time.sleep(0.15); continue

        # simple saturation gate
        if all(x < SAT_CUTOFF for x in [E,F,G,H,I,J,S,T,U,V,W,L]):
            sumE+=E; sumF+=F; sumG+=G; sumH+=H; sumI+=I; sumJ+=J
            sumS+=S; sumT+=T; sumU+=U; sumV+=V; sumW+=W; sumL+=L
            valid += 1
        time.sleep(0.12)

    # turn off bulb if API present
    try:
        if led_white is not None:
            try: sensor.disable_bulb(led_white)
            except AttributeError: sensor.disableBulb(led_white)
    except Exception:
        pass

    if valid < 2:
        print("MEASUREMENT ERROR – adjust position/lighting")
        return

    # averages
    E=sumE/valid; F=sumF/valid; G=sumG/valid; H=sumH/valid
    I=sumI/valid; J=sumJ/valid; S=sumS/valid
    T=sumT/valid; U=sumU/valid; V=sumV/valid; W=sumW/valid; L=sumL/valid

    total_sig = E+F+G+H+I+J+S+T+U+V+W+L
    if total_sig < MIN_TOTAL_SIGNAL:
        print("NO FRUIT / very low signal – move closer or increase illumination")
        return

    # indices
    green_mean = (E+F+G+H)/4.0
    GR   = safe_div(green_mean, S)
    ARI  = (safe_div(1.0, G) - safe_div(1.0, J))        # 1/560 - 1/705
    NIR  = (V + W) / 2.0
    NDVI = ((NIR - S) / (NIR + S)) if (NIR + S) > 0 else 0.0
    NDWI = ((W - L) / (W + L)) if (W + L) > 0 else 0.0
    T_over_S = safe_div(T, S)                            # 730/680
    U_over_S = safe_div(U, S)                            # 760/680

    # scoring (0..100)
    pts_GR   = scale_pts(score_index(GR,   GR_FRESH_MIN,   GR_STALE_MAX,   True),  W_GR)
    pts_ARI  = scale_pts(score_index(ARI,  ARI_FRESH_MAX,  ARI_STALE_MIN,  False), W_ARI)
    pts_NDWI = scale_pts(score_index(NDWI, NDWI_FRESH_MAX, NDWI_STALE_MIN, False), W_NDWI)
    pts_TOS  = scale_pts(score_index(T_over_S, TOS_FRESH_MAX, TOS_STALE_MIN, False), W_TOS)
    pts_UOS  = scale_pts(score_index(U_over_S, UOS_FRESH_MAX, UOS_STALE_MIN, False), W_UOS)

    score = max(0, min(100, pts_GR + pts_ARI + pts_NDWI + pts_TOS + pts_UOS))

    # category & probability
    if score >= 60:
        category = "FRESH"
        probability = 70 + int((score - 60) * 1.0)
    elif score >= 50:
        category = "AVERAGE"
        probability = 50 + int((score - 50) * 1.67)
    else:
        category = "STALE"
        probability = int(score * 1.0)

    probability = max(0, min(100, probability))

    # confidence heuristic by signal
    if   total_sig >= 2000: confidence = 95
    elif total_sig >= 1000: confidence = 85
    elif total_sig >=  400: confidence = 70
    elif total_sig >=  200: confidence = 55
    else:                   confidence = 40

    # print results
    print("\n=== Spectral Freshness ===")
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
    print("========================================\n")

def main():
    led = LedStrip() 
    label, conf = find_fruit_from_camera(led=led)
    if label == "null":
        print("No confident fruit detected — exiting.")
        return

    print(f"Detected: {label} (conf {conf:.2f})")

    measure_freshness_once()

if __name__ == "__main__":
    main()