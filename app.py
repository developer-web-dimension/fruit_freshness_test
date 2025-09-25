from __future__ import annotations

import sys, time, signal
from typing import Optional, Dict
import board
import neopixel
import numpy as np
import cv2
from PIL import Image, ImageOps
import os
import subprocess
import threading

import tensorflow as tf
from tensorflow.keras.models import load_model
import qwiic_as7265x

#        CONFIG
# Paths for your model/labels
MODEL_PATH  = "/home/Subral/python/fresh_fruit/model/keras_model.h5"
LABELS_PATH = "/home/Subral/python/fresh_fruit/model/labels.txt"

# Camera inference
INPUT_SIZE      = (224, 224)
CONF_THRESHOLD  = 0.70
CAMERA_INDEX    = 0

# NeoPixel strip
LED_PIN         = board.D18
LED_COUNT       = 30
LED_BRIGHTNESS  = 1.0  # reduce if PSU is small

# AS7265x capture config
NUM_FRAMES        = 3
SAT_CUTOFF        = 3000.0
MIN_TOTAL_SIGNAL  = 60.0
INTEG_CYCLES      = 60
GAIN_SETTING      = 3
LED_CURRENT       = 2  # usually 25mA enum in driver

# Apple thresholds
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

SOUND_MAP = {
    "FRESH":   "/home/Subral/python/fresh_fruit/sounds/fresh.wav",
    "AVERAGE": "/home/Subral/python/fresh_fruit/sounds/average.wav",
    "STALE":   "/home/Subral/python/fresh_fruit/sounds/stale.wav",
}

# Weights
W_GR   = 5
W_ARI  = 40
W_NDWI = 10
W_TOS  = 5
W_UOS  = 5

def play_category_sound(category: str):
    """Play a short WAV for the given category using aplay (non-blocking)."""
    wav = SOUND_MAP.get(category.upper())
    if not wav or not os.path.exists(wav):
        return  # silent if file missing
    # Run aplay quietly, detached, so it never blocks your loop
    def _run():
        try:
            subprocess.run(["aplay", "-q", wav], check=False)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


def disable_bulb(sensor, which="white"):
    try:
        call_any(sensor, ["disable_bulb", "disableBulb"], which)
    except Exception:
        # Fallback for older API versions
        call_any(sensor, ["disable_indicator", "disableIndicator"])

def white_on(sensor):
    """Turn ON AS7265x white lamp with API compatibility."""
    try:
        sensor.enable_bulb(sensor.kLedWhite)
        try:
            sensor.set_bulb_current(sensor.kLedCurrentLimit25mA, sensor.kLedWhite)
        except Exception:
            pass
    except Exception:
        enable_bulb(sensor, "white")
        set_bulb_current(sensor, LED_CURRENT, "white")

def white_off(sensor):
    """Turn OFF AS7265x white lamp with API compatibility."""
    try:
        sensor.disable_bulb(sensor.kLedWhite)
    except Exception:
        disable_bulb(sensor, "white")



#      COMPAT SHIMS
def DepthwiseConv2D_compat(**kwargs):
    """
    Compatibility shim for DepthwiseConv2D layer loading.
    Removes unsupported 'groups' parameter for TensorFlow 2.15+ with Python 3.11
    """
    kwargs.pop("groups", None)
    kwargs.pop("depth_multiplier", kwargs.get("depth_multiplier", 1))
    return tf.keras.layers.DepthwiseConv2D(depth_multiplier=kwargs.get("depth_multiplier", 1), **kwargs)

CUSTOM_OBJECTS = {"DepthwiseConv2D": DepthwiseConv2D_compat}

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



# --- ADD: simple soundboard ---
class SoundBoard:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.cache = {}
        # init mixer once
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

    def _load(self, path):
        if path in self.cache:
            return self.cache[path]
        if not os.path.isfile(path):
            print(f"[audio] Missing file: {path}")
            return None
        try:
            snd = pygame.mixer.Sound(path)
            self.cache[path] = snd
            return snd
        except Exception as e:
            print(f"[audio] Failed to load {path}: {e}")
            return None

    def play(self, filename: str, block: bool=False):
        path = os.path.join(self.base_dir, filename)
        snd = self._load(path)
        if not snd:
            return
        ch = snd.play()
        if block and ch is not None:
            while ch.get_busy():
                pygame.time.wait(50)

    def stop_all(self):
        pygame.mixer.stop()


#     LED STRIP CONTROL
class LedStrip:
    """Control NeoPixel LED strip for illumination during camera capture."""
    
    def __init__(self, pin=LED_PIN, count=LED_COUNT, brightness=LED_BRIGHTNESS):
        print(f"Initializing LED strip: Pin {pin}, Count {count}, Brightness {brightness}")
        self.strip = neopixel.NeoPixel(pin, count, brightness=brightness, auto_write=False)
        self.is_on = False

    def on_white(self):
        """Turn on all LEDs with white color."""
        print("LED: Turning ON white LEDs for camera illumination")
        self.strip.fill((255, 255, 255))
        self.strip.show()
        self.is_on = True

    def off(self):
        """Turn off all LEDs."""
        print("LED: Turning OFF LEDs")
        self.strip.fill((0, 0, 0))
        self.strip.show()
        self.is_on = False

    def status(self):
        """Return current LED status."""
        return "ON" if self.is_on else "OFF"


#     CAMERA + CLASSIFIER
class FruitClassifier:
    """
    Real-time fruit classification using TensorFlow model and OpenCV camera.
    """
    
    def __init__(self, model_path: str, labels_path: str, input_size=(224,224), conf_thresh=0.70, cam_index=0):
        # Load TensorFlow model with custom objects for compatibility
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path, compile=False, custom_objects=CUSTOM_OBJECTS)
        
        # Load class labels
        print(f"Loading labels from: {labels_path}")
        with open(labels_path, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(self.class_names)} classes: {self.class_names}")
        
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.cam_index = cam_index
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Pre-allocate buffer for better performance
        self.buf = np.empty((1, input_size[1], input_size[0], 3), dtype=np.float32)

    def start_camera(self):
        """Initialize camera capture."""
        print("Starting camera...")
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("ERROR: Could not open camera.")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # tune per camera; negative logs for UVC cams
        t0 = time.time()
        while time.time() - t0 < 0.5:  # ~0.5s warmup
            self.cap.read()
        print("Camera started successfully!")

    def stop_camera(self):
        """Clean up camera resources."""
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("Camera stopped")

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess camera frame for model inference.
        Converts BGR to RGB, resizes, and normalizes to [-1, 1] range.
        """
        # Convert BGR to RGB and create PIL Image
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        # Resize to model input size using high-quality resampling
        image = ImageOps.fit(image, self.input_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        arr = np.asarray(image, dtype=np.float32)
        self.buf[0] = (arr / 127.5) - 1.0  # Normalize to [-1, 1]
        return self.buf

    def detect_until_found(self, led_controller: LedStrip) -> tuple[str, float]:
        cv2.namedWindow("Real-Time Classification (strobe)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real-Time Classification (strobe)", 800, 600)

        if self.cap is None:
            raise RuntimeError("Camera not started. Call start_camera() first.")

        print("Starting detection loop with per-frame LED strobe (ON->capture->OFF)")
        frame_count = 0

        while True:
            # 1) LED ON + tiny settle
            if not led_controller.is_on:
                led_controller.on_white()
                # allow LED brightness + camera exposure to stabilize a bit
                time.sleep(0.1)  

            # 2) Captur
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # show a black frame so the window visibly updates
                cv2.imshow("Real-Time Classification (strobe)", np.zeros((480, 640, 3), dtype=np.uint8))
                cv2.waitKey(1)
                continue
            # 3) LED OFF immediately after acquiring the frame
            led_controller.off()

            frame_count += 1

            # ---- inference as-is ----
            data = self._preprocess(frame)
            preds = self.model.predict(data, verbose=0)[0]
            idx = int(np.argmax(preds))
            confidence = float(preds[idx])
            label = self.class_names[idx]

            # draw overlay & show
            cv2.putText(frame, f"{label} ({confidence:.2f})", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("Real-Time Classification (strobe)", frame)
            cv2.waitKey(1)

            if confidence >= self.conf_thresh and label.lower() != "null":
                print(f"\n*** PRODUCT DETECTED: {label} ({confidence:.2f}) ***")
                cv2.destroyAllWindows()
                return (label, confidence)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                print("Detection cancelled by user")
                raise KeyboardInterrupt


#     AS7265x ANALYZER
def safe_div(numerator, denominator):
    """Safe division that returns 0 if denominator is 0."""
    return (numerator / denominator) if denominator != 0.0 else 0.0

def score_index(value, fresh_threshold, stale_threshold, higher_is_better):
    """
    Score a spectral index value based on freshness thresholds.
    Returns a score from 0-20 points.
    """
    if higher_is_better:
        fresh_min, stale_max = fresh_threshold, stale_threshold
        if value >= fresh_min:
            return 20
        if value <= stale_max:
            return 0
        # Linear interpolation between thresholds
        return int(20.0 * (value - stale_max) / (fresh_min - stale_max))
    else:
        fresh_max, stale_min = fresh_threshold, stale_threshold
        if value <= fresh_max:
            return 20
        if value >= stale_min:
            return 0
        # Linear interpolation between thresholds
        return int(20.0 * (stale_min - value) / (stale_min - fresh_max))

def scale_pts(points_0to20, weight):
    """Scale 0-20 point score by weight factor."""
    return int((points_0to20 * weight) / 20)

# Sensor API compatibility functions
def set_integration_cycles(sensor, cycles):
    """Set integration cycles with API compatibility."""
    call_any(sensor, ["set_integration_cycles", "setIntegrationCycles"], cycles)

def set_gain(sensor, gain_enum):
    """Set sensor gain with API compatibility."""
    call_any(sensor, ["set_gain", "setGain"], gain_enum)

def take_measurements(sensor):
    """Trigger sensor measurements with API compatibility."""
    call_any(sensor, ["take_measurements", "takeMeasurements"])

def enable_bulb(sensor, which="white"):
    """Enable sensor illumination bulb with API compatibility."""
    try:
        call_any(sensor, ["enable_bulb", "enableBulb"], which)
    except Exception:
        # Fallback for older API versions
        call_any(sensor, ["enable_indicator", "enableIndicator"])

def set_bulb_current(sensor, current_enum, which="white"):
    """Set bulb current with API compatibility."""
    try:
        call_any(sensor, ["set_bulb_current", "setBulbCurrent"], which, current_enum)
    except Exception:
        # Some versions may not support this function
        pass

def get_calibrated_channel(sensor, channel):
    """Get calibrated channel reading with API compatibility."""
    channel_lower = channel.lower()
    return call_any(sensor, [f"get_calibrated_{channel_lower}", f"getCalibrated{channel}"])

def run_freshness_analyzer_loop(detected_product: str, max_measurements: int = 3):
    """
    Capture up to `max_measurements` spectral measurements and then stop.
    Each measurement averages over NUM_FRAMES frames.
    """
    print(f"\n{'='*50}")
    print(f"SPECTRAL FRESHNESS ANALYSIS (max {max_measurements} runs)")
    print(f"Detected Product: {detected_product}")
    print(f"{'='*50}")
    
    _last_sound = {"cat": None}
    sensor = qwiic_as7265x.QwiicAS7265x()
    if not sensor.begin():
        print("AS7265x Sensor not found! Check wiring/I2C.")
        return

    print("AS7265x sensor initialized successfully!")

    # Configure
    set_integration_cycles(sensor, INTEG_CYCLES)
    set_gain(sensor, GAIN_SETTING)

    # Enable illumination
    try:
        sensor.enable_bulb(sensor.kLedWhite)
        sensor.set_bulb_current(sensor.kLedCurrentLimit25mA, sensor.kLedWhite)
    except Exception:
        enable_bulb(sensor, "white")
        set_bulb_current(sensor, LED_CURRENT, "white")
    print("Sensor LED illumination enabled")

    print(f"\nAnalyzing freshness of: {detected_product}")
    print("Position the fruit 3–5 cm from the sensor; use a dark shroud if possible.")
    time.sleep(2)

    # NOTE: if you truly have 18 channels, make sure this list matches your driver’s channel names.
    channels = list("EFGHIJSTUVWL")  # (kept as in your code)
    
    for measurement_count in range(1, max_measurements + 1):
        print(f"\n--- Measurement #{measurement_count} of {max_measurements} for {detected_product} ---")
        
        sum_values = {ch: 0.0 for ch in channels}
        valid_frames = 0

        for frame_num in range(NUM_FRAMES):
            try:
                take_measurements(sensor)
                readings = {ch: get_calibrated_channel(sensor, ch) for ch in channels}

                if all(value < SAT_CUTOFF for value in readings.values()):
                    for ch, val in readings.items():
                        sum_values[ch] += val
                    valid_frames += 1
                else:
                    print(f"Frame {frame_num + 1}: Saturation detected, skipping...")
            except Exception as e:
                print(f"Read error on frame {frame_num + 1}: {e}")
                time.sleep(0.3)
                continue

            time.sleep(0.10)

        if valid_frames < 2:
            print("MEASUREMENT ERROR — Low valid frames. Adjust position/lighting and continuing...")
            continue

        avg_values = {ch: sum_values[ch] / valid_frames for ch in channels}
        E, F, G, H = avg_values['E'], avg_values['F'], avg_values['G'], avg_values['H']
        I, J, S, T = avg_values['I'], avg_values['J'], avg_values['S'], avg_values['T']
        U, V, W, L = avg_values['U'], avg_values['V'], avg_values['W'], avg_values['L']

        total_signal = sum(avg_values.values())
        if total_signal < MIN_TOTAL_SIGNAL:
            print("NO FRUIT DETECTED / Very low signal — move fruit closer.")
            continue

        green_mean = (E + F + G + H) / 4.0
        GR  = safe_div(green_mean, S)
        ARI = (safe_div(1.0, G) - safe_div(1.0, J))
        NIR = (V + W) / 2.0
        NDVI = safe_div(NIR - S, NIR + S)
        NDWI = safe_div(W - L, W + L)
        T_over_S = safe_div(T, S)
        U_over_S = safe_div(U, S)

        pts_GR   = scale_pts(score_index(GR,  GR_FRESH_MIN,  GR_STALE_MAX,  True),  W_GR)
        pts_ARI  = scale_pts(score_index(ARI, ARI_FRESH_MAX, ARI_STALE_MIN, False), W_ARI)
        pts_NDWI = scale_pts(score_index(NDWI, NDWI_FRESH_MAX, NDWI_STALE_MIN, False), W_NDWI)
        pts_TOS  = scale_pts(score_index(T_over_S, TOS_FRESH_MAX, TOS_STALE_MIN, False), W_TOS)
        pts_UOS  = scale_pts(score_index(U_over_S, UOS_FRESH_MAX, UOS_STALE_MIN, False), W_UOS)

        total_score = max(0, min(100, pts_GR + pts_ARI + pts_NDWI + pts_TOS + pts_UOS))

        if total_score >= 60:
            category = "FRESH";   probability = 70 + min(30, int((total_score - 60) * 0.75))
        elif total_score >= 56:
            category = "AVERAGE"; probability = 45 + int((total_score - 45) * 1.67)
        else:
            category = "STALE";   probability = max(10, int(total_score * 0.9))
        probability = max(0, min(100, probability))

        if total_signal >= 2000:   confidence = 95
        elif total_signal >= 1000: confidence = 85
        elif total_signal >= 400:  confidence = 70
        elif total_signal >= 200:  confidence = 55
        else:                      confidence = 40


        if category != _last_sound["cat"]:
            play_category_sound(category)
            _last_sound["cat"] = category


        print(f"\n{'='*60}")
        print(f"FRESHNESS ANALYSIS RESULTAS")
        print(f"{'='*60}")
        print(f"FRESHNESS_SCORE: {total_score}/100")
        print(f"PROBABILITY: {probability}%")
        print(f"CATEGORY: {category}")
        print("\n--- Spectral Index Details ---")
        print(f"Green/Red Ratio: {GR:.3f} (points: {pts_GR})")
        print(f"Anthocyanin Index: {ARI:.6f} (points: {pts_ARI})")
        print(f"NDVI: {NDVI:.3f}")
        print(f"Water Index: {NDWI:.3f} (points: {pts_NDWI})")
        print(f"T/S Ratio: {T_over_S:.3f} (points: {pts_TOS})")
        print(f"U/S Ratio: {U_over_S:.3f} (points: {pts_UOS})")
        print(f"Total Signal Strength: {total_signal:.0f} (frames: {valid_frames})")
        print(f"{'='*60}")

        time.sleep(1.5)  # small pause between capped runs

    # Turn off sensor lamp before leaving
    try:
        white_off(sensor)
    except Exception:
        pass
    print("Reached max measurements; stopping spectral loop.")


#           MAIN
def main():
    """Main application entry point with complete LED control workflow."""
    print("=" * 60)
    print("FRUIT FRESHNESS ANALYZER")
    print("Advanced Computer Vision + Spectral Analysis System")
    print("=" * 60)
    
    # Initialize LED strip - LEDs turn ON at startup
    print("\nPhase 1: Initializing LED illumination system...")
    led = LedStrip()
    led.on_white()
    time.sleep(0.5)
    print("LEDs are now ON and will stay ON until product detection")

    # Initialize classifier
    print("\nPhase 2: Loading AI classification model...")
    classifier = FruitClassifier(
        model_path=MODEL_PATH,
        labels_path=LABELS_PATH,
        input_size=INPUT_SIZE,
        conf_thresh=CONF_THRESHOLD,
        cam_index=CAMERA_INDEX,
    )

    def cleanup_handler(sig=None, frame=None):
        """Clean shutdown handler for signals."""
        print("\n\nShutting down system...")
        try:
            classifier.stop_camera()
        except Exception:
            pass
        try:
            led.off()
            print("LEDs turned OFF")
        except Exception:
            pass
        print("System shutdown complete")
        sys.exit(0)

    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    detected_label = None
    confidence = 0.0

    try:
        # Phase 3: Camera-based product detection (LEDs stay ON)
        print("\nPhase 3: Starting camera-based product detection...")
        print("LEDs will remain ON during this phase for optimal illumination")
        
        classifier.start_camera()
        
        # Detection loop - LEDs stay on until product found
        detected_label, confidence = classifier.detect_until_found(led)
        print(f"\n*** DETECTION SUCCESSFUL ***")
        print(f"Product: {detected_label}")
        print(f"Confidence: {confidence:.2f}")
        print("LEDs have been turned OFF automatically")
        
    except KeyboardInterrupt:
        print("\nDetection cancelled by user")
        cleanup_handler()
    except Exception as e:
        print(f"\nError during detection: {e}")
        cleanup_handler()
    finally:
        # Ensure camera is stopped and LEDs are off
        try:
            classifier.stop_camera()
        except Exception:
            pass
        # LEDs should already be OFF from detection, but ensure it
        if led.is_on:
            led.off()

    # Phase 4: Spectral analysis for freshness assessment (LEDs remain OFF)
    print(f"\nPhase 4: Starting spectral freshness analysis...")
    
    if detected_label:
        try:
            run_freshness_analyzer_loop(detected_label, max_measurements=1)
        except KeyboardInterrupt:
            print(f"\nSpectral analysis stopped by user.")
        except Exception as e:
            print(f"Error during spectral analysis: {e}")
        finally:
            cleanup_handler()
    else:
        print("No product was detected, skipping spectral analysis")
        cleanup_handler()


if __name__ == "__main__":
    main()