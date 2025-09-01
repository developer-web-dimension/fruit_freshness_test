#!/usr/bin/env python3
import os
import signal
import sys
import time
import numpy as np
import cv2
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow.keras.models import load_model

def DepthwiseConv2D_compat(**kwargs):
    kwargs.pop("groups", None)
    return tf.keras.layers.DepthwiseConv2D(**kwargs)

CUSTOM_OBJECTS = {"DepthwiseConv2D": DepthwiseConv2D_compat}

np.set_printoptions(suppress=True)

MODEL_PATH = "/home/Subral/python/arduino/model/keras_model.h5"
LABELS_PATH = "/home/Subral/python/arduino/model/labels.txt"

# Load model (try with compat first; if not needed, it still works)
model = load_model(MODEL_PATH, compile=False, custom_objects=CUSTOM_OBJECTS)

# Load labels
with open(LABELS_PATH, "r") as f:
    class_names = [ln.strip() for ln in f.readlines()]

# Inference settings
INPUT_SIZE = (224, 224)
CONF_THRESHOLD = 0.70

# Preallocate input buffer (N,H,W,C) float32
data = np.empty((1, 224, 224, 3), dtype=np.float32)

# Camera
cap = cv2.VideoCapture(0)  # change index if you have multiple cameras
if not cap.isOpened():
    print("ERROR: Could not open camera.")
    sys.exit(1)

def cleanup_and_exit(_sig=None, _frame=None):
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
    sys.exit(0)

# Graceful exit on Ctrl+C
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to grab frame")
        break

    # Convert BGR -> RGB, then PIL for the same preprocessing you used
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Resize/crop to 224x224
    image = ImageOps.fit(image, INPUT_SIZE, Image.Resampling.LANCZOS)

    # To numpy
    image_array = np.asarray(image, dtype=np.float32)

    # Normalize to [-1, 1] (Teachable-Machine style)
    data[0] = (image_array / 127.5) - 1.0

    # Predict
    preds = model.predict(data, verbose=0)          # shape: (1, num_classes)
    probs = preds[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])

    label = class_names[idx] if conf >= CONF_THRESHOLD else "null"

    # Overlay
    cv2.putText(
        frame,
        f"{label} ({conf:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Real-Time Classification", frame)
    # ~33 FPS key polling; press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cleanup_and_exit()
