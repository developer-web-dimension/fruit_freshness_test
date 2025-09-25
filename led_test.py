# import time, board, neopixel

# LED_PIN   = board.D18
# LED_COUNT = 30

# # Lower brightness if your PSU is small (full white draws a lot!)
# strip = neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=1.0, auto_write=False)

# try:
#     # Pure white for RGB strips = (255,255,255)
#     strip.fill((255, 255, 255))
#     strip.show()
#     print("White ON. Press Ctrl+C to turn off.")
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     pass
# finally:
#     strip.fill((0, 0, 0))
#     strip.show()


import time, board, neopixel
import colorsys

LED_PIN   = board.D18
LED_COUNT = 30

strip = neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=1.0, auto_write=False)

def wheel_cycle(wait=0.05):
    """Cycle through all colors across 360° hue."""
    while True:
        for hue in range(360):
            # Convert hue (0–1), saturation=1, value=1 to RGB
            r, g, b = colorsys.hsv_to_rgb(hue/360.0, 1.0, 1.0)
            # Scale to 0–255
            color = (int(r*255), int(g*255), int(b*255))
            strip.fill(color)
            strip.show()
            time.sleep(wait)

try:
    print("Cycling through all colors... Press Ctrl+C to stop.")
    wheel_cycle(0.05)   # adjust wait for speed (smaller = faster)
except KeyboardInterrupt:
    pass
finally:
    strip.fill((0, 0, 0))
    strip.show()
    print("LEDs off.")
