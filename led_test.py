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



# # #!/usr/bin/env python3
# # import time, math, random, argparse, colorsys
# # import board, neopixel

# # # ====== Config ======
# # LED_PIN   = board.D18          # your data pin
# # LED_COUNT = 30                 # number of LEDs
# # AUTO_WRITE = False             # faster updates
# # DEFAULT_BRIGHTNESS = 0.3       # 0..1
# # # =====================

# # strip = None  # global handle set in main()


# # # ---------- helpers ----------
# # def parse_hex_color(s):
# #     s = s.strip().lstrip("#")
# #     r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
# #     return (r, g, b)

# # def clamp01(x): return 0.0 if x < 0 else 1.0 if x > 1 else x

# # def rgb_to_rgbw(rgb):
# #     """Naive RGB->RGBW extraction: move common part to W channel."""
# #     r, g, b = rgb
# #     w = min(r, g, b)
# #     return (r - w, g - w, b - w, w)

# # def make_color(rgb):
# #     """Return a tuple sized to the strip (RGB or RGBW)."""
# #     if strip.bpp == 4:
# #         return rgb_to_rgbw(rgb)
# #     return rgb

# # def scale_color(rgb, s):
# #     r, g, b = rgb
# #     return (int(r*s), int(g*s), int(b*s))

# # def wheel(pos):
# #     """0..255 -> RGB rainbow color"""
# #     pos %= 256
# #     if pos < 85:
# #         return (255 - pos*3, pos*3, 0)
# #     if pos < 170:
# #         pos -= 85
# #         return (0, 255 - pos*3, pos*3)
# #     pos -= 170
# #     return (pos*3, 0, 255 - pos*3)

# # def show():
# #     strip.show()

# # def black():
# #     if strip.bpp == 4: strip.fill((0,0,0,0))
# #     else:              strip.fill((0,0,0))


# # # ---------- styles (effects) ----------
# # def style_solid(rgb, duration, **kw):
# #     strip.fill(make_color(rgb)); show()
# #     t0 = time.time()
# #     while duration < 0 or (time.time()-t0) < duration:
# #         time.sleep(0.1)

# # def style_blink(rgb, duration, on=0.3, off=0.3, **kw):
# #     t0 = time.time()
# #     while duration < 0 or (time.time()-t0) < duration:
# #         strip.fill(make_color(rgb)); show(); time.sleep(on)
# #         black(); show(); time.sleep(off)

# # def style_wipe(rgb, duration, delay=0.02, **kw):
# #     t0 = time.time()
# #     while duration < 0 or (time.time()-t0) < duration:
# #         for i in range(len(strip)):
# #             strip[i] = make_color(rgb); show(); time.sleep(delay)
# #         time.sleep(0.3)
# #         black(); show(); time.sleep(0.3)

# # def style_breathe(rgb, duration, speed=1.5, **kw):
# #     """Sinusoidal brightness 0->1->0."""
# #     t0 = time.time()
# #     while duration < 0 or (time.time()-t0) < duration:
# #         phase = (time.time()*speed) % (2*math.pi)
# #         s = 0.5*(1 - math.cos(phase))  # 0..1
# #         c = make_color(scale_color(rgb, s))
# #         for i in range(len(strip)): strip[i] = c
# #         show(); time.sleep(0.01)

# # def style_theater(rgb, duration, gap=3, delay=0.05, **kw):
# #     t0 = time.time(); n = len(strip)
# #     step = 0
# #     while duration < 0 or (time.time()-t0) < duration:
# #         black()
# #         for i in range(step, n, gap):
# #             strip[i] = make_color(rgb)
# #         show(); time.sleep(delay)
# #         step = (step + 1) % gap

# # def style_rainbow(duration, speed=2.0, **kw):
# #     t0 = time.time(); n = len(strip)
# #     while duration < 0 or (time.time()-t0) < duration:
# #         base = int(time.time()*256*speed)  # animate hue
# #         for i in range(n):
# #             strip[i] = make_color(wheel((i*256//n + base) & 255))
# #         show(); time.sleep(0.01)

# # def style_comet(rgb, duration, tail=10, delay=0.02, fade=0.7, **kw):
# #     n = len(strip); t0 = time.time(); head = 0
# #     buf = [(0,0,0)]*n
# #     while duration < 0 or (time.time()-t0) < duration:
# #         # fade buffer
# #         buf = [scale_color(c, fade) for c in buf]
# #         # draw head
# #         buf[head] = rgb
# #         # copy to strip with tail blend
# #         for i in range(n):
# #             strip[i] = make_color(buf[i])
# #         show(); time.sleep(delay)
# #         head = (head + 1) % n

# # def style_sparkle(rgb, duration, density=0.1, delay=0.02, **kw):
# #     n = len(strip); t0 = time.time()
# #     while duration < 0 or (time.time()-t0) < duration:
# #         # decay
# #         for i in range(n):
# #             if random.random() < 0.2: strip[i] = make_color(scale_color((*(strip[i][:3]),)[0:3], 0.5))  # gentle fade
# #         # new sparks
# #         hits = max(1, int(n*density*0.1))
# #         for _ in range(hits):
# #             i = random.randrange(n)
# #             strip[i] = make_color(rgb)
# #         show(); time.sleep(delay)

# # def style_gradient(color1, color2, duration, **kw):
# #     """Static 2-color gradient across strip."""
# #     n = len(strip); c1, c2 = color1, color2
# #     for i in range(n):
# #         t = i/(n-1 if n>1 else 1)
# #         r = int(c1[0]*(1-t) + c2[0]*t)
# #         g = int(c1[1]*(1-t) + c2[1]*t)
# #         b = int(c1[2]*(1-t) + c2[2]*t)
# #         strip[i] = make_color((r,g,b))
# #     show()
# #     t0 = time.time()
# #     while duration < 0 or (time.time()-t0) < duration:
# #         time.sleep(0.1)


# # STYLES = {
# #     "solid":    style_solid,
# #     "blink":    style_blink,
# #     "wipe":     style_wipe,
# #     "breathe":  style_breathe,
# #     "theater":  style_theater,
# #     "rainbow":  style_rainbow,
# #     "comet":    style_comet,
# #     "sparkle":  style_sparkle,
# #     "gradient": style_gradient,
# # }


# # def main():
# #     global strip
# #     p = argparse.ArgumentParser(description="NeoPixel style runner")
# #     p.add_argument("--style", choices=STYLES.keys(), default="solid")
# #     p.add_argument("--color", default="#ffffff", help="hex RGB for solid/blink/wipe/breathe/comet/sparkle")
# #     p.add_argument("--color2", default="#0000ff", help="hex RGB for gradient second color")
# #     p.add_argument("--brightness", type=float, default=DEFAULT_BRIGHTNESS)
# #     p.add_argument("--duration", type=float, default=-1, help="seconds; -1 = run forever")
# #     # style-specific tunables
# #     p.add_argument("--speed", type=float, default=1.5, help="breathe/rainbow speed")
# #     p.add_argument("--on", type=float, default=0.3, help="blink on-time")
# #     p.add_argument("--off", type=float, default=0.3, help="blink off-time")
# #     p.add_argument("--delay", type=float, default=0.02, help="frame delay for wipe/comet/etc.")
# #     args = p.parse_args()

# #     strip = neopixel.NeoPixel(
# #         LED_PIN, LED_COUNT,
# #         brightness=clamp01(args.brightness),
# #         auto_write=AUTO_WRITE
# #     )

# #     rgb  = parse_hex_color(args.color)
# #     rgb2 = parse_hex_color(args.color2)

# #     try:
# #         fn = STYLES[args.style]
# #         if args.style == "gradient":
# #             fn(rgb, rgb2, args.duration)
# #         elif args.style in ("breathe", "rainbow"):
# #             fn(rgb if args.style=="breathe" else None, args.duration, speed=args.speed)
# #         elif args.style == "blink":
# #             fn(rgb, args.duration, on=args.on, off=args.off)
# #         elif args.style in ("wipe","comet","sparkle"):
# #             fn(rgb, args.duration, delay=args.delay)
# #         else:
# #             fn(rgb, args.duration)
# #     except KeyboardInterrupt:
# #         pass
# #     finally:
# #         black(); show()

# # if __name__ == "__main__":
# #     main()

