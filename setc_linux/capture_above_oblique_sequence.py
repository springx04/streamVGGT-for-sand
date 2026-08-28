import ctypes
import time

from PIL import ImageGrab


x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
xtst = ctypes.cdll.LoadLibrary("libXtst.so.6")
x11.XOpenDisplay.restype = ctypes.c_void_p
display = x11.XOpenDisplay(None)
if not display:
    raise RuntimeError("cannot open X display")


def move(x, y):
    xtst.XTestFakeMotionEvent(display, -1, x, y, 0)
    x11.XFlush(display)


def drag(dx, dy):
    x0, y0 = 800, 450
    move(x0, y0)
    xtst.XTestFakeButtonEvent(display, 1, True, 0)
    for step in range(1, 25):
        move(x0 + dx * step // 24, y0 + dy * step // 24)
        time.sleep(0.012)
    xtst.XTestFakeButtonEvent(display, 1, False, 0)
    x11.XFlush(display)
    time.sleep(1.0)


# Tilt away from the near-top view while staying above the reconstructed floor.
drag(0, 70)
for index in range(10):
    image = ImageGrab.grab().crop((490, 198, 1120, 705))
    image.save(f"/tmp/above_oblique_{index:02d}.jpg", quality=92)
    time.sleep(0.8)

x11.XCloseDisplay(display)
