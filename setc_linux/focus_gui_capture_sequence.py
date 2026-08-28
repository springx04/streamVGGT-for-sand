import ctypes
import time

from PIL import ImageGrab


x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
xtst = ctypes.cdll.LoadLibrary("libXtst.so.6")
x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
x11.XKeysymToKeycode.restype = ctypes.c_uint
display = x11.XOpenDisplay(None)
if not display:
    raise RuntimeError("cannot open X display")

# Alt+Tab from the editor back to the already-running GUI.
alt = x11.XKeysymToKeycode(display, 0xFFE9)
tab = x11.XKeysymToKeycode(display, 0xFF09)
xtst.XTestFakeKeyEvent(display, alt, True, 0)
xtst.XTestFakeKeyEvent(display, tab, True, 0)
xtst.XTestFakeKeyEvent(display, tab, False, 0)
xtst.XTestFakeKeyEvent(display, alt, False, 0)
x11.XFlush(display)
time.sleep(2.0)

for index in range(10):
    image = ImageGrab.grab()
    image.save(f"/tmp/focused_gui_{index:02d}.jpg", quality=92)
    image.crop((490, 198, 1120, 705)).save(
        f"/tmp/focused_cloud_{index:02d}.jpg", quality=94
    )
    time.sleep(0.8)

x11.XCloseDisplay(display)
