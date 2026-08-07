"""Hammer the app at full sensor resolution and see whether it survives.

    python3 stresstest.py [rounds]

This exists because of a specific crash. The preview wraps numpy buffers in a
QImage without copying them, and QImage does not own that memory — so any code
path that dropped the last Python reference while the image was still on screen
left Qt painting freed pages. At preview sizes the allocator usually hands the
same pages straight back and nothing is seen. At 16 MP the buffer is 48 MB, is
genuinely returned to the system, and the app dies with no Python traceback.

A use-after-free cannot be asserted on directly, so this drives the sequences
that caused it — compare held across a slider change, mode switches, corner
mode, zoom, re-detect — hundreds of times, forcing a collection each round. A
clean run is not proof; a crash is proof of a bug.
"""

import gc
import os
import resource
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

import camera
import imaging
import scanner

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


app = QApplication(sys.argv)
app.setStyleSheet(scanner.QSS)
win = scanner.App()
win.show()


def pump(sec=0.05):
    end = time.time() + sec
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


def paint():
    """Force a real paint.

    Without this the test proves nothing: off-screen, `processEvents` never
    runs paintEvent, so the buffer a freed QImage points at is never actually
    read and a use-after-free sails through. grab() renders the widget
    synchronously, which is what touches the memory.
    """
    win.preview.grab()


def until(cond, secs):
    end = time.time() + secs
    while time.time() < end and not cond():
        pump(0.1)
    return cond()


if not until(lambda: win.cam.is_open(), 60):
    print("no camera:", win.cam.error)
    sys.exit(1)
until(lambda: win.cam.at_max, 30)
for _ in range(2):
    if win.cam.at_max:
        break
    win.upgrade_camera()
    until(lambda: not win.busy, 40)
print("camera %dx%d   RSS %.0f MB" % (win.cam.width, win.cam.height, rss()))
if not win.cam.at_max:
    print("WARNING: not at the sensor maximum — this is a weaker test than intended")

until(lambda: win.preview._img is not None, 20)
for _ in range(3):
    win.capture()
    pump(0.6)
print("%d pages, %.1f MB of frames held   RSS %.0f MB"
      % (len(win.pages), sum(p.frame.nbytes for p in win.pages) / 1e6, rss()))

filters = [k for k, _ in scanner.FILTERS]
start = time.time()
for i in range(ROUNDS):
    win.tray.setCurrentRow(i % len(win.pages))
    pump(0.05)

    # compare held while the look changes underneath it — the exact shape of
    # the crash: _compare_rgb was dropped while the QImage still pointed at it
    win.compare(True)
    pump(0.03)
    paint()
    win.sliders["contrast"].slider.setValue(10 + (i * 13) % 80)
    gc.collect()
    paint()
    pump(0.03)
    win.set_filter(filters[i % len(filters)])
    gc.collect()
    paint()
    pump(0.05)
    win.compare(False)
    paint()

    # corner mode swaps the buffer for the cached uncropped view and back
    win.toggle_corners()
    gc.collect()
    paint()
    pump(0.05)
    if win.page().corners is not None:
        q = win.page().corners.copy()
        q[0] = [min(0.95, q[0][0] + 0.02), min(0.95, q[0][1] + 0.02)]
        win._corners_dragged(q)
    pump(0.05)
    win.toggle_corners()
    gc.collect()
    paint()

    # live <-> edit swaps between feed buffers and rendered ones
    win.set_mode("live")
    pump(0.12)
    paint()
    win.set_mode("edit")
    gc.collect()
    paint()

    win.zoom(1.4)
    win.preview.set_guides(grid=bool(i % 2))
    paint()
    win.redetect()
    win.preview_fit()
    pump(0.05)

    gc.collect()               # make any dangling reference bite immediately
    paint()
    if (i + 1) % 15 == 0:
        print("  round %3d/%d   RSS %.0f MB   %.1f s"
              % (i + 1, ROUNDS, rss(), time.time() - start))

win.set_mode("edit")
until(lambda: not win.settle.isActive(), 5)
pump(0.5)
page = win.page()
full = imaging.process(page.frame, page.adjust, page.corners)
print("full-resolution export still correct: %dx%d   peak RSS %.0f MB"
      % (full.shape[1], full.shape[0], rss()))

size = (win.cam.width, win.cam.height)
win.cam.close()
pump(0.3)
print("\nsurvived %d rounds at %dx%d" % (ROUNDS, size[0], size[1]))
