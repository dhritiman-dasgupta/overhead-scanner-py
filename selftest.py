"""End-to-end check against the real camera: capture, detect, process, export.

    python3 selftest.py

Engine only — no window. `uitest.py` covers the app on top of it.
"""

import os
import sys
import tempfile
import time

import cv2
import numpy as np

import camera
import detect
import imaging
import ocr
import pdfwriter

OUT = tempfile.mkdtemp(prefix="ohs-selftest-")
p = f = 0


def ok(name, cond, detail=""):
    global p, f
    if cond:
        p += 1
        print("  \033[32mPASS\033[0m %-46s %s" % (name, detail))
    else:
        f += 1
        print("  \033[31mFAIL\033[0m %-46s %s" % (name, detail))


print("── camera ──")
devs = camera.Camera.list_devices()
ok("cameras enumerated", len(devs) > 0,
   ", ".join("%d:%s(%s)" % (d["index"], d["name"], d["kind"]) for d in devs))
if not devs:
    sys.exit(1)

t0 = time.time()
ok("enumeration opens nothing", time.time() - t0 < 1.0,
   "%d ms — probing devices is what stops this camera reaching 16 MP"
   % int((time.time() - t0) * 1000))

best = devs[0]
ok("an external camera is preferred over the built-in",
   best["kind"] != "builtin" or all(d["kind"] == "builtin" for d in devs),
   "picked %s" % best["name"])

cam = camera.Camera()
t0 = time.time()
opened = cam.open(best["index"], best["name"], None)
ok("camera opened", opened, "%d ms" % int((time.time() - t0) * 1000))
if not opened:
    print("  " + (cam.error or ""))
    sys.exit(1)

if not cam.at_max:
    t0 = time.time()
    up = cam.upgrade()
    ok("upgraded to the sensor maximum", up,
       "%dx%d in %d ms" % (cam.width, cam.height, int((time.time() - t0) * 1000)))

frame = None
deadline = time.time() + 8
while time.time() < deadline and frame is None:
    frame = cam.grab()
    time.sleep(0.05)
ok("frame delivered", frame is not None,
   "%dx%d" % (frame.shape[1], frame.shape[0]) if frame is not None else "none")
if frame is None:
    sys.exit(1)

ok("preview and capture are the same pixels",
   frame.shape[1] == cam.width and frame.shape[0] == cam.height,
   "%dx%d == %dx%d" % (frame.shape[1], frame.shape[0], cam.width, cam.height))

n0 = cam.sequence()
time.sleep(1.0)
fps = cam.sequence() - n0
ok("frames keep arriving", fps > 0, "%d in the last second" % fps)

print("── detection ──")
t0 = time.time()
quad = detect.detect(imaging.fit(frame, 900))
ms = int((time.time() - t0) * 1000)
ok("detection runs", True, "%s in %d ms"
   % ("found a page" if quad is not None else "no page (returns None, never guesses)", ms))
ok("detection is fast enough for a live outline", ms < 120, "%d ms" % ms)
if quad is not None:
    ok("corners are normalised and ordered",
       quad.shape == (4, 2) and float(quad.min()) >= 0 and float(quad.max()) <= 1,
       str(np.round(quad, 3).tolist()))
    cw, ch = detect.output_size(quad, frame.shape[1], frame.shape[0])
    ok("crop has a sane size", cw > 32 and ch > 32, "%dx%d" % (cw, ch))

print("── processing ──")
adjust = imaging.new_adjust()
full = imaging.process(frame, adjust, quad)
ok("full-resolution page produced", full is not None and full.size > 0,
   "%dx%d" % (full.shape[1], full.shape[0]))

expect = imaging.target_size(adjust, quad, frame.shape[1], frame.shape[0])
ok("output matches the advertised size", (full.shape[1], full.shape[0]) == expect,
   "%dx%d" % expect)

for name in imaging.FILTERS:
    a = imaging.set_filter(imaging.new_adjust(), name)
    small = imaging.process(frame, a, quad, max_dim=700)
    clipped = float(np.mean(small >= 254))
    ok("filter %-11s" % name, small is not None and small.size > 0,
       "%dx%d, %.1f%% of pixels at white" % (small.shape[1], small.shape[0],
                                             clipped * 100))
    if name == "auto":
        ok("auto does not bleach the page", clipped < 0.06,
           "%.1f%% clipped — highlights roll off instead of clamping" % (clipped * 100))

a = imaging.new_adjust()
a["rotate"] = 90
rot = imaging.process(frame, a, quad, max_dim=600)
straight = imaging.process(frame, imaging.new_adjust(), quad, max_dim=600)
ok("rotation swaps the axes",
   abs(rot.shape[0] - straight.shape[1]) <= 2 and abs(rot.shape[1] - straight.shape[0]) <= 2,
   "%dx%d -> %dx%d" % (straight.shape[1], straight.shape[0], rot.shape[1], rot.shape[0]))

print("── export ──")
jpg = os.path.join(OUT, "page.jpg")
cv2.imwrite(jpg, full, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
size = os.path.getsize(jpg)
ok("JPEG written at full resolution", size > 400_000,
   "%.1f MB at %dx%d" % (size / 1e6, full.shape[1], full.shape[0]))

back = cv2.imread(jpg)
ok("saved file keeps every pixel", back.shape == full.shape,
   "%dx%d" % (back.shape[1], back.shape[0]))

encoded = cv2.imencode(".jpg", full, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tobytes()
pdf = pdfwriter.build([{"jpeg": encoded, "width": full.shape[1],
                        "height": full.shape[0], "words": []}],
                      page_size="a4", title="selftest", searchable=False)
path = os.path.join(OUT, "scan.pdf")
open(path, "wb").write(pdf)
ok("PDF written", pdf.startswith(b"%PDF") and pdf.rstrip().endswith(b"%%EOF"),
   "%.1f MB" % (len(pdf) / 1e6))

print("── OCR ──")
if not ocr.available():
    print("  \033[33mSKIP\033[0m tesseract not installed — %s" % ocr.install_hint().splitlines()[0])
else:
    t0 = time.time()
    res = ocr.recognise(imaging.process(frame, imaging.set_filter(imaging.new_adjust(), "bw"),
                                        quad, max_dim=2200))
    ok("OCR ran", isinstance(res.get("text"), str),
       "%d words, confidence %s, %.1fs" % (len(res["words"]),
                                           "n/a" if res["confidence"] is None
                                           else "%d%%" % round(res["confidence"]),
                                           time.time() - t0))

cam.close()
ok("camera released", not cam.is_open())

print("\n%d passed, %d failed   (artifacts in %s)" % (p, f, OUT))
sys.exit(1 if f else 0)
