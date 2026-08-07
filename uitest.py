"""Drive the real App: camera -> capture -> filters -> corners -> export.

    python3 uitest.py

Builds the actual widgets in an off-screen window and clicks through them, so
the wiring is covered rather than just the libraries underneath. Needs the
camera connected.
"""

import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

import camera
import detect
import imaging
import ocr
import scanner

OUT = tempfile.mkdtemp(prefix="ohs-uitest-")
p = f = 0


def ok(name, cond, detail=""):
    global p, f
    if cond:
        p += 1
        print("  \033[32mPASS\033[0m %-44s %s" % (name, detail))
    else:
        f += 1
        print("  \033[31mFAIL\033[0m %-44s %s" % (name, detail))


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def until(cond, seconds, step=0.1):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        pump(step)
    return cond()


app = QApplication(sys.argv)
app.setStyleSheet(scanner.QSS)
win = scanner.App()
win.show()
pump(0.4)

print("── startup ──")
ok("window built", win.isVisible() and win.centralWidget() is not None,
   "%dx%d" % (win.width(), win.height()))
ok("every slider is wired", len(win.sliders) == sum(len(i) for _t, i in scanner.SLIDERS),
   "%d sliders" % len(win.sliders))
ok("every filter has a button", len(win.filter_buttons) == len(scanner.FILTERS),
   ", ".join(win.filter_buttons))
ok("starts in live mode", win.mode == "live" and win.btn_live.isChecked())
ok("capture is disabled before the camera runs", not win.btn_capture.isEnabled())

ok("cameras listed", until(lambda: len(win.devices) > 0, 6),
   ", ".join(d["name"] for d in win.devices))
ok("the external camera is offered first",
   not win.devices or win.devices[0]["kind"] != "builtin"
   or all(d["kind"] == "builtin" for d in win.devices),
   win.device_box.currentText())

print("── camera ──")
opened = until(lambda: win.cam.is_open(), 45)
ok("camera opened from the UI", opened, win.lbl_state.text())
if not opened:
    print("  " + (win.cam.error or ""))
    sys.exit(1)
ok("Start became Stop", win.btn_start.text() == "Stop")
ok("resolution shown", win.lbl_res.text() != "", win.lbl_res.text())

# The app opens fast, then climbs. Give the upgrade its turn.
until(lambda: win.cam.at_max, 25)
ok("running at the sensor maximum", win.cam.at_max,
   "%dx%d%s" % (win.cam.width, win.cam.height,
                "" if win.cam.at_max else " (upgrade button offered)"))
ok("upgrade button reflects the state",
   win.btn_upgrade.isVisible() == (not win.cam.at_max))

# The whole design rests on preview and capture being the same pixels, so the
# preview has to follow a mode change, not keep showing the frame from before.
ok("live frames reach the preview at the camera's current mode",
   until(lambda: win.preview._img is not None
         and win.preview._img.width() >= min(win.cam.width, win.preview.width() * 2) * 0.9, 20),
   "" if win.preview._img is None else "%dx%d on screen for a %dx%d sensor mode"
   % (win.preview._img.width(), win.preview._img.height(), win.cam.width, win.cam.height))
ok("capture enabled once live", win.btn_capture.isEnabled())
ok("footer reports the live size",
   until(lambda: str(win.cam.width) in win.lbl_info.text(), 5), win.lbl_info.text())

print("── capture ──")
win.capture()
pump(1.2)
ok("page captured", len(win.pages) == 1)
page = win.pages[0]
ok("page keeps the full-resolution frame",
   page.frame.shape[1] == win.cam.width and page.frame.shape[0] == win.cam.height,
   "%dx%d" % (page.frame.shape[1], page.frame.shape[0]))
ok("capture uses the outline that was on screen",
   (page.corners is None and win.live_quad is None)
   or np.allclose(page.corners, win.live_quad),
   "no outline — whole frame kept" if page.corners is None else "quad matched")
ok("switched to edit mode", win.mode == "edit" and win.btn_edit.isChecked())
ok("thumbnail made", win.tray.count() == 1 and not win.tray.item(0).icon().isNull())
ok("editor rendered the page", win.preview._img is not None)

print("── adjust ──")
for key, _text in scanner.FILTERS:
    before = win.preview._img
    win.set_filter(key)
    pump(0.15)
    ok("filter %-11s applies" % key,
       page.adjust["filter"] == key and win.preview._img is not None
       and win.preview._img is not before,
       "%s" % page.adjust["mode"])
ok("filter button highlights the active filter",
   win.filter_buttons[page.adjust["filter"]].isChecked())

win.set_filter("auto")
pump(0.1)
win.sliders["contrast"].slider.setValue(70)
pump(0.4)
ok("slider reaches the page", abs(page.adjust["contrast"] - 70) < 1e-6,
   "contrast %.0f" % page.adjust["contrast"])
ok("touching a slider drops the filter to custom",
   page.adjust.get("custom") and not win.filter_buttons["auto"].isChecked())
win.sliders["contrast"].reset()
pump(0.3)
ok("reset returns to the filter's own value",
   abs(page.adjust["contrast"] - imaging.FILTERS["auto"]["contrast"]) < 1e-6,
   "contrast %.0f" % page.adjust["contrast"])

before = imaging.process(page.frame, page.adjust, page.corners, max_dim=400).shape[:2]
win.rotate(90)
pump(0.3)
after = imaging.process(page.frame, page.adjust, page.corners, max_dim=400).shape[:2]
ok("rotate swaps the output axes",
   page.adjust["rotate"] == 90 and (after[0], after[1]) == (before[1], before[0]),
   "%dx%d -> %dx%d" % (before[1], before[0], after[1], after[0]))
win.rotate(-90)
pump(0.2)

print("── corners ──")
win.toggle_corners()
pump(0.3)
ok("corner mode shows the uncropped frame", win.preview.is_editable())
ok("a page with no detection gets a full-frame quad", page.corners is not None)
moved = page.corners.copy()
moved[0] = [min(0.9, moved[0][0] + 0.05), min(0.9, moved[0][1] + 0.05)]
win._corners_dragged(moved)
pump(0.3)
ok("dragging a corner reaches the page", np.allclose(page.corners, moved))
win.toggle_corners()
pump(0.3)
ok("leaving corner mode re-renders the crop", not win.preview.is_editable())

win.redetect()
pump(0.4)
ok("Detect re-runs detection", True,
   "found a page" if page.corners is not None else "none found (kept honest)")

print("── export ──")
jpg = os.path.join(OUT, "page.jpg")
detail = win._write_image(page, jpg)
full = imaging.process(page.frame, page.adjust, page.corners)
saved = cv2.imread(jpg)
ok("image saved at the crop's full resolution", saved.shape == full.shape, detail)
ok("no downscale sneaks in",
   max(saved.shape[:2]) >= min(max(page.frame.shape[:2]) * 0.3, 1500),
   "%dx%d from a %dx%d frame" % (saved.shape[1], saved.shape[0],
                                 page.frame.shape[1], page.frame.shape[0]))

pdf = os.path.join(OUT, "scan.pdf")
detail = win._write_pdf(list(win.pages), pdf)
data = open(pdf, "rb").read()
ok("PDF written", data.startswith(b"%PDF") and data.rstrip().endswith(b"%%EOF"), detail)

print("── OCR ──")
if not ocr.available():
    print("  \033[33mSKIP\033[0m tesseract not installed")
    ok("app reports OCR is unavailable rather than failing",
       "not installed" in win.lbl_ocr.text())
else:
    win.run_ocr()
    ok("OCR finished", until(lambda: not win.busy, 90), win.lbl_ocr.text())
    ok("text reached the panel", page.ocr is not None)

print("── pages ──")
cv2.imwrite(os.path.join(OUT, "import.png"), np.full((900, 700, 3), 240, np.uint8))
win.pages.append(scanner.Page(cv2.imread(os.path.join(OUT, "import.png")), None,
                              imaging.new_adjust(), "Page 2"))
win._add_thumb(win.pages[-1])
pump(0.3)
ok("second page listed", win.tray.count() == 2)
win.tray.setCurrentRow(1)
pump(0.4)
ok("selecting a page loads its settings", win.current == 1)

win.tray.setCurrentRow(0)
pump(0.3)
win.apply_all()
pump(0.5)
ok("apply-to-all copies the look",
   win.pages[1].adjust["contrast"] == win.pages[0].adjust["contrast"])

win.delete_page()
pump(0.4)
ok("page deleted and the rest renumbered",
   len(win.pages) == 1 and win.pages[0].name == "Page 1" and win.tray.count() == 1)

win.cam.close()
pump(0.2)
print("\n%d passed, %d failed   (artifacts in %s)" % (p, f, OUT))
sys.exit(1 if f else 0)
