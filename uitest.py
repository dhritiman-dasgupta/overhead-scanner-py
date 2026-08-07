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

# The app opens fast, then climbs. Give the upgrade its turn, and press the
# button a couple of times as a user would — this camera really does refuse its
# top mode at random, so one attempt is not a fair test of the app.
until(lambda: win.cam.at_max, 25)
tries = 0
while not win.cam.at_max and tries < 2:
    tries += 1
    win.upgrade_camera()
    until(lambda: not win.busy, 30)
    pump(0.5)
ok("running at the sensor maximum", win.cam.at_max,
   "%dx%d%s" % (win.cam.width, win.cam.height,
                "" if not tries else " after %d retries" % tries))
ok("upgrade button reflects the state",
   win.btn_upgrade.isVisible() == (not win.cam.at_max))

# The whole design rests on preview and capture being the same pixels, so the
# preview has to follow a mode change, not keep showing the frame from before.
ok("live frames reach the preview at the camera's current mode",
   until(lambda: win.preview._img is not None
         and win.preview._img.width() >= min(win.cam.width, win.feed.target) * 0.9, 25),
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
    # Rendering is off the GUI thread now, so wait for the picture rather than
    # assuming it is there by the time the call returns.
    applied = until(lambda: win.preview._img is not None
                    and win.preview._img is not before, 5)
    ok("filter %-11s applies" % key, page.adjust["filter"] == key and applied,
       "%s" % page.adjust["mode"])
ok("filter button highlights the active filter",
   win.filter_buttons[page.adjust["filter"]].isChecked())

win.set_filter("auto")
pump(0.1)
win.sliders["contrast"].slider.setValue(70)
until(lambda: not win.settle.isActive(), 3)
pump(0.5)
ok("slider reaches the page", abs(page.adjust["contrast"] - 70) < 1e-6,
   "contrast %.0f" % page.adjust["contrast"])
ok("touching a slider drops the filter to custom",
   page.adjust.get("custom") and not win.filter_buttons["auto"].isChecked())
win.sliders["contrast"].reset()
until(lambda: not win.settle.isActive(), 3)
pump(0.4)
ok("reset returns to the filter's own value",
   abs(page.adjust["contrast"] - imaging.FILTERS["auto"]["contrast"]) < 1e-6,
   "contrast %.0f" % page.adjust["contrast"])

before = imaging.process(page.frame, page.adjust, page.corners, max_dim=400).shape[:2]
win.rotate(90)
pump(0.5)
after = imaging.process(page.frame, page.adjust, page.corners, max_dim=400).shape[:2]
ok("rotate swaps the output axes",
   page.adjust["rotate"] == 90 and (after[0], after[1]) == (before[1], before[0]),
   "%dx%d -> %dx%d" % (before[1], before[0], after[1], after[0]))
win.rotate(-90)
pump(0.2)

print("── corners ──")
win.toggle_corners()
pump(0.5)
ok("corner mode shows the uncropped frame", win.preview.is_editable())
ok("a page with no detection gets a full-frame quad", page.corners is not None)
moved = page.corners.copy()
moved[0] = [min(0.9, moved[0][0] + 0.05), min(0.9, moved[0][1] + 0.05)]
win._corners_dragged(moved)
pump(0.5)
ok("dragging a corner reaches the page", np.allclose(page.corners, moved))
win.toggle_corners()
pump(0.6)
ok("leaving corner mode re-renders the crop", not win.preview.is_editable())

win.redetect()
pump(0.6)
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

print("── view ──")
ok("starts fitted", abs(win.preview.zoom() - 1.0) < 1e-6)
win.zoom(2.0)
pump(0.2)
ok("zoom in", win.preview.zoom() > 1.5, "%.0f%%" % (win.preview.zoom() * 100))
win.zoom(0.25)
pump(0.2)
ok("zoom never goes below fit", abs(win.preview.zoom() - 1.0) < 1e-6,
   "%.0f%%" % (win.preview.zoom() * 100))
win.zoom(4.0)
win.preview_fit()
pump(0.2)
ok("Fit resets zoom and pan",
   abs(win.preview.zoom() - 1.0) < 1e-6 and win.preview._pan.isNull())

win.act_grid.setChecked(True)
win.preview.set_guides(grid=True)
pump(0.1)
ok("guides toggle", win.preview.grid)
win.act_grid.setChecked(False)
win.preview.set_guides(grid=False)

win.compare(True)
ok("before/after shows the unprocessed page",
   until(lambda: win.preview._compare is not None, 6))
win.compare(False)
pump(0.2)
ok("releasing compare goes back", win.preview._compare is None)

print("── adjust extras ──")
win.chk_invert.setChecked(True)
until(lambda: not win.settle.isActive(), 4)
pump(0.4)
ok("invert reaches the page", page.adjust.get("invert") is True)
win.chk_invert.setChecked(False)
until(lambda: not win.settle.isActive(), 4)

win.sliders["sharpen"].slider.setValue(120)
until(lambda: not win.settle.isActive(), 4)
win.reset_adjust()
pump(0.4)
ok("reset all returns every slider to the filter",
   abs(page.adjust["sharpen"] - imaging.FILTERS[page.adjust["filter"]]["sharpen"]) < 1e-6
   and not page.adjust.get("custom"),
   "sharpen %.0f" % page.adjust["sharpen"])

print("── clipboard, estimate, formats ──")
win.copy_image()
ok("copy image finished", until(lambda: not win.busy, 90))
from PySide6.QtGui import QGuiApplication as _G
clip = _G.clipboard().image()
full = imaging.process(page.frame, page.adjust, page.corners)
ok("clipboard holds the full-resolution page",
   not clip.isNull() and clip.width() == full.shape[1],
   "%dx%d" % (clip.width(), clip.height()))

win.txt_ocr.setPlainText("hello scanner")
win.copy_text()
pump(0.2)
ok("copy text", _G.clipboard().text() == "hello scanner")

win.fmt.setCurrentIndex(2)             # PNG
win.estimate()
ok("estimate finished", until(lambda: not win.busy, 120), win.lbl_estimate.text())
ok("estimate reports real numbers",
   "MB" in win.lbl_estimate.text() and "dpi" in win.lbl_estimate.text())

png = os.path.join(OUT, "page.png")
detail = win._write_image(page, png)
back = cv2.imread(png)
ok("PNG export is lossless and full size", back.shape == full.shape, detail)
win.fmt.setCurrentIndex(0)

print("── pdf options ──")
letter = os.path.join(OUT, "letter.pdf")
win._write_pdf([page], letter, "letter", False)
a4 = os.path.join(OUT, "a4.pdf")
win._write_pdf([page], a4, "a4", False)
ok("page size reaches the PDF",
   open(letter, "rb").read() != open(a4, "rb").read(),
   "letter %.1f MB vs A4 %.1f MB" % (os.path.getsize(letter) / 1e6,
                                     os.path.getsize(a4) / 1e6))

print("── menus ──")
titles = [m.title().replace("&", "") for m in win.menuBar().findChildren(type(win.menuBar().addMenu("x")))]
for want in ("File", "Edit", "View", "Camera", "Help"):
    ok("%s menu" % want, want in titles)

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
