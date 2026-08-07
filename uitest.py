"""Drive the real App class: camera -> capture -> filters -> crop -> export."""
import os, sys, time, tempfile
import tkinter as tk
import numpy as np, cv2

p = f = 0
def ok(name, cond, detail=""):
    global p, f
    if cond: p += 1; print("  \033[32mPASS\033[0m %-44s %s" % (name, detail))
    else:    f += 1; print("  \033[31mFAIL\033[0m %-44s %s" % (name, detail))

import scanner
root = tk.Tk()
root.withdraw()                       # real widgets, no visible window
app = scanner.App(root)
root.update()

app.refresh_devices()
root.update()
ok("devices listed", len(app.devices) > 0,
   ", ".join("idx%d max %dx%d" % (d["index"], d["max_width"], d["max_height"]) for d in app.devices))
ok("highest-resolution camera is first",
   not app.devices or app.devices[0]["max_width"] >= max(d["max_width"] for d in app.devices),
   "picks idx %d" % app.devices[0]["index"] if app.devices else "")

app.toggle_camera()
for _ in range(80):
    root.update(); time.sleep(0.1)
    if app.cam.is_open(): break
ok("camera opened from the UI", app.cam.is_open(),
   "%dx%d" % (app.cam.width, app.cam.height))
ok("running at full sensor resolution", (app.cam.width, app.cam.height) == (4656, 3496),
   "%dx%d" % (app.cam.width, app.cam.height))

for _ in range(30): root.update(); time.sleep(0.1)     # let preview + detection run
ok("live preview rendered", app._preview_img is not None)
ok("live info line populated", "live" in app.lbl_info.cget("text"),
   app.lbl_info.cget("text"))

app.capture(); root.update()
ok("capture created a page", len(app.pages) == 1)
page = app.cur()
ok("page holds the full-resolution frame", page is not None and page.frame.shape[:2] == (3496, 4656),
   str(None if page is None else page.frame.shape))
ok("mode switched to edit", app.mode == "edit")
ok("edit view rendered", app._preview_img is not None)
ok("thumbnail built", app.pages[0].thumb is not None)

for name in ("bw", "photo", "auto"):
    app.set_filter(name); root.update()
    ok("filter %s applied" % name, app.cur().adjust["filter"] == name,
       app.lbl_info.cget("text")[:52])

before = app.cur().adjust["contrast"]
app.var["contrast"].set(40); app._apply_adjust("contrast", 40.0); root.update()
ok("slider edits the page", app.cur().adjust["contrast"] == 40 and app.cur().adjust["filter"] == "custom",
   "contrast %s -> %s" % (before, app.cur().adjust["contrast"]))

app.rotate(90); root.update()
ok("rotate recorded", app.cur().adjust["rotate"] == 90)
app.rotate(-90); root.update()

app.redetect(); root.update()
ok("re-detect ran", True, app.lbl_hint.cget("text"))

app.toggle_corners(); root.update()
ok("corner mode entered", app.corner_mode)
if app.cur().corners is None: app.cur().corners = scanner.detect.full_frame()
orig = np.array(app.cur().corners, copy=True)
app._drag_corner = 0
app._view = (0, 0, 400, 300)
class E: x, y = 40, 30
app._on_drag(E()); root.update()
ok("dragging moves a corner", not np.allclose(orig[0], app.cur().corners[0]),
   "%s -> %s" % (np.round(orig[0],3).tolist(), np.round(np.array(app.cur().corners[0]),3).tolist()))
app._on_release(E()); root.update()
app.toggle_corners(); root.update()

app.set_filter("auto")
full = app._render_full(app.cur())
w, h = app.cur().frame.shape[1], app.cur().frame.shape[0]
exp = scanner.imaging.target_size(app.cur().adjust, app.cur().corners, w, h)
ok("export renders at full crop resolution", (full.shape[1], full.shape[0]) == exp,
   "%dx%d" % (full.shape[1], full.shape[0]))
ok("export is not downscaled", max(full.shape[:2]) > 1500, "%dx%d" % (full.shape[1], full.shape[0]))

tmp = tempfile.mkdtemp()
jpg = os.path.join(tmp, "out.jpg")
okj, buf = cv2.imencode(".jpg", full, [int(cv2.IMWRITE_JPEG_QUALITY), 98]); buf.tofile(jpg)
ok("jpeg written", os.path.getsize(jpg) > 100000, "%.2f MB" % (os.path.getsize(jpg)/1048576.0))

entries = [{"jpeg": buf.tobytes(), "width": full.shape[1], "height": full.shape[0]}]
data = scanner.pdfwriter.build(entries, page_size="a4", title="uitest")
ok("pdf built", data.startswith(b"%PDF") and data.rstrip().endswith(b"%%EOF"),
   "%.2f MB" % (len(data)/1048576.0))

app.delete_page(); root.update()
ok("page deleted", len(app.pages) == 0 and app.mode == "live")

app.cam.close(); root.destroy()
print("\n%s%d passed, %d failed\033[0m" % ("\033[31m" if f else "\033[32m", p, f))
sys.exit(1 if f else 0)
