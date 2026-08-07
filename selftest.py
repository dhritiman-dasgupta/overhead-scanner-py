"""End-to-end check against the real camera: capture, detect, process, export."""
import time, os, sys
import cv2, numpy as np
import camera, detect, imaging, pdfwriter

OUT = "/private/tmp/claude-501/-Users-apple/f74ecf1f-4a16-4b31-bb2e-8c95c0466309/scratchpad"
p, f = 0, 0
def ok(name, cond, detail=""):
    global p, f
    if cond: p += 1; print("  \033[32mPASS\033[0m %-46s %s" % (name, detail))
    else:    f += 1; print("  \033[31mFAIL\033[0m %-46s %s" % (name, detail))

devs = camera.Camera.list_devices()
ok("camera enumerated", len(devs) > 0, str(devs))
if not devs: sys.exit(1)

cam = camera.Camera()
t0 = time.time()
opened = cam.open(devs[0]["index"])
ok("camera opened", opened, "%d ms" % int((time.time()-t0)*1000))
if not opened: print(cam.error); sys.exit(1)
ok("running at the sensor maximum", (cam.width, cam.height) == (4656, 3496),
   "%dx%d @ %.0f fps" % (cam.width, cam.height, cam.fps))

time.sleep(2.0)
t0 = time.time(); frame = cam.grab(); grab_ms = int((time.time()-t0)*1000)
ok("frame captured", frame is not None,
   "" if frame is None else "%dx%d in %d ms" % (frame.shape[1], frame.shape[0], grab_ms))
cam.close()
if frame is None: sys.exit(1)
ok("capture is full resolution", frame.shape[:2] == (3496, 4656), str(frame.shape))
cv2.imwrite(OUT + "/py-raw.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

t0 = time.time(); quad = detect.detect(frame); det_ms = int((time.time()-t0)*1000)
ok("detection ran", True, ("found" if quad is not None else "declined") + "  %d ms" % det_ms)
if quad is not None:
    ok("corners are normalised", float(quad.min()) >= 0 and float(quad.max()) <= 1,
       str(np.round(quad, 3).tolist()))

for name in ("auto", "photo", "bw"):
    a = imaging.set_filter(imaging.new_adjust(), name)
    t0 = time.time()
    out = imaging.process(frame, a, quad, None)
    ms = int((time.time() - t0) * 1000)
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    clip = 100.0 * float((g >= 254).sum()) / g.size
    ok("filter %-6s full res" % name, out.shape[0] > 0,
       "%dx%d  %4d ms  clipped %.2f%%" % (out.shape[1], out.shape[0], ms, clip))
    cv2.imwrite(OUT + "/py-%s.jpg" % name, out, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

a = imaging.set_filter(imaging.new_adjust(), "auto")
full = imaging.process(frame, a, quad, None)
exp = detect.output_size(quad, frame.shape[1], frame.shape[0]) if quad is not None \
      else (frame.shape[1], frame.shape[0])
ok("export matches the crop's own pixels", (full.shape[1], full.shape[0]) == exp,
   "%dx%d" % (full.shape[1], full.shape[0]))

okj, buf = cv2.imencode(".jpg", full, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
ok("jpeg q98 encoded", okj, "%.2f MB" % (len(buf) / 1048576.0))
pdf = pdfwriter.build([{"jpeg": buf.tobytes(), "width": full.shape[1],
                        "height": full.shape[0]}], page_size="a4", title="test")
ok("pdf built", pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF"),
   "%.2f MB" % (len(pdf) / 1048576.0))
i = pdf.rfind(b"startxref")
off = int(pdf[i+10:pdf.find(b"\n", i+10)])
ok("pdf xref resolves", pdf[off:off+4] == b"xref")
open(OUT + "/py-scan.pdf", "wb").write(pdf)

print("\n%s%d passed, %d failed\033[0m" % ("\033[31m" if f else "\033[32m", p, f))
sys.exit(1 if f else 0)
