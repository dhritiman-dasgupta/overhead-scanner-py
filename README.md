# Overhead Scanner (Python)

> **Related:** [`overhead-scanner`](https://github.com/dhritiman-dasgupta/overhead-scanner) — browser sibling of this project, the same detection approach in a web app.


A desktop app for an overhead document camera: capture at the sensor's full
resolution, deskew, clean up, OCR, and export to image or searchable PDF.
Everything runs locally.

```bash
pip3 install --user PySide6            # once
python3 scanner.py
```

Also needs `opencv-python`, `numpy` and `pillow`.

Optionally build a real macOS app — own Dock icon, and the camera permission
attributed to *Overhead Scanner* rather than to the interpreter:

```bash
./make-app.sh && open "Overhead Scanner.app"
```

It wraps the checked-out source rather than copying it, so there is nothing to
rebuild after an edit.

---

## Why this exists rather than the browser version

The browser build hit a hardware wall that no amount of JavaScript could get
past, and this is the measured record of it on the camera it was built for
(a 4656×3496 UVC document camera):

| route | resolution | notes |
|---|---|---|
| Chrome video stream | 1598×1200 | the 4656×3496 mode takes **66 s** to a first frame — unusable as a preview |
| Chrome `takePhoto()` | 4656×3496 | works, but ~3.3 s per shot, and its field of view differs from the preview |
| macOS AVFoundation video | 1920×1080 | capped by the session preset whatever `activeFormat` says; `inputPriority` is unavailable on macOS |
| **OpenCV (this app)** | **4656×3496** | direct, ~8–10 fps, and it *is* the preview |

So there is no preview mode and no capture mode here. The camera runs at full
resolution the whole time and a capture is the frame you were already looking
at. Preview and capture cannot disagree about resolution, field of view or what
the page looked like, because they are the same pixels — which removes an
entire class of bug the browser version kept running into.

### And why Qt rather than Tkinter

The only Python on this machine is Apple's, which ships **Tk 8.5.9**. On macOS
26 that Tk does not repaint: the window comes up white, `tk.Button` ignores
every colour it is given, and the camera preview never appears. This is a
platform fault with no application-level fix, so the UI is PySide6, which draws
everything itself and looks the same on macOS, Windows and Linux.

---

## Using it

1. The **USB camera is chosen for you**. Devices are named through
   AVFoundation and ranked external → built-in → iPhone, because an overhead
   scanner is an external camera and index 0 is usually the laptop webcam
   pointing at your face.
2. The preview appears in about two seconds at whatever the camera streams by
   default, then climbs to the full sensor mode a moment later. The badge in
   the top bar turns green at the maximum, amber below it, and **Try 16 MP**
   asks again.
3. Put a page down. A green outline shows the page it found.
4. **Capture** (Space), or tick **Auto** to shoot whenever the scene goes
   still — the page-turn workflow.
5. Pick a filter, adjust if needed, then **Save image** (⌘S) or **Save PDF**
   (⌘P).

The footer reads `live 4656×3496 · crop 4623×3687 · ≈403 dpi`. That dpi figure
is the one to watch: it is what an A4 sheet would come out at, and it tells you
whether small print will survive before you scan a stack.

### Keys

`Space` capture · `C` corners · `D` detect · `A` auto · `B` hold to compare ·
`[` `]` rotate · `←` `→` page · `⌫` delete · `⌘=` `⌘-` `⌘0` zoom ·
`⌘S` image · `⌘P` PDF · `⌘R` start/stop · `⌘/` shortcuts

Scroll to zoom about the pointer, drag to pan, double-click to fit.

### Checking a scan before you commit to a stack

- **Zoom in.** The reason to shoot at 16 MP is detail, and the only way to know
  it survived is to look at it. Above 250% the view stops smoothing and shows
  you the actual pixels.
- **Hold `B`.** Side by side with the unprocessed page, in one keypress. It is
  the fastest way to tell a filter that is helping from one that is inventing.
- **Watch the dpi in the footer**, and **Estimate** on the Export tab, which
  renders at full resolution and reports the real file size.

### Filters

| | |
|---|---|
| **Auto** | The default. Start here. |
| **Black & white** | Text-only pages. Smallest files, best OCR. |
| **Colour doc** | Photos, highlighter, coloured print. |
| **Whiteboard** | Hard background removal, punchy pens. |
| **Ink boost** | Pencil, faded carbon, faint print. |
| **Photo** | Barely touches the image. |

Every filter is a preset over the sliders below it; nudge anything and the
value turns blue and the filter drops to *custom*.

### Getting a sharp scan

- **Fill the frame with the page.** This is the single biggest lever and the
  easiest to overlook. A crop is only as sharp as the sensor pixels that landed
  on it: 16 MP of a page covering a third of the frame yields a ~1400 px scan,
  and nothing downstream puts detail back. The footer shows what percentage of
  the frame the page fills and warns below about a third.
- **A desk that isn't paper-coloured** makes detection close to perfect.
- If the page isn't found the footer says so and the whole frame is kept — it
  never guesses. Press **Corners** and drag the four handles.

---

## OCR

Uses the Tesseract command-line binary, found automatically if installed:

```bash
brew install tesseract
```

Without it the OCR panel says so plainly and the rest of the app is unaffected.
Recognition runs on the *processed* page, so a good Black & white or Auto
filter lifts accuracy more than any OCR setting. Word boxes feed the searchable
PDF's invisible text layer.

---

## Layout

```
scanner.py     the app: window, preview loop, pages, export
qtui.py        theme, custom widgets, the preview/corner-drag canvas
camera.py      device discovery and full-resolution capture
detect.py      page detection, homography, perspective warp
imaging.py     the processing pipeline
ocr.py         Tesseract wrapper, degrades cleanly when absent
pdfwriter.py   PDF writer with an invisible OCR text layer
selftest.py    engine, end to end, against a real camera
uitest.py      the real App class, driven off-screen
stresstest.py  long run at full resolution, watching for faults and leaks
make-app.sh    wraps the source in a macOS .app bundle
make-icon.py   draws the icon (called by make-app.sh)
tools_listcams.swift   index → camera name; compiled to ./listcams on first run
```

### Three things worth knowing before editing `camera.py`

**Never probe a camera you are about to use.** Opening and releasing a device
is all a resolution probe does, and this camera then refuses its 16 MP mode for
several seconds afterwards. An earlier version enumerated devices by probing
each one, and that probe was itself the reason the camera that had just
reported 4656×3496 handed back 1920×1080 when the app opened it. Enumeration
now reads names from AVFoundation and opens nothing.

**One mode request per session.** A mode change this camera declines does not
merely fail, it wedges the session — nothing comes back afterwards, not even
the mode that was working a second earlier. The version that walked a ladder of
six resolutions looking for a smaller one turned a recoverable stumble into a
camera that returned nothing at all.

**Let it settle before asking.** Measured: requesting 16 MP 0.6 s after opening
fails about half the time; at 1.2 s the same request is answered in 0.1 s.

### Two things worth knowing before editing `scanner.py`

**Nothing slow may run on the GUI thread.** Scaling one 16 MP frame for display
is 27 ms and detection another 28 ms; doing that in a timer callback at ten
frames a second consumed nearly half the GUI thread in 45 ms lumps, which is
precisely what a stuttering button feels like. `LiveFeed` does the scaling and
detection on its own thread and hands over a contiguous RGB buffer that the GUI
wraps without copying. `Renderer` does the same for the editor, coalescing
requests so a slider drag produces one render per completed frame rather than a
backlog. Measured, before and after:

| | before | after |
|---|---|---|
| GUI stall, live preview (p95 / worst) | 43.9 / 57.1 ms | **3.8 / 15.0 ms** |
| GUI stall, slider drag (p95 / worst) | 267.9 / 286.4 ms | **1.1 / 3.3 ms** |

**Never touch the VideoCapture from two threads.** Releasing or reconfiguring
an AVFoundation session while another thread is inside `read()` is a native
crash, not an exception — and at 16 MP a read takes ~100 ms, or seconds on a
misbehaving camera, so the reader is very often mid-read exactly when the app
wants to stop it. Every call into the capture object is serialised by
`Camera._cap_lock`, and `close()` would rather leak a capture object than
release one underneath a live read.

### Two things worth knowing before editing `imaging.py`

**Nothing clips.** Highlights roll off above a knee rather than being clamped
at 255 — in the divide, in the tone curve and in the sharpener. Each was
independently capable of turning printed detail on a bright label into blank
white. An unsharp mask *deliberately* overshoots at an edge; clamping that
overshoot is how sharpening destroys the detail it was meant to reveal.

**Every stage works in place.** At 16 MP one float32 copy of a page is 207 MB,
and a pipeline where each stage allocated its own output peaked at 2.08 GB for
a single export. It is ~1.5 GB now, and the output is unchanged to better than
40 dB PSNR — invisible, and checked by `selftest.py`.

**The illumination estimate takes the larger of two readings** — a heavily
blurred one, which passes a lamp's falloff through and ignores content, and the
barely-smoothed block maxima, which is what paper actually reads *there*. The
blur alone lets whatever surrounds the paper drag the estimate down: a page that
doesn't fill the frame has dark mat on the other side of its edge, so the gain
climbs and lifts the ink along with the paper, bleaching text near the edges.

### Detection

Otsu threshold, dilate to close the gaps that lines of text cut through the
paper, largest connected component, convex hull to four corners. **Both
polarities are tried** — the page-is-brighter guess is wrong often enough in
ordinary use, and guessing wrong meant no outline at all. The region must
**fill its own outline** (ink covers a fraction of its bounding box, paper
nearly all of it), which stops it cropping to the text block. It returns
nothing rather than guessing.

---

## Tests

```bash
python3 selftest.py     # 28 assertions: camera -> capture -> detect -> filters -> PDF
python3 uitest.py       # 68 assertions: the real App class, off-screen
python3 stresstest.py   # hammers the app at full resolution and watches memory
```

All three need the camera connected, and none should be run while the app is
open — they compete for the device. `uitest.py` builds real Qt widgets in an
off-screen window and drives capture, filters, sliders, corner dragging, zoom,
compare, clipboard, export and page management, so the UI wiring is covered
rather than just the libraries under it.

`stresstest.py` exists because of crashes that leave no traceback. It forces a
real repaint after every buffer swap — off-screen, `processEvents` never runs
`paintEvent`, so a memory fault sails straight through and the test proves
nothing.

If the app does die, it now leaves a record: `faulthandler` is armed against
`~/Library/Logs/overhead-scanner.log`, so even a segmentation fault inside
OpenCV or Qt writes a Python stack there.

---

## Known limits

- **~8–10 fps preview at 16 MP.** That is the camera's own frame rate in that
  mode, not the app's. Fine for documents; it is not a video app.
- **The camera is genuinely erratic about its top mode.** It will refuse 16 MP
  for a few seconds after anything else has had it — including the app's own
  previous run. The app opens low, upgrades, and offers **Try 16 MP**; if that
  fails twice, unplug it and back in.
- **Device indices shift** when cameras are plugged in or out, so the list is
  re-read on every Rescan rather than remembered.
- **No auto-orientation.** A page placed upside down is captured upside down;
  the rotate buttons fix it. Detecting text orientation needs OCR.
- **Curved book spines aren't dewarped** — a quadrilateral can't model a curl.
- **The macOS menu bar says "Python".** The bundle is correct everywhere else —
  window title, Dock, and the camera permission — but the app menu takes its
  name from the running executable's own bundle, and that is the system
  interpreter. Fixing it means shipping a private Python (py2app or
  PyInstaller) and a ~200 MB bundle.
- **A full-resolution export peaks around 1.5 GB** and takes ~2 s. Fine on this
  machine; worth knowing before scanning a hundred pages on a smaller one.
