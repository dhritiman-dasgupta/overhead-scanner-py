# Overhead Scanner (Python)

A desktop app for an overhead document camera: capture at the sensor's full
resolution, deskew, clean up, OCR, and export to image or searchable PDF.
Everything runs locally.

```bash
python3 scanner.py
```

Needs `opencv-python`, `numpy` and `pillow` — all already present on this
machine. Tkinter ships with Python.

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
| **OpenCV (this app)** | **4656×3496** | direct, ~5–10 fps, and it *is* the preview |

So there is no preview mode and no capture mode here. The camera runs at full
resolution the whole time and a capture is the frame you were already looking
at. Preview and capture cannot disagree about resolution, field of view or what
the page looked like, because they are the same pixels — which removes an
entire class of bug the browser version kept running into.

---

## Using it

1. Pick the camera. The list shows each device's **real maximum**, probed by
   asking for it, and the highest-resolution one is selected first. On a laptop
   the document camera is rarely index 0, and choosing by index quietly hands
   you 1080p from the built-in webcam.
2. **Start camera**, put a page down. A green outline shows the page it found.
3. **Capture**, or tick **Auto** to shoot whenever the scene goes still — the
   page-turn workflow.
4. Pick a filter, adjust if needed, then **Save image** or **Save PDF**.

The footer reads `live 4656×3496  crop → 1632×2155  ≈184 dpi`. That dpi figure
is the one to watch: it is what an A4 sheet would come out at, and it tells you
whether small print will survive before you scan a stack.

### Filters

| | |
|---|---|
| **Auto** | The default. Start here. |
| **Black & white** | Text-only pages. Smallest files, best OCR. |
| **Colour doc** | Photos, highlighter, coloured print. |
| **Whiteboard** | Hard background removal, punchy pens. |
| **Ink boost** | Pencil, faded carbon, faint print. |
| **Photo** | Barely touches the image. |

Every filter is a preset over the sliders below it; nudge anything and it
switches to *custom*.

### Getting a sharp scan

- **Fill the frame with the page.** This is the single biggest lever and the
  easiest to overlook. A crop is only as sharp as the sensor pixels that landed
  on it: 16 MP of a page covering a third of the frame yields a ~1400 px scan,
  and nothing downstream puts detail back. Capture warns when the page covers
  less than about half the frame.
- **A desk that isn't paper-coloured** makes detection close to perfect.
- If the page isn't found the footer says so and the whole frame is kept — it
  never guesses. Use **Corners** and drag the four handles.

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
camera.py      device discovery and full-resolution capture
detect.py      page detection, homography, perspective warp
imaging.py     the processing pipeline
ocr.py         Tesseract wrapper, degrades cleanly when absent
pdfwriter.py   PDF writer with an invisible OCR text layer
selftest.py    end-to-end check against a real camera
uitest.py      drives the real UI headlessly
```

### Two things worth knowing before editing `imaging.py`

**Nothing clips.** Highlights roll off above a knee rather than being clamped
at 255 — in the divide, in the tone curve and in the sharpener. Each was
independently capable of turning printed detail on a bright label into blank
white. An unsharp mask *deliberately* overshoots at an edge; clamping that
overshoot is how sharpening destroys the detail it was meant to reveal.

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
python3 selftest.py     # camera -> capture -> detect -> filters -> PDF
python3 uitest.py       # the real App class, headless
```

Both need the camera connected. `uitest.py` builds real Tk widgets in a
withdrawn window and drives capture, filters, sliders, corner dragging and
export, so the UI wiring is covered rather than just the libraries under it.

---

## Known limits

- **~5–10 fps preview at 16 MP.** That is the camera's own frame rate in that
  mode, not the app's. Fine for documents; it is not a video app.
- **Device indices shift** when cameras are plugged in or out, and a camera
  released moments earlier can briefly report a lower maximum. Enumeration
  re-probes each time, so use the dropdown rather than assuming index 0.
- **No auto-orientation.** A page placed upside down is captured upside down;
  the rotate buttons fix it. Detecting text orientation needs OCR.
- **Curved book spines aren't dewarped** — a quadrilateral can't model a curl.
