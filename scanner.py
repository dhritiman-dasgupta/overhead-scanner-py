#!/usr/bin/env python3
"""Overhead Scanner — desktop app for a document camera.

    python3 scanner.py

The camera runs at its full sensor resolution the whole time and a capture is
the frame you were already looking at. Preview and capture therefore cannot
disagree about resolution or field of view, because they are the same pixels.
"""

import faulthandler
import os
import sys
import threading
import time
import traceback

import cv2
import numpy as np
from PySide6.QtCore import QObject, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QAction, QGuiApplication, QIcon, QKeySequence, QPixmap,
                           QPainter)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QProgressBar, QScrollArea, QTabWidget,
                               QTextEdit, QVBoxLayout, QWidget)

import camera
import detect
import imaging
import ocr
import pdfwriter
import qtui
from qtui import (ACCENT, BAD, BG, FG, FG2, FG3, GOOD, LINE, PANEL, PANEL2, QSS,
                  WARN, PreviewView, Section, SliderRow, Toast, button, hair,
                  label, spacer, to_qimage)

DETECT_EVERY = 0.40          # seconds between live page detections
PREVIEW_MAX  = 2200          # cap on the long edge sent to the screen
EDIT_MAX     = 1700          # long edge a settled editor preview renders at
DRAFT_MAX    = 900           # ...and while a slider is still moving
SETTLE_MS    = 260           # quiet time before the draft is replaced
THUMB        = QSize(112, 140)
A4_INCHES    = 11.69         # long edge, for the dpi readout

FILTERS = [("auto", "Auto"), ("original", "Original"),
           ("color", "Colour doc"), ("gray", "Greyscale"),
           ("bw", "Black & white"), ("whiteboard", "Whiteboard"),
           ("ink", "Ink boost"), ("photo", "Photo")]

SLIDERS = [
    ("Lighting", [("flatten", "Shadow removal", 0, 100, 0),
                  ("temp", "Temperature", -100, 100, 0),
                  ("tint", "Tint", -100, 100, 0)]),
    ("Tone", [("exposure", "Exposure", -100, 100, 0),
              ("contrast", "Contrast", -100, 100, 0),
              ("gamma", "Gamma", 0.3, 3.0, 2),
              ("highlights", "Highlights", -100, 100, 0),
              ("shadows", "Shadows", -100, 100, 0)]),
    ("Colour & detail", [("saturation", "Saturation", -100, 100, 0),
                         ("vibrance", "Vibrance", -100, 100, 0),
                         ("denoise", "Denoise", 0, 100, 0),
                         ("sharpen", "Sharpen", 0, 150, 0)]),
    ("Black & white", [("threshold", "Threshold bias", -50, 50, 0),
                       ("window", "Local window", 20, 300, 0)]),
    ("Geometry", [("straighten", "Straighten", -15, 15, 1)]),
]

OUTSIZES = [("detected", "Fit detected page"), ("native", "Original pixels"),
            ("a4", "A4 · 300 dpi"), ("letter", "Letter · 300 dpi")]

FORMATS = [(".jpg", "JPEG · quality 98", [int(cv2.IMWRITE_JPEG_QUALITY), 98]),
           (".jpg", "JPEG · quality 92", [int(cv2.IMWRITE_JPEG_QUALITY), 92]),
           (".png", "PNG · lossless", [int(cv2.IMWRITE_PNG_COMPRESSION), 6]),
           (".tif", "TIFF · uncompressed", [])]

SHORTCUTS = [
    ("Space", "Capture the current frame"),
    ("A", "Auto capture on stillness"),
    ("C", "Corner editor — drag the four handles"),
    ("D", "Detect the page again"),
    ("B", "Hold to compare against the original"),
    ("[  ]", "Rotate left / right"),
    ("←  →", "Previous / next page"),
    ("⌫", "Delete the current page"),
    ("⌘= ⌘- ⌘0", "Zoom in / out / fit"),
    ("scroll", "Zoom about the pointer; drag to pan; double-click to fit"),
    ("⌘S", "Save this page as an image"),
    ("⌘P", "Save every page as a PDF"),
    ("⌘R", "Start or stop the camera"),
]


class Page:
    __slots__ = ("frame", "corners", "adjust", "name", "ocr")

    def __init__(self, frame, corners, adjust, name):
        self.frame = frame              # full-resolution BGR, pristine
        self.corners = corners
        self.adjust = adjust
        self.name = name
        self.ocr = None


class Job(QObject):
    """Run one callable off the GUI thread and deliver its result back on it.

    Everything slow in this app — opening a camera, processing 16 MP, OCR,
    writing a PDF — has to leave the GUI thread or the window stops repainting,
    which is exactly the failure the previous build shipped with.
    """

    done = Signal(object, object)       # result, error

    def run(self, fn, *args, **kwargs):
        def work():
            try:
                self.done.emit(fn(*args, **kwargs), None)
            except Exception as exc:                # noqa: BLE001 - reported in UI
                self.done.emit(None, exc)
        threading.Thread(target=work, daemon=True).start()
        return self


class LiveFeed(QObject):
    """Scale and detect on a worker thread; the GUI thread only draws.

    Measured on the 16 MP camera, the old arrangement did all of this in the
    timer callback: 27 ms to downscale a 48 MB frame, 18 ms to turn it into a
    QImage, and every 400 ms another 28 ms to detect. At ten frames a second
    that is nearly half the GUI thread, delivered in 45 ms lumps — which is
    exactly what a stuttering button and a laggy window feel like.

    What crosses the thread boundary is a contiguous RGB buffer, so the GUI can
    wrap it in a QImage without copying anything.
    """

    frame = Signal(object, int, int, object, float)   # rgb, src_w, src_h, quad, motion
    failed = Signal(str)

    DETECT_WORK = 900        # long edge detection runs at
    MOTION_WORK = 240        # long edge the stillness check runs at

    def __init__(self, cam):
        super().__init__()
        self.cam = cam
        self.target = 1400
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)
            self._thread = None

    def pause(self, on):
        self._paused.set() if on else self._paused.clear()

    def _work(self):
        last_seq = -1
        last_detect = 0.0
        quad = None
        miss = 0
        prev = None
        motion = 0.0
        while not self._stop.is_set():
            if self._paused.is_set() or not self.cam.is_open():
                time.sleep(0.05)
                last_seq = -1
                continue
            seq = self.cam.sequence()
            src = self.cam.latest()
            if src is None or seq == last_seq:
                time.sleep(0.008)
                continue
            last_seq = seq

            now = time.time()
            if now - last_detect >= DETECT_EVERY:
                last_detect = now
                work = imaging.fit(src, self.DETECT_WORK)
                found = detect.detect(work, sticky=True)
                if found is not None:
                    quad, miss = found, 0
                else:
                    miss += 1
                    if miss > 3:
                        quad = None
                g = cv2.cvtColor(imaging.fit(work, self.MOTION_WORK), cv2.COLOR_BGR2GRAY)
                if prev is not None and prev.shape == g.shape:
                    motion = float(np.mean(cv2.absdiff(g, prev)))
                prev = g

            small = imaging.fit(src, self.target)
            rgb = np.ascontiguousarray(small[:, :, ::-1])
            self.frame.emit(rgb, src.shape[1], src.shape[0],
                            None if quad is None else quad.copy(), motion)

    def _loop(self):
        """Guarded wrapper. An exception in here used to kill the thread and
        freeze the preview for good, with nothing on screen to say why."""
        while not self._stop.is_set():
            try:
                self._work()
            except Exception as exc:                # noqa: BLE001
                self.failed.emit(str(exc))
                time.sleep(0.25)


class Renderer(QObject):
    """One background renderer for the editor, newest request wins.

    Coalescing rather than queueing is the point: dragging a slider produces
    dozens of requests and only the last one is worth anything. A queue would
    render every intermediate value and finish seconds after the drag ended.
    """

    ready = Signal(object, int)      # bgr image, request id

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._req = None
        self._wake = threading.Event()
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def request(self, rid, frame, adjust, corners, max_dim, fast):
        with self._lock:
            self._req = (rid, frame, dict(adjust),
                         None if corners is None else np.array(corners), max_dim, fast)
        self._wake.set()

    def stop(self):
        self._stop = True
        self._wake.set()

    def _loop(self):
        while not self._stop:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                req = self._req
                self._req = None
            if req is None or self._stop:
                continue
            try:
                rid, frame, adjust, corners, max_dim, fast = req
                img = imaging.process(frame, adjust, corners, max_dim=max_dim, fast=fast)
            except Exception:                       # noqa: BLE001 - stale request
                continue
            self.ready.emit(img, rid)


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Overhead Scanner")
        self.resize(1480, 940)
        self.setMinimumSize(1120, 720)

        self.cam = camera.Camera()
        self.devices = []
        self.pages = []
        self.current = -1
        self.mode = "live"
        self.busy = False
        self.auto = False
        self.live = False           # camera was running and has not been stopped
        self._retries = 0
        self._upgrade_tried = False
        self._scanning = False

        self.live_quad = None
        self._still_since = 0.0
        self._auto_last = 0.0
        self._rgb = None            # buffer the on-screen QImage points into
        self._compare_rgb = None
        self._comparing = False
        self._want_camera = ""      # name remembered from the last session
        self._fps_at = (0.0, 0)
        self._rid = 0               # newest render request
        self._uncropped = None      # cached full view, for corner dragging
        self._jobs = []
        self.sliders = {}
        self.filter_buttons = {}

        # Built before the widgets: set_mode() talks to the feed, and the
        # window starts in live mode.
        self.feed = LiveFeed(self.cam)
        self.feed.frame.connect(self._live_frame)
        self.feed.failed.connect(
            lambda msg: self.toast.show_message("Preview: %s" % msg, "bad"))
        self.renderer = Renderer()
        self.renderer.ready.connect(self._rendered)

        self._build()
        self._menus()
        self._restore_state()
        self._sync_controls()
        self.set_mode("live")
        self.feed.start()

        # A slow heartbeat, only to notice a camera that has died. Frames
        # arrive by signal now, not by polling.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._health)
        self.timer.start(500)

        # Fires once the operator stops moving a slider: swaps the draft for a
        # full-quality render and refreshes the thumbnail.
        self.settle = QTimer(self)
        self.settle.setSingleShot(True)
        self.settle.timeout.connect(self._settled)

        QTimer.singleShot(120, self.rescan)

    # ══ layout ═══════════════════════════════════════════════════

    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        col = QVBoxLayout(root)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        col.addWidget(self._topbar())
        col.addWidget(hair())

        body = QWidget()
        row = QHBoxLayout(body)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._tray())
        row.addWidget(hair(vertical=True))
        row.addWidget(self._stage(), 1)
        row.addWidget(hair(vertical=True))
        row.addWidget(self._inspector())
        col.addWidget(body, 1)

        self.toast = Toast(self.preview)
        self._shortcuts()

    def _topbar(self):
        bar = QWidget()
        bar.setObjectName("topbar")
        bar.setFixedHeight(56)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(8)

        lay.addWidget(label("Overhead Scanner", "title"))
        lay.addSpacing(18)

        lay.addWidget(label("Camera", "muted"))
        self.device_box = QComboBox()
        self.device_box.setMinimumWidth(230)
        self.device_box.currentIndexChanged.connect(self._device_changed)
        lay.addWidget(self.device_box)

        self.btn_start = button("Start", self.toggle_camera, kind="primary",
                                tip="Open the selected camera  (⌘R restarts)")
        lay.addWidget(self.btn_start)
        lay.addWidget(button("Rescan", self.rescan, kind="ghost",
                             tip="Look for cameras again"))

        lay.addWidget(spacer())

        self.btn_upgrade = button("Try 16 MP", self.upgrade_camera, kind="ghost",
                                  tip="Ask the camera for its top mode again")
        self.btn_upgrade.hide()
        lay.addWidget(self.btn_upgrade)
        self.lbl_state = label("", "muted")
        lay.addWidget(self.lbl_state)
        self.lbl_res = label("", "badge")
        lay.addWidget(self.lbl_res)
        return bar

    def _tray(self):
        panel = QWidget()
        panel.setObjectName("panel")
        panel.setFixedWidth(168)
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        head = QWidget()
        head.setObjectName("panel")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(14, 12, 10, 8)
        self.lbl_pages = label("Pages", "heading")
        hl.addWidget(self.lbl_pages)
        hl.addStretch(1)
        add = button("＋", self.import_images, kind="ghost", tip="Import images")
        add.setFixedWidth(28)
        hl.addWidget(add)
        col.addWidget(head)

        self.tray = QListWidget()
        self.tray.setIconSize(THUMB)
        self.tray.setViewMode(QListWidget.IconMode)
        self.tray.setFlow(QListWidget.TopToBottom)
        self.tray.setWrapping(False)
        self.tray.setResizeMode(QListWidget.Adjust)
        self.tray.setMovement(QListWidget.Static)
        self.tray.setSpacing(2)
        self.tray.currentRowChanged.connect(self._tray_changed)
        col.addWidget(self.tray, 1)

        foot = QWidget()
        foot.setObjectName("panel")
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(12, 6, 12, 12)
        fl.addWidget(button("Clear all", self.clear_pages, kind="ghost"))
        col.addWidget(foot)
        return panel

    def _stage(self):
        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        bar = QWidget()
        bar.setObjectName("bar")
        bar.setFixedHeight(52)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(6)

        self.btn_live = button("Live", lambda: self.set_mode("live"), checkable=True)
        self.btn_edit = button("Edit", lambda: self.set_mode("edit"), checkable=True)
        lay.addWidget(self.btn_live)
        lay.addWidget(self.btn_edit)
        lay.addWidget(spacer())

        self.btn_capture = button("●   Capture", self.capture, kind="record",
                                  tip="Keep the current frame  (Space)")
        lay.addWidget(self.btn_capture)
        self.btn_auto = button("Auto", self.toggle_auto, checkable=True,
                               tip="Capture whenever the scene goes still")
        lay.addWidget(self.btn_auto)
        lay.addWidget(spacer())

        self.btn_corners = button("Corners", self.toggle_corners, checkable=True,
                                  tip="Drag the four page corners  (C)")
        lay.addWidget(self.btn_corners)
        lay.addWidget(button("Detect", self.redetect, tip="Find the page again  (D)"))
        lay.addWidget(button("Delete", self.delete_page, kind="ghost", tip="⌫"))

        lay.addSpacing(10)
        self.btn_compare = button("Before", None, kind="ghost",
                                  tip="Hold to see the unprocessed page  (B)")
        self.btn_compare.pressed.connect(lambda: self.compare(True))
        self.btn_compare.released.connect(lambda: self.compare(False))
        lay.addWidget(self.btn_compare)
        for text, tip, fn in (("−", "Zoom out  (⌘-)", lambda: self.zoom(1 / 1.4)),
                              ("+", "Zoom in  (⌘=)", lambda: self.zoom(1.4)),
                              ("Fit", "Fit to the window  (⌘0)", self.preview_fit)):
            b = button(text, fn, kind="ghost", tip=tip)
            b.setFixedWidth(46 if text == "Fit" else 30)
            lay.addWidget(b)
        col.addWidget(bar)
        col.addWidget(hair())

        self.preview = PreviewView()
        self.preview.corners_changed.connect(self._corners_dragged)
        col.addWidget(self.preview, 1)

        col.addWidget(hair())
        foot = QWidget()
        foot.setObjectName("bar")
        foot.setFixedHeight(30)
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(14, 0, 14, 0)
        self.lbl_info = label("", "note")
        fl.addWidget(self.lbl_info)
        fl.addStretch(1)
        self.lbl_hint = label("", "note")
        fl.addWidget(self.lbl_hint)
        # How still the scene is. Auto capture fires when this empties, and
        # without a meter that moment is invisible — you end up guessing why
        # nothing fired, or why it fired early.
        fl.addSpacing(10)
        self.motion = QProgressBar()
        self.motion.setRange(0, 100)
        self.motion.setFixedWidth(70)
        self.motion.setTextVisible(False)
        self.motion.setToolTip("Scene movement")
        self.motion.hide()
        fl.addWidget(self.motion)
        self.lbl_fps = label("", "note")
        self.lbl_fps.setFixedWidth(46)
        self.lbl_fps.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        fl.addWidget(self.lbl_fps)
        col.addWidget(foot)
        return wrap

    def _inspector(self):
        tabs = QTabWidget()
        tabs.setFixedWidth(322)
        tabs.addTab(self._scroll(self._adjust_tab()), "Adjust")
        tabs.addTab(self._scroll(self._export_tab()), "Export")
        return tabs

    @staticmethod
    def _scroll(inner):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setWidget(inner)
        return area

    def _adjust_tab(self):
        page = QWidget()
        page.setObjectName("panel")
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 0, 0, 16)
        col.setSpacing(0)

        sec = Section("Filter")
        grid = QGridLayout()
        grid.setSpacing(4)
        for i, (key, text) in enumerate(FILTERS):
            b = button(text, lambda k=key: self.set_filter(k), checkable=True)
            grid.addWidget(b, i // 2, i % 2)
            self.filter_buttons[key] = b
        sec.add_layout(grid)
        col.addWidget(sec)

        sec = Section("Rotate & flip")
        row = QHBoxLayout()
        row.setSpacing(4)
        for text, tip, fn in (("↺", "Rotate left  ([)", lambda: self.rotate(-90)),
                              ("↻", "Rotate right  (])", lambda: self.rotate(90)),
                              ("⇋", "Flip horizontally", lambda: self.flip("fliph")),
                              ("⇵", "Flip vertically", lambda: self.flip("flipv"))):
            row.addWidget(button(text, fn, tip=tip))
        sec.add_layout(row)
        col.addWidget(sec)

        sec = Section("Output size")
        self.chk_invert = QCheckBox("Invert (negatives, chalkboards)")
        self.chk_invert.toggled.connect(self._invert_changed)
        sec.add(self.chk_invert)
        self.outsize = QComboBox()
        for _key, text in OUTSIZES:
            self.outsize.addItem(text)
        self.outsize.currentIndexChanged.connect(self._outsize_changed)
        sec.add(self.outsize)
        col.addWidget(sec)

        for title, items in SLIDERS:
            col.addWidget(hair())
            sec = Section(title)
            for key, text, lo, hi, dec in items:
                row = SliderRow(key, text, lo, hi, imaging.DEFAULTS[key], dec)
                row.changed.connect(self._slider_changed)
                self.sliders[key] = sec.add(row)
            col.addWidget(sec)

        col.addWidget(hair())
        sec = Section("Apply")
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(button("Reset all", self.reset_adjust,
                             tip="Back to the filter's own settings"))
        row.addWidget(button("To every page", self.apply_all))
        sec.add_layout(row)
        col.addWidget(sec)
        col.addStretch(1)
        return page

    def _export_tab(self):
        page = QWidget()
        page.setObjectName("panel")
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 0, 0, 16)
        col.setSpacing(0)

        sec = Section("This page")
        self.fmt = QComboBox()
        for _ext, text, _params in FORMATS:
            self.fmt.addItem(text)
        sec.add(self.fmt)
        sec.add(button("Save image…", self.export_image, kind="primary", tip="⌘S"))
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(button("Copy", self.copy_image, tip="Put the page on the clipboard"))
        row.addWidget(button("Print…", self.print_page))
        row.addWidget(button("Estimate", self.estimate,
                             tip="Render it and report the real file size"))
        sec.add_layout(row)
        self.lbl_estimate = label("Saved at the crop's full resolution — never "
                                  "downscaled.", "note")
        self.lbl_estimate.setWordWrap(True)
        sec.add(self.lbl_estimate)
        col.addWidget(sec)

        col.addWidget(hair())
        sec = Section("All pages")
        row = QHBoxLayout()
        row.setSpacing(4)
        self.pdf_size = QComboBox()
        self.pdf_size.addItems(["A4", "Letter"])
        row.addWidget(self.pdf_size)
        self.chk_searchable = QCheckBox("Searchable")
        self.chk_searchable.setChecked(True)
        self.chk_searchable.setToolTip("Embed the OCR text as an invisible layer")
        row.addWidget(self.chk_searchable)
        sec.add_layout(row)
        sec.add(button("Save PDF…", self.export_pdf, kind="primary", tip="⌘P"))
        sec.add(button("Save text…", self.export_text))
        col.addWidget(sec)

        col.addWidget(hair())
        sec = Section("OCR")
        row = QHBoxLayout()
        row.setSpacing(4)
        self.ocr_lang = QComboBox()
        langs = ocr.languages() or ["eng"]
        self.ocr_lang.addItems(langs)
        if "eng" in langs:
            self.ocr_lang.setCurrentText("eng")
        self.ocr_lang.setToolTip("Recognition language")
        row.addWidget(self.ocr_lang, 1)
        self.ocr_psm = QComboBox()
        for code, text in (("3", "Auto layout"), ("4", "Columns"), ("6", "One block"),
                           ("11", "Sparse text"), ("1", "Auto + orientation")):
            self.ocr_psm.addItem(text, code)
        self.ocr_psm.setToolTip("Page segmentation — how the layout is read")
        row.addWidget(self.ocr_psm, 1)
        sec.add_layout(row)
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(button("Read page", self.run_ocr))
        row.addWidget(button("Read all", self.run_ocr_all))
        row.addWidget(button("Copy text", self.copy_text))
        sec.add_layout(row)
        self.lbl_ocr = label("", "note")
        self.lbl_ocr.setWordWrap(True)
        sec.add(self.lbl_ocr)
        self.txt_ocr = QTextEdit()
        self.txt_ocr.setMinimumHeight(220)
        # Read-only for a reason beyond the obvious: an editable field swallows
        # the single-key shortcuts (Space, C, D) the rest of the app runs on.
        self.txt_ocr.setReadOnly(True)
        sec.add(self.txt_ocr)
        if not ocr.available():
            self.lbl_ocr.setText(ocr.install_hint())
            self.lbl_ocr.setStyleSheet("color: %s;" % WARN)
        col.addWidget(sec)
        col.addStretch(1)
        return page

    def _shortcuts(self):
        """Single-key actions. The menu bar carries the ⌘ ones."""
        def act(seq, fn):
            a = QAction(self)
            a.setShortcut(QKeySequence(seq))
            a.triggered.connect(lambda *_: fn())
            self.addAction(a)

        act("Space", self.capture)
        act("C", self.toggle_corners)
        act("D", self.redetect)
        act("A", self.toggle_auto)
        act("[", lambda: self.rotate(-90))
        act("]", lambda: self.rotate(90))
        act("Backspace", self.delete_page)
        act("Left", lambda: self.step_page(-1))
        act("Right", lambda: self.step_page(1))

        hold = QAction(self)
        hold.setShortcut(QKeySequence("B"))
        hold.triggered.connect(lambda *_: self.compare(not self._comparing))
        self.addAction(hold)

    def _menus(self):
        """A real menu bar. On macOS its absence is what makes an app feel
        like a script someone left running."""
        bar = self.menuBar()

        def add(menu, text, fn, seq=None, checkable=False):
            a = QAction(text, self)
            if seq:
                a.setShortcut(QKeySequence(seq))
            a.setCheckable(checkable)
            a.triggered.connect(lambda *_: fn())
            menu.addAction(a)
            return a

        m = bar.addMenu("&File")
        add(m, "Import images…", self.import_images, "Ctrl+O")
        m.addSeparator()
        add(m, "Save Image…", self.export_image, "Ctrl+S")
        add(m, "Save PDF…", self.export_pdf, "Ctrl+P")
        add(m, "Save Text…", self.export_text)
        m.addSeparator()
        add(m, "Print…", self.print_page, "Ctrl+Shift+P")

        m = bar.addMenu("&Edit")
        add(m, "Copy Image", self.copy_image, "Ctrl+C")
        add(m, "Copy Text", self.copy_text, "Ctrl+Shift+C")
        m.addSeparator()
        add(m, "Reset Adjustments", self.reset_adjust)
        add(m, "Apply to Every Page", self.apply_all)
        add(m, "Delete Page", self.delete_page)
        add(m, "Clear All Pages", self.clear_pages)

        m = bar.addMenu("&View")
        add(m, "Zoom In", lambda: self.zoom(1.4), "Ctrl+=")
        add(m, "Zoom Out", lambda: self.zoom(1 / 1.4), "Ctrl+-")
        add(m, "Fit to Window", self.preview_fit, "Ctrl+0")
        m.addSeparator()
        self.act_grid = add(m, "Thirds Grid",
                            lambda: self.preview.set_guides(grid=self.act_grid.isChecked()),
                            "Ctrl+G", checkable=True)
        self.act_cross = add(m, "Centre Cross",
                             lambda: self.preview.set_guides(cross=self.act_cross.isChecked()),
                             checkable=True)

        m = bar.addMenu("&Camera")
        add(m, "Start / Stop", self.toggle_camera, "Ctrl+R")
        add(m, "Rescan Devices", self.rescan)
        add(m, "Try Full Resolution", self.upgrade_camera)
        m.addSeparator()
        add(m, "Capture", self.capture)
        add(m, "Auto Capture", self.toggle_auto)

        m = bar.addMenu("&Help")
        add(m, "Keyboard Shortcuts", self.show_help, "Ctrl+/")

    def show_help(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard shortcuts")
        dlg.setStyleSheet(QSS)
        col = QVBoxLayout(dlg)
        col.setContentsMargins(22, 18, 22, 18)
        col.setSpacing(9)
        col.addWidget(label("Keyboard shortcuts", "title"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        for i, (keys, what) in enumerate(SHORTCUTS):
            k = label(keys, "badge")
            k.setStyleSheet("color: %s; font-family: Menlo; font-size: 11px;" % ACCENT)
            grid.addWidget(k, i, 0, Qt.AlignRight)
            grid.addWidget(label(what, "sublabel"), i, 1)
        col.addLayout(grid)
        col.addSpacing(6)
        close = button("Close", dlg.accept, kind="primary")
        col.addWidget(close, 0, Qt.AlignRight)
        dlg.exec()

    # ══ window state ═════════════════════════════════════════════

    def _restore_state(self):
        st = QSettings("overhead-scanner", "app")
        geo = st.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        self.act_grid.setChecked(st.value("grid", False, type=bool))
        self.act_cross.setChecked(st.value("cross", False, type=bool))
        self.preview.set_guides(grid=self.act_grid.isChecked(),
                                cross=self.act_cross.isChecked())
        want = st.value("camera", "", type=str)
        if want:
            self._want_camera = want

    def _save_state(self):
        st = QSettings("overhead-scanner", "app")
        st.setValue("geometry", self.saveGeometry())
        st.setValue("grid", self.act_grid.isChecked())
        st.setValue("cross", self.act_cross.isChecked())
        st.setValue("camera", self.cam.name or "")

    # ══ camera ═══════════════════════════════════════════════════

    def rescan(self):
        """Re-enumerate. Cheap and safe: it opens no camera at all.

        That matters more than it sounds. An earlier version measured each
        device's maximum by opening it, and the opening itself left this camera
        refusing its 16 MP mode for the next few seconds — so the app that had
        just been told "4656x3496" would then settle at 1080p.

        Off the GUI thread all the same: the very first run may have to compile
        the little AVFoundation name helper.
        """
        if self._scanning:
            return
        self._scanning = True
        self.lbl_state.setText("looking for cameras…")
        self._job(camera.Camera.list_devices)(self._devices_listed)

    def _devices_listed(self, devices, err):
        self._scanning = False
        if err is not None:
            self.lbl_state.setText("could not list cameras")
            self.toast.show_message(str(err), "bad")
            return
        was = self.device_box.currentData()
        self.devices = devices or []
        self.device_box.blockSignals(True)
        self.device_box.clear()
        for d in self.devices:
            tag = {"external": "USB", "builtin": "built-in",
                   "continuity": "iPhone"}.get(d["kind"], "")
            text = "%s  ·  %s" % (d["name"], tag) if tag else d["name"]
            self.device_box.addItem(text, d["index"])
        # Prefer the camera in use, then the one from last session, then the
        # best-ranked device.
        for wanted in (was, self._want_camera):
            if wanted in (None, ""):
                continue
            i = (self.device_box.findData(wanted) if isinstance(wanted, int)
                 else self.device_box.findText(str(wanted), Qt.MatchStartsWith))
            if i >= 0:
                self.device_box.setCurrentIndex(i)
                break
        self._want_camera = ""
        self.device_box.blockSignals(False)

        if not self.devices:
            self.lbl_state.setText("no camera found")
            self.preview.set_placeholder("No camera found.\nPlug one in, then press Rescan.")
            return
        self.lbl_state.setText("%d camera%s" % (len(self.devices),
                                                "" if len(self.devices) == 1 else "s"))
        if not self.cam.is_open():
            self.start_camera()

    def toggle_camera(self):
        self.stop_camera() if self.cam.is_open() else self.start_camera()

    def start_camera(self):
        if self.busy or not self.devices:
            return
        index = self.device_box.currentData()
        name = self.device_box.currentText().split("  ·")[0]
        self.busy = True
        self.btn_start.setEnabled(False)
        self.btn_start.setText("Opening…")
        self.lbl_state.setText("bringing up %s" % name)
        self.preview.set_placeholder("Opening %s…\nFirst frame takes a few seconds "
                                     "at full resolution." % name)
        detect.reset_sticky()
        # Open at whatever the device streams by default, which it never
        # refuses and answers in about a second, then climb to the full sensor
        # mode a moment later. Asking for 16 MP up front is a coin toss on this
        # camera and costs eight seconds of black window when it loses; this
        # way there is a picture almost immediately and the resolution arrives
        # while you are still putting the page down.
        self._upgrade_tried = False
        self._job(camera.Camera.open, self.cam, index, name, None)(self._camera_opened)

    RETRIES = 2

    def _camera_opened(self, ok, err):
        self.busy = False
        self.btn_start.setEnabled(True)
        self.btn_start.setText("Stop" if ok else "Start")
        if not ok:
            msg = self.cam.error or (str(err) if err else "camera did not start")
            # A failed open is usually not a broken camera, it is an impatient
            # one: this device refuses its top mode for a few seconds after
            # anything else has had it, so a rest and a retry is the fix, and
            # making the operator press the button again is just theatre.
            if self._retries < self.RETRIES:
                self._retries += 1
                self.lbl_state.setText("camera busy — retrying (%d of %d)"
                                       % (self._retries, self.RETRIES))
                self.preview.set_placeholder("Camera did not come up.\nGiving it a "
                                             "moment and trying again…")
                # Measured: this camera needs a few seconds of being left
                # alone before it will bring its top mode up again. Retrying
                # straight away is what makes the second attempt fail too.
                QTimer.singleShot(4000, self.start_camera)
                return
            self._retries = 0
            self.lbl_state.setText("failed")
            self.preview.set_placeholder("Could not start the camera.\n%s\n\n"
                                         "Unplug it and back in, then press Start."
                                         % msg)
            self.toast.show_message(msg, "bad")
            return
        self._retries = 0
        self.live = True
        self.lbl_state.setText(self.cam.name or "")
        self._update_res()
        self.set_mode("live")
        self.toast.show_message("%s · %d×%d" % (self.cam.name, self.cam.width,
                                                self.cam.height), "good")
        if not self.cam.at_max and not self._upgrade_tried:
            QTimer.singleShot(1800, self._auto_upgrade)

    def _auto_upgrade(self):
        """Climb to the sensor's full mode once, on its own.

        Once only, and never again automatically: a camera whose maximum really
        is 1080p would otherwise be reopened every few seconds forever. After
        this the button in the top bar is the way to ask again.
        """
        if self._upgrade_tried or self.busy or not self.cam.is_open() or self.cam.at_max:
            return
        self._upgrade_tried = True
        self.upgrade_camera()

    def _update_res(self):
        self.lbl_res.setText("%d×%d" % (self.cam.width, self.cam.height))
        top = camera.LADDER[0]
        below = self.cam.is_open() and not self.cam.at_max
        self.lbl_res.setStyleSheet("color: %s;" % (WARN if below else GOOD))
        self.btn_upgrade.setVisible(bool(below))
        self.btn_upgrade.setToolTip("Camera is at %d×%d; ask for %d×%d again"
                                    % (self.cam.width, self.cam.height, top[0], top[1]))

    def upgrade_camera(self):
        if self.busy or not self.cam.is_open():
            return
        self.busy = True
        self.btn_upgrade.setEnabled(False)
        self.lbl_state.setText("asking for full resolution…")
        self._job(camera.Camera.upgrade, self.cam)(self._upgraded)

    def _upgraded(self, ok, _err):
        self.busy = False
        self.btn_upgrade.setEnabled(True)
        self.live = self.cam.is_open()
        self.lbl_state.setText(self.cam.name if self.live else "stopped")
        self._update_res()
        if ok:
            self.toast.show_message("Now at %d×%d" % (self.cam.width, self.cam.height),
                                    "good")
        elif self.live:
            self.toast.show_message(
                "Camera would not go above %d×%d just now — Try 16 MP again in a "
                "few seconds" % (self.cam.width, self.cam.height), "warn", 5000)
        else:
            self.toast.show_message(self.cam.error or "camera stopped", "bad")

    def stop_camera(self):
        self.live = False
        self.cam.close()
        self.btn_start.setText("Start")
        self.lbl_res.setText("")
        self.btn_upgrade.hide()
        self.live_quad = None
        self.lbl_state.setText("stopped")
        if self.mode == "live":
            self.preview.clear()
            self.preview.set_placeholder("Camera stopped.\nPress Start.")

    def _device_changed(self, _i):
        if self.cam.is_open():
            self.stop_camera()
            QTimer.singleShot(400, self.start_camera)

    # ══ live loop ════════════════════════════════════════════════

    def _health(self):
        """Notice a camera that was running and has died.

        Testing `cam.error` alone also fires while a failed open is being
        retried, and used to overwrite the reason on screen with a bare
        "stopped".
        """
        if self.live and not self.busy and self.cam.error and not self.cam.is_open():
            msg = self.cam.error
            self.stop_camera()
            self.toast.show_message(msg, "bad")
            return
        # Frame rate, from the camera's own counter — the honest number, not
        # how often the window happens to repaint.
        now, seq = time.time(), self.cam.sequence()
        was_at, was_seq = self._fps_at
        if self.cam.is_open() and was_at and now > was_at:
            self.lbl_fps.setText("%.0f fps" % ((seq - was_seq) / (now - was_at)))
        elif not self.cam.is_open():
            self.lbl_fps.setText("")
        self._fps_at = (now, seq)

    def _live_frame(self, rgb, src_w, src_h, quad, motion):
        if self.mode != "live":
            return
        # Hold the buffer: the QImage points straight into it rather than
        # copying, and letting it be collected would paint garbage.
        self._rgb = rgb
        self.preview.set_image(qtui.wrap_rgb(rgb))
        self.preview.set_quad(quad)
        self.live_quad = quad
        self._live_info(src_w, src_h)
        self._check_still(motion)
        if self.auto:
            self.motion.setValue(int(min(100, motion * 18)))

    def _live_info(self, w, h):
        if self.live_quad is None:
            self.lbl_info.setText("live %d×%d  ·  no page found — the whole frame "
                                  "will be kept" % (w, h))
            self.lbl_hint.setText("")
            return
        cw, ch = detect.output_size(self.live_quad, w, h)
        dpi = max(cw, ch) / A4_INCHES
        cover = self._quad_area(self.live_quad)
        self.lbl_info.setText("live %d×%d  ·  crop %d×%d  ·  ≈%d dpi"
                              % (w, h, cw, ch, round(dpi)))
        if cover < 0.35:
            self.lbl_hint.setText("page fills %d%% of the frame — move the camera "
                                  "closer for a sharper scan" % round(cover * 100))
            self.lbl_hint.setStyleSheet("color: %s;" % WARN)
        else:
            self.lbl_hint.setText("page fills %d%%" % round(cover * 100))
            self.lbl_hint.setStyleSheet("color: %s;" % FG3)

    @staticmethod
    def _quad_area(quad):
        p = np.asarray(quad, dtype=np.float64)
        x, y = p[:, 0], p[:, 1]
        return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0

    def _check_still(self, motion):
        """Auto capture fires once the scene stops moving, not on a timer.

        A page turn is a burst of change followed by stillness; shooting on
        stillness is what makes a stack of pages a rhythm rather than a chore.
        """
        if not self.auto:
            return
        now = time.time()
        if motion > 2.4:
            self._still_since = now
        elif (now - self._still_since > 1.1 and now - self._auto_last > 2.6
                and self.live_quad is not None):
            self._auto_last = now
            self.capture()

    def toggle_auto(self):
        self.auto = not self.auto
        self.btn_auto.setChecked(self.auto)
        self.motion.setVisible(self.auto)
        self._still_since = time.time()
        self.toast.show_message("Auto capture on — hold still after each page"
                                if self.auto else "Auto capture off")

    # ══ pages ════════════════════════════════════════════════════

    def capture(self):
        if self.mode != "live":
            self.set_mode("live")
            return
        frame = self._fresh_frame()
        if frame is None:
            self.toast.show_message("No frame to capture", "bad")
            return
        # The quad on screen is the quad that gets used. Re-running detection on
        # the captured frame is how the previous build ended up cropping
        # something other than the outline the operator was looking at.
        quad = None if self.live_quad is None else self.live_quad.copy()
        page = Page(frame, quad, imaging.new_adjust(),
                    "Page %d" % (len(self.pages) + 1))
        self.pages.append(page)
        self._add_thumb(page)
        self.current = len(self.pages) - 1
        self.tray.setCurrentRow(self.current)
        self.set_mode("edit")
        if quad is None:
            self.toast.show_message("No page found — kept the whole frame. "
                                    "Use Corners.", "warn")

    def _fresh_frame(self, wait=1.2):
        """A frame at the resolution the camera says it is running at.

        Belt and braces after a mode change: the app advertises 4656x3496 in
        the top bar the moment the mode comes up, and a capture in the gap
        before the next frame arrives would otherwise be silently smaller than
        the number on screen.
        """
        deadline = time.time() + wait
        while True:
            frame = self.cam.grab()
            if frame is None:
                if time.time() > deadline:
                    return None
            elif (frame.shape[1] >= self.cam.width * 0.95
                    and frame.shape[0] >= self.cam.height * 0.95):
                return frame
            elif time.time() > deadline:
                return frame
            QApplication.processEvents()
            time.sleep(0.03)

    def page(self):
        return self.pages[self.current] if 0 <= self.current < len(self.pages) else None

    def _add_thumb(self, page):
        item = QListWidgetItem(page.name)
        item.setSizeHint(QSize(THUMB.width() + 30, THUMB.height() + 28))
        self.tray.addItem(item)
        self._refresh_thumb(len(self.pages) - 1)
        self._count()

    def _refresh_thumb(self, i):
        if not (0 <= i < len(self.pages)) or i >= self.tray.count():
            return
        page = self.pages[i]
        try:
            img = imaging.process(page.frame, page.adjust, page.corners, max_dim=260)
        except Exception:
            img = imaging.fit(page.frame, 260)
        qimg = to_qimage(img)
        if qimg is not None:
            self.tray.item(i).setIcon(QIcon(QPixmap.fromImage(qimg)))

    def _count(self):
        n = len(self.pages)
        self.lbl_pages.setText("Pages · %d" % n if n else "Pages")

    def _tray_changed(self, row):
        if row < 0 or row == self.current:
            return
        self.current = row
        self._sync_controls()
        self.set_mode("edit")

    def step_page(self, delta):
        if not self.pages:
            return
        self.tray.setCurrentRow(max(0, min(len(self.pages) - 1, self.current + delta)))

    def delete_page(self):
        page = self.page()
        if page is None:
            return
        i = self.current
        self.pages.pop(i)
        self.tray.blockSignals(True)
        self.tray.takeItem(i)
        for j in range(i, len(self.pages)):
            self.pages[j].name = "Page %d" % (j + 1)
            self.tray.item(j).setText(self.pages[j].name)
        self.tray.blockSignals(False)
        self.current = min(i, len(self.pages) - 1)
        self._count()
        if self.pages:
            self.tray.setCurrentRow(self.current)
            self._sync_controls()
            self._render_page()
        else:
            self.set_mode("live")

    def clear_pages(self):
        if not self.pages:
            return
        if QMessageBox.question(self, "Clear all",
                                "Discard all %d pages?" % len(self.pages)
                                ) != QMessageBox.Yes:
            return
        self.pages = []
        self.current = -1
        self.tray.blockSignals(True)
        self.tray.clear()
        self.tray.blockSignals(False)
        self._count()
        self.set_mode("live")

    def import_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import images", os.path.expanduser("~"),
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp)")
        added = 0
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            page = Page(img, detect.detect(imaging.fit(img, 900)),
                        imaging.new_adjust(), "Page %d" % (len(self.pages) + 1))
            self.pages.append(page)
            self._add_thumb(page)
            added += 1
        if added:
            self.current = len(self.pages) - 1
            self.tray.setCurrentRow(self.current)
            self._sync_controls()
            self.set_mode("edit")
            self.toast.show_message("Imported %d image%s" % (added, "" if added == 1 else "s"),
                                    "good")

    # ══ modes ════════════════════════════════════════════════════

    def set_mode(self, mode):
        if mode == "edit" and self.page() is None:
            mode = "live"
        self.mode = mode
        self.btn_live.setChecked(mode == "live")
        self.btn_edit.setChecked(mode == "edit")
        self.btn_capture.setEnabled(mode == "live" and self.cam.is_open())
        if mode == "live":
            self.btn_corners.setChecked(False)
            self.preview.set_editable(False)
            self.preview.set_placeholder("Camera stopped.\nPress Start."
                                         if not self.cam.is_open() else "Starting…")
            if not self.cam.is_open():
                self.preview.clear()
        else:
            self._uncropped = None
            self._render_page()
        # No point scaling 16 MP frames nobody is looking at.
        self.feed.pause(mode != "live")

    def toggle_corners(self):
        if self.mode != "edit" or self.page() is None:
            self.toast.show_message("Capture a page first", "warn")
            self.btn_corners.setChecked(False)
            return
        on = not self.preview.is_editable()
        self.btn_corners.setChecked(on)
        page = self.page()
        if on and page.corners is None:
            page.corners = detect.full_frame().copy()
        self.preview.set_editable(on)
        self._compare_rgb = None
        self._uncropped = None if on else self._uncropped
        self._render_page()
        if not on:
            self._refresh_thumb(self.current)

    def _corners_dragged(self, quad):
        page = self.page()
        if page is None:
            return
        page.corners = np.asarray(quad, dtype=np.float32)
        self._page_info(page)
        self.settle.start(SETTLE_MS)

    def redetect(self):
        page = self.page()
        if page is None:
            self.toast.show_message("Capture a page first", "warn")
            return
        quad = detect.detect(imaging.fit(page.frame, 900))
        page.corners = quad
        self._uncropped = None
        self._render_page()
        self.settle.start(SETTLE_MS)
        self.toast.show_message("Page found" if quad is not None else
                                "No page found — use Corners",
                                "good" if quad is not None else "warn")

    # ══ rendering the editor ═════════════════════════════════════

    def _render_page(self, draft=False):
        """Ask the background renderer for the current page.

        Nothing here touches the pipeline: a full-quality editor render is
        ~240 ms on a 16 MP frame, which on the GUI thread is a quarter-second
        freeze for every slider tick. Requests coalesce, so a drag produces one
        render per completed frame instead of a backlog.
        """
        page = self.page()
        if page is None or self.mode != "edit":
            return
        self._page_info(page)
        if self.preview.is_editable():
            # While the corners are being dragged, show the uncropped frame —
            # you cannot place a corner on a picture the corners already cut.
            # Cached, because it does not change while you drag.
            if self._uncropped is None:
                small = imaging.fit(page.frame, max(700, self.preview.width() * 2))
                self._uncropped = np.ascontiguousarray(small[:, :, ::-1])
            self._rgb = self._uncropped
            self.preview.set_image(qtui.wrap_rgb(self._uncropped))
            self.preview.set_quad(page.corners if page.corners is not None
                                  else detect.full_frame())
            return
        if not draft:
            self._compare_rgb = None
        self._rid += 1
        limit = (DRAFT_MAX if draft
                 else max(700, min(EDIT_MAX, self.preview.width() * 2)))
        self.renderer.request(self._rid, page.frame, page.adjust, page.corners,
                              limit, draft)

    def _rendered(self, img, rid):
        if rid != self._rid or self.mode != "edit" or self.preview.is_editable():
            return                      # superseded, or no longer on screen
        self._rgb = np.ascontiguousarray(img[:, :, ::-1])
        self.preview.set_image(qtui.wrap_rgb(self._rgb))
        self.preview.set_quad(None)

    def _page_info(self, page):
        fh, fw = page.frame.shape[:2]
        ow, oh = imaging.target_size(page.adjust, page.corners, fw, fh)
        dpi = max(ow, oh) / A4_INCHES
        self.lbl_info.setText("%s  ·  source %d×%d  ·  output %d×%d  ·  ≈%d dpi"
                              % (page.name, fw, fh, ow, oh, round(dpi)))
        self.lbl_hint.setText("drag the corners" if self.preview.is_editable() else "")
        self.lbl_hint.setStyleSheet("color: %s;" % ACCENT)

    def _queue_render(self):
        """A knob moved: draft now, full quality and a new thumbnail on stop."""
        self._compare_rgb = None
        self._render_page(draft=True)
        self.settle.start(SETTLE_MS)

    def _settled(self):
        self._render_page(draft=False)
        self._refresh_thumb(self.current)

    # ══ adjustments ══════════════════════════════════════════════

    def _sync_controls(self):
        """Push the current page's settings into the panel without echoing back."""
        page = self.page()
        a = page.adjust if page else imaging.new_adjust()
        preset = imaging.FILTERS.get(a.get("filter"), {})
        for key, row in self.sliders.items():
            row.set_value(a.get(key, imaging.DEFAULTS[key]),
                          default=preset.get(key, imaging.DEFAULTS[key]))
        for key, b in self.filter_buttons.items():
            b.setChecked(key == a.get("filter") and not a.get("custom"))
        self.chk_invert.blockSignals(True)
        self.chk_invert.setChecked(bool(a.get("invert")))
        self.chk_invert.blockSignals(False)
        self.outsize.blockSignals(True)
        keys = [k for k, _ in OUTSIZES]
        self.outsize.setCurrentIndex(keys.index(a.get("outsize", "detected"))
                                     if a.get("outsize") in keys else 0)
        self.outsize.blockSignals(False)
        self.txt_ocr.setPlainText((page.ocr or {}).get("text", "") if page else "")

    def _slider_changed(self, key, value):
        page = self.page()
        if page is None:
            return
        page.adjust[key] = value
        page.adjust["custom"] = True
        for _k, b in self.filter_buttons.items():
            b.setChecked(False)
        self._queue_render()

    def set_filter(self, name):
        page = self.page()
        if page is None:
            self.toast.show_message("Capture a page first", "warn")
            self._sync_controls()
            return
        imaging.set_filter(page.adjust, name)
        page.adjust["custom"] = False
        self._sync_controls()
        self._render_page()
        self.settle.start(SETTLE_MS)

    def _outsize_changed(self, i):
        page = self.page()
        if page is None:
            return
        page.adjust["outsize"] = OUTSIZES[i][0]
        self._render_page()
        self.settle.start(SETTLE_MS)

    def rotate(self, deg):
        page = self.page()
        if page is None:
            return
        page.adjust["rotate"] = (page.adjust.get("rotate", 0) + deg) % 360
        self._render_page()
        self.settle.start(SETTLE_MS)

    def flip(self, key):
        page = self.page()
        if page is None:
            return
        page.adjust[key] = not page.adjust.get(key, False)
        self._render_page()
        self.settle.start(SETTLE_MS)

    def apply_all(self):
        page = self.page()
        if page is None or len(self.pages) < 2:
            return
        keep = ("filter", "custom", "mode", "flatten", "wb", "temp", "tint",
                "exposure", "contrast", "gamma", "highlights", "shadows",
                "saturation", "vibrance", "denoise", "sharpen", "threshold",
                "window", "outsize")
        for i, other in enumerate(self.pages):
            if other is page:
                continue
            for k in keep:
                if k in page.adjust:
                    other.adjust[k] = page.adjust[k]
            self._refresh_thumb(i)
        self.toast.show_message("Applied to %d pages" % len(self.pages), "good")

    # ══ view ═════════════════════════════════════════════════════

    def zoom(self, factor):
        self.preview.zoom_by(factor)

    def preview_fit(self):
        self.preview.fit()

    def compare(self, on):
        """Hold to see the page before any processing.

        The single most useful check there is: it answers "is the filter
        helping or inventing?" in one keypress, which no amount of staring at
        the processed image will.
        """
        page = self.page()
        if on and (page is None or self.mode != "edit"):
            return
        self._comparing = bool(on)
        self.btn_compare.setDown(self._comparing)
        if not on:
            self.preview.set_compare(None)
            return
        if self._compare_rgb is None:
            a = dict(page.adjust)
            imaging.set_filter(a, "original")
            for k in imaging.GEOMETRY_KEYS:            # keep the crop and rotation
                a[k] = page.adjust.get(k, imaging.DEFAULTS[k])
            img = imaging.process(page.frame, a, page.corners, max_dim=DRAFT_MAX,
                                  fast=True)
            self._compare_rgb = np.ascontiguousarray(img[:, :, ::-1])
        self.preview.set_compare(qtui.wrap_rgb(self._compare_rgb))

    def _invert_changed(self, on):
        page = self.page()
        if page is None:
            return
        page.adjust["invert"] = bool(on)
        self._render_page()
        self.settle.start(SETTLE_MS)

    def reset_adjust(self):
        page = self.page()
        if page is None:
            return
        imaging.set_filter(page.adjust, page.adjust.get("filter", "auto"))
        page.adjust["custom"] = False
        self._sync_controls()
        self._render_page()
        self.settle.start(SETTLE_MS)
        self.toast.show_message("Back to the %s filter's own settings"
                                % page.adjust.get("filter", "auto"))

    # ══ clipboard and printing ═══════════════════════════════════

    def copy_image(self):
        page = self.page()
        if page is None:
            self.toast.show_message("Nothing to copy", "warn")
            return
        self._start_job("Rendering…", self._to_clipboard, page)

    def _to_clipboard(self, page):
        img = self._full(page)
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        # Copy: the clipboard outlives this array.
        QGuiApplication.clipboard().setImage(qtui.wrap_rgb(rgb).copy())
        return "%d×%d to the clipboard" % (img.shape[1], img.shape[0])

    def copy_text(self):
        text = self.txt_ocr.toPlainText()
        if not text.strip():
            self.toast.show_message("No text yet — press Read page", "warn")
            return
        QGuiApplication.clipboard().setText(text)
        self.toast.show_message("%d characters copied" % len(text), "good")

    def print_page(self):
        page = self.page()
        if page is None:
            self.toast.show_message("Nothing to print", "warn")
            return
        try:
            from PySide6.QtPrintSupport import QPrintDialog, QPrinter
        except ImportError:
            self.toast.show_message("Printing needs PySide6-Addons", "bad")
            return
        printer = QPrinter(QPrinter.HighResolution)
        if QPrintDialog(printer, self).exec() != QDialog.Accepted:
            return
        img = imaging.process(page.frame, page.adjust, page.corners, max_dim=3000)
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        qimg = qtui.wrap_rgb(rgb)
        painter = QPainter(printer)
        area = painter.viewport()
        size = qimg.size()
        size.scale(area.size(), Qt.KeepAspectRatio)
        painter.setViewport(area.x(), area.y(), size.width(), size.height())
        painter.setWindow(qimg.rect())
        painter.drawImage(0, 0, qimg)
        painter.end()
        self.toast.show_message("Sent to the printer", "good")

    # ══ export ═══════════════════════════════════════════════════

    def _format(self):
        return FORMATS[max(0, self.fmt.currentIndex())]

    def estimate(self):
        page = self.page()
        if page is None:
            self.toast.show_message("Nothing to measure", "warn")
            return
        self.lbl_estimate.setText("rendering at full resolution…")
        self._start_job("Estimating…", self._estimate, page, quiet="estimate")

    def _estimate(self, page):
        ext, name, params = self._format()
        img = self._full(page)
        ok, buf = cv2.imencode(ext, img, params)
        if not ok:
            raise RuntimeError("could not encode %s" % ext)
        dpi = max(img.shape[:2]) / A4_INCHES
        return ("%d×%d  ·  %.1f MP  ·  %s  ·  %.1f MB  ·  ≈%d dpi on A4"
                % (img.shape[1], img.shape[0], img.shape[0] * img.shape[1] / 1e6,
                   name.split(" · ")[0], len(buf) / 1e6, round(dpi)))

    def _full(self, page):
        return imaging.process(page.frame, page.adjust, page.corners)

    def export_image(self):
        page = self.page()
        if page is None:
            self.toast.show_message("Nothing to save", "warn")
            return
        ext, _name, _params = self._format()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", os.path.join(os.path.expanduser("~/Desktop"),
                                             page.name.replace(" ", "-").lower() + ext),
            "JPEG (*.jpg);;PNG (*.png);;TIFF (*.tif)")
        if not path:
            return
        self._start_job("Saving…", self._write_image, page, path)

    def _write_image(self, page, path):
        img = self._full(page)
        chosen, _name, params = self._format()
        ext = os.path.splitext(path)[1].lower()
        if ext != chosen:
            params = ([int(cv2.IMWRITE_JPEG_QUALITY), 98]
                      if ext in (".jpg", ".jpeg") else [])
        if not cv2.imwrite(path, img, params):
            raise RuntimeError("could not write %s" % path)
        return "Saved %s  ·  %d×%d  ·  %.1f MB" % (
            os.path.basename(path), img.shape[1], img.shape[0],
            os.path.getsize(path) / 1e6)

    def export_pdf(self):
        if not self.pages:
            self.toast.show_message("Nothing to save", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF", os.path.join(os.path.expanduser("~/Desktop"), "scan.pdf"),
            "PDF (*.pdf)")
        if not path:
            return
        self._start_job("Building PDF…", self._write_pdf, list(self.pages), path,
                        self.pdf_size.currentText().lower(),
                        self.chk_searchable.isChecked())

    def _write_pdf(self, pages, path, size="a4", searchable=True):
        out = []
        for page in pages:
            img = self._full(page)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                raise RuntimeError("could not encode %s" % page.name)
            words = []
            if page.ocr:
                k = img.shape[1] / float(page.ocr.get("img_w") or img.shape[1])
                words = [{"text": w["text"], "x0": w["x0"] * k, "y0": w["y0"] * k,
                          "x1": w["x1"] * k, "y1": w["y1"] * k}
                         for w in page.ocr.get("words", [])]
            out.append({"jpeg": buf.tobytes(), "width": img.shape[1],
                        "height": img.shape[0], "words": words})
        data = pdfwriter.build(out, page_size=size, title="Scan",
                               searchable=searchable and any(p["words"] for p in out))
        with open(path, "wb") as fh:
            fh.write(data)
        return "Saved %s  ·  %d page%s  ·  %.1f MB" % (
            os.path.basename(path), len(out), "" if len(out) == 1 else "s",
            len(data) / 1e6)

    def export_text(self):
        texts = [(p.name, (p.ocr or {}).get("text", "")) for p in self.pages]
        if not any(t for _n, t in texts):
            self.toast.show_message("No OCR text yet — press Read all", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save text", os.path.join(os.path.expanduser("~/Desktop"), "scan.txt"),
            "Text (*.txt)")
        if not path:
            return
        with open(path, "w") as fh:
            for name, text in texts:
                fh.write("── %s ──\n%s\n\n" % (name, text))
        self.toast.show_message("Saved %s" % os.path.basename(path), "good")

    # ══ OCR ══════════════════════════════════════════════════════

    def run_ocr(self):
        page = self.page()
        if page is None:
            self.toast.show_message("Capture a page first", "warn")
            return
        if not ocr.available():
            self.toast.show_message("Tesseract is not installed", "bad")
            return
        self.lbl_ocr.setText("reading…")
        self._start_job("Reading…", self._read, [page], self.ocr_lang.currentText(),
                        int(self.ocr_psm.currentData()), quiet="ocr")

    def run_ocr_all(self):
        if not self.pages:
            return
        if not ocr.available():
            self.toast.show_message("Tesseract is not installed", "bad")
            return
        self.lbl_ocr.setText("reading %d pages…" % len(self.pages))
        self._start_job("Reading…", self._read, list(self.pages),
                        self.ocr_lang.currentText(), int(self.ocr_psm.currentData()),
                        quiet="ocr")

    def _read(self, pages, lang="eng", psm=3):
        confs = []
        for page in pages:
            img = imaging.process(page.frame, page.adjust, page.corners, max_dim=2600)
            page.ocr = ocr.recognise(img, lang=lang, psm=psm)
            if page.ocr.get("confidence") is not None:
                confs.append(page.ocr["confidence"])
        return confs

    # ══ job plumbing ═════════════════════════════════════════════

    def _job(self, fn, *args):
        """Run `fn` on a worker thread; returns a `connect`-style callable."""
        job = Job()
        self._jobs.append(job)

        def connect(handler):
            def deliver(result, err):
                self._jobs.remove(job)
                handler(result, err)
            job.done.connect(deliver)
            job.run(fn, *args)
        return connect

    def _start_job(self, message, fn, *args, **kw):
        quiet = kw.pop("quiet", False)
        if self.busy:
            self.toast.show_message("Still working on the last one…", "warn")
            return
        self.busy = True
        self.lbl_state.setText(message)
        self.setCursor(Qt.BusyCursor)
        self._job(fn, *args)(lambda r, e: self._job_done(r, e, quiet))

    def _job_done(self, result, err, quiet):
        self.busy = False
        self.unsetCursor()
        self.lbl_state.setText(self.cam.name if self.cam.is_open() else "stopped")
        if err is not None:
            self.toast.show_message(str(err), "bad")
            if quiet == "ocr":
                self.lbl_ocr.setText(str(err))
                self.lbl_ocr.setStyleSheet("color: %s;" % BAD)
            elif quiet == "estimate":
                self.lbl_estimate.setText(str(err))
            return
        if quiet == "estimate":
            self.lbl_estimate.setText(result)
            self.lbl_estimate.setStyleSheet("color: %s;" % FG2)
            return
        if quiet == "ocr":
            page = self.page()
            self.txt_ocr.setPlainText((page.ocr or {}).get("text", "") if page else "")
            confs = result or []
            self.lbl_ocr.setText(
                "%d page%s read · mean confidence %d%%"
                % (len(confs), "" if len(confs) == 1 else "s",
                   round(sum(confs) / len(confs))) if confs else "no text found")
            self.lbl_ocr.setStyleSheet("color: %s;" % (GOOD if confs else WARN))
            return
        self.toast.show_message(result, "good")

    # ══ window ═══════════════════════════════════════════════════

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.toast.reposition()
        # Ask the feed for frames that match the widget, at retina density and
        # never more than the sensor is delivering.
        self.feed.target = max(640, min(PREVIEW_MAX, self.preview.width() * 2))
        if self.mode == "edit":
            self._uncropped = None
            self._queue_render()

    def closeEvent(self, e):
        self._save_state()
        self.timer.stop()
        self.feed.stop()
        self.renderer.stop()
        self.cam.close()
        super().closeEvent(e)


LOG_PATH = os.path.expanduser("~/Library/Logs/overhead-scanner.log")


def _start_logging():
    """Leave a record of a crash, including a native one.

    A segmentation fault in OpenCV or Qt kills the process with no Python
    traceback and, on this machine, no report in DiagnosticReports either —
    which left a real crash with nothing to go on but "it crashed".
    `faulthandler` prints a Python stack from the signal handler, so next time
    there is something to read.
    """
    try:
        fh = open(LOG_PATH, "a", buffering=1)
    except OSError:
        return None
    fh.write("\n=== started %s ===\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    faulthandler.enable(fh)

    def on_error(kind, value, tb):
        traceback.print_exception(kind, value, tb, file=fh)
        traceback.print_exception(kind, value, tb, file=sys.stderr)

    sys.excepthook = on_error
    threading.excepthook = lambda a: on_error(a.exc_type, a.exc_value, a.exc_traceback)
    return fh


def main():
    _start_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("Overhead Scanner")
    app.setStyleSheet(QSS)
    win = App()
    win.show()
    win.raise_()
    win.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
