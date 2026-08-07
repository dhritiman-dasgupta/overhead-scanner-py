#!/usr/bin/env python3
"""Overhead Scanner — desktop app for a document camera.

    python3 scanner.py

The camera runs at its full sensor resolution the whole time and a capture is
the frame you were already looking at. Preview and capture therefore cannot
disagree about resolution or field of view, because they are the same pixels.
"""

import os
import sys
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox,
                               QScrollArea, QTabWidget, QTextEdit, QVBoxLayout,
                               QWidget)

import camera
import detect
import imaging
import ocr
import pdfwriter
import qtui
from qtui import (ACCENT, BAD, BG, FG, FG2, FG3, GOOD, LINE, PANEL, PANEL2, QSS,
                  WARN, PreviewView, Section, SliderRow, Toast, button, hair,
                  label, spacer, to_qimage)

PREVIEW_MS   = 70            # display refresh; the camera itself runs ~10 fps
DETECT_EVERY = 0.40          # seconds between live page detections
EDIT_MAX     = 1700          # long edge the editor previews at
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
        self._quad_miss = 0
        self._last_detect = 0.0
        self._last_seq = -1
        self._prev_small = None
        self._still_since = 0.0
        self._auto_last = 0.0
        self._jobs = []
        self.sliders = {}
        self.filter_buttons = {}

        self._build()
        self._sync_controls()
        self.set_mode("live")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(PREVIEW_MS)

        self.repro = QTimer(self)
        self.repro.setSingleShot(True)
        self.repro.timeout.connect(self._render_page)

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
        sec = Section("All pages")
        sec.add(button("Apply these settings to every page", self.apply_all))
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
        sec.add(button("Save image…", self.export_image, kind="primary",
                       tip="⌘S"))
        note = label("Always the full resolution of the crop. JPEG is written at "
                     "quality 98; choose PNG or TIFF for no compression at all.",
                     "note")
        note.setWordWrap(True)
        sec.add(note)
        col.addWidget(sec)

        col.addWidget(hair())
        sec = Section("All pages")
        sec.add(button("Save PDF…", self.export_pdf, kind="primary", tip="⌘P"))
        sec.add(button("Save text…", self.export_text))
        col.addWidget(sec)

        col.addWidget(hair())
        sec = Section("OCR")
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(button("Read page", self.run_ocr))
        row.addWidget(button("Read all", self.run_ocr_all))
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
        act("Ctrl+S", self.export_image)
        act("Ctrl+P", self.export_pdf)
        act("Ctrl+R", self.toggle_camera)
        act("Left", lambda: self.step_page(-1))
        act("Right", lambda: self.step_page(1))

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
        if was is not None:
            i = self.device_box.findData(was)
            if i >= 0:
                self.device_box.setCurrentIndex(i)
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
        self._last_seq = -1
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

    def _tick(self):
        if self.mode != "live":
            return
        # Only react to a camera that *was* running and has died. Testing
        # `cam.error` alone also fires while a failed open is being retried,
        # and overwrote the reason on screen with a bare "stopped".
        if self.live and not self.busy and self.cam.error and not self.cam.is_open():
            msg = self.cam.error
            self.stop_camera()
            self.toast.show_message(msg, "bad")
            return
        frame = self.cam.latest()
        if frame is None:
            return
        seq = self.cam.sequence()
        if seq == self._last_seq:
            return
        self._last_seq = seq

        now = time.time()
        if now - self._last_detect >= DETECT_EVERY:
            self._last_detect = now
            small = imaging.fit(frame, 900)
            quad = detect.detect(small, sticky=True)
            if quad is not None:
                self.live_quad = quad
                self._quad_miss = 0
            else:
                self._quad_miss += 1
                if self._quad_miss > 3:
                    self.live_quad = None
            self._check_still(small)

        view = imaging.fit(frame, max(640, self.preview.width() * 2))
        self.preview.set_image(to_qimage(view))
        self.preview.set_quad(self.live_quad)
        self._live_info(frame)

    def _live_info(self, frame):
        h, w = frame.shape[:2]
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

    def _check_still(self, small):
        """Auto capture fires once the scene stops moving, not on a timer.

        A page turn is a burst of change followed by stillness; shooting on
        stillness is what makes a stack of pages a rhythm rather than a chore.
        """
        if not self.auto:
            self._prev_small = None
            return
        g = cv2.cvtColor(imaging.fit(small, 240), cv2.COLOR_BGR2GRAY)
        now = time.time()
        if self._prev_small is not None and self._prev_small.shape == g.shape:
            diff = float(np.mean(cv2.absdiff(g, self._prev_small)))
            if diff > 2.4:
                self._still_since = now
            elif (now - self._still_since > 1.1 and now - self._auto_last > 2.6
                    and self.live_quad is not None):
                self._auto_last = now
                self.capture()
        self._prev_small = g

    def toggle_auto(self):
        self.auto = not self.auto
        self.btn_auto.setChecked(self.auto)
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
            self._last_seq = -1
        else:
            self._render_page()

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
        self._render_page()

    def _corners_dragged(self, quad):
        page = self.page()
        if page is None:
            return
        page.corners = np.asarray(quad, dtype=np.float32)
        self._render_page()

    def redetect(self):
        page = self.page()
        if page is None:
            self.toast.show_message("Capture a page first", "warn")
            return
        quad = detect.detect(imaging.fit(page.frame, 900))
        page.corners = quad
        self._render_page()
        self.toast.show_message("Page found" if quad is not None else
                                "No page found — use Corners",
                                "good" if quad is not None else "warn")

    # ══ rendering the editor ═════════════════════════════════════

    def _render_page(self):
        page = self.page()
        if page is None or self.mode != "edit":
            return
        editing = self.preview.is_editable()
        if editing:
            # While the corners are being dragged, show the uncropped frame —
            # you cannot place a corner on a picture the corners already cut.
            view = imaging.fit(page.frame, max(700, self.preview.width() * 2))
            self.preview.set_image(to_qimage(view))
            self.preview.set_quad(page.corners if page.corners is not None
                                  else detect.full_frame())
        else:
            limit = max(700, min(EDIT_MAX, self.preview.width() * 2))
            try:
                img = imaging.process(page.frame, page.adjust, page.corners,
                                      max_dim=limit)
            except Exception as exc:                # noqa: BLE001
                self.toast.show_message("Preview failed: %s" % exc, "bad")
                return
            self.preview.set_image(to_qimage(img))
            self.preview.set_quad(None)
        self._page_info(page)
        self._refresh_thumb(self.current)

    def _page_info(self, page):
        fh, fw = page.frame.shape[:2]
        ow, oh = imaging.target_size(page.adjust, page.corners, fw, fh)
        dpi = max(ow, oh) / A4_INCHES
        self.lbl_info.setText("%s  ·  source %d×%d  ·  output %d×%d  ·  ≈%d dpi"
                              % (page.name, fw, fh, ow, oh, round(dpi)))
        self.lbl_hint.setText("drag the corners" if self.preview.is_editable() else "")
        self.lbl_hint.setStyleSheet("color: %s;" % ACCENT)

    def _queue_render(self):
        self.repro.start(110)

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
        for k, b in self.filter_buttons.items():
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

    def _outsize_changed(self, i):
        page = self.page()
        if page is None:
            return
        page.adjust["outsize"] = OUTSIZES[i][0]
        self._queue_render()

    def rotate(self, deg):
        page = self.page()
        if page is None:
            return
        page.adjust["rotate"] = (page.adjust.get("rotate", 0) + deg) % 360
        self._render_page()

    def flip(self, key):
        page = self.page()
        if page is None:
            return
        page.adjust[key] = not page.adjust.get(key, False)
        self._render_page()

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

    # ══ export ═══════════════════════════════════════════════════

    def _full(self, page):
        return imaging.process(page.frame, page.adjust, page.corners)

    def export_image(self):
        page = self.page()
        if page is None:
            self.toast.show_message("Nothing to save", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", os.path.join(os.path.expanduser("~/Desktop"),
                                             page.name.replace(" ", "-").lower() + ".jpg"),
            "JPEG (*.jpg);;PNG (*.png);;TIFF (*.tif)")
        if not path:
            return
        self._start_job("Saving…", self._write_image, page, path)

    def _write_image(self, page, path):
        img = self._full(page)
        ext = os.path.splitext(path)[1].lower()
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 98] if ext in (".jpg", ".jpeg") else []
        if not cv2.imwrite(path, img, params):
            raise RuntimeError("could not write %s" % path)
        return "%s  ·  %d×%d  ·  %.1f MB" % (os.path.basename(path), img.shape[1],
                                             img.shape[0],
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
        self._start_job("Building PDF…", self._write_pdf, list(self.pages), path)

    def _write_pdf(self, pages, path):
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
        data = pdfwriter.build(out, page_size="a4", title="Scan",
                               searchable=any(p["words"] for p in out))
        with open(path, "wb") as fh:
            fh.write(data)
        return "%s  ·  %d page%s  ·  %.1f MB" % (
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
        self._start_job("Reading…", self._read, [page], quiet=True)

    def run_ocr_all(self):
        if not self.pages:
            return
        if not ocr.available():
            self.toast.show_message("Tesseract is not installed", "bad")
            return
        self.lbl_ocr.setText("reading %d pages…" % len(self.pages))
        self._start_job("Reading…", self._read, list(self.pages), quiet=True)

    def _read(self, pages):
        confs = []
        for page in pages:
            img = imaging.process(page.frame, page.adjust, page.corners, max_dim=2600)
            page.ocr = ocr.recognise(img)
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
            self.lbl_ocr.setText(str(err))
            return
        if quiet:
            page = self.page()
            text = (page.ocr or {}).get("text", "") if page else ""
            self.txt_ocr.setPlainText(text)
            confs = result or []
            self.lbl_ocr.setText(
                "%d page%s read · mean confidence %d%%"
                % (len(confs), "" if len(confs) == 1 else "s",
                   round(sum(confs) / len(confs))) if confs else "no text found")
            self.lbl_ocr.setStyleSheet("color: %s;" % (GOOD if confs else WARN))
            return
        self.toast.show_message("Saved %s" % result, "good")

    # ══ window ═══════════════════════════════════════════════════

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.toast.reposition()
        if self.mode == "edit":
            self._queue_render()

    def closeEvent(self, e):
        self.timer.stop()
        self.cam.close()
        super().closeEvent(e)


def main():
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
