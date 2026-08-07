#!/usr/bin/env python3
"""Overhead Scanner — desktop app.

    python3 scanner.py

The camera runs at its full sensor resolution the whole time and a capture is
the frame you were already looking at. Preview and capture therefore cannot
disagree about resolution or field of view, because they are the same pixels.
"""

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import camera
import detect
import imaging
import ocr
import pdfwriter

BG      = "#0d1014"
PANEL   = "#141920"
LINE    = "#262e3a"
FG      = "#e6ecf3"
FG2     = "#97a3b4"
FG3     = "#63707f"
ACCENT  = "#4da3ff"
GOOD    = "#35d39a"
BAD     = "#ff6b6b"

PREVIEW_FPS = 8
DETECT_EVERY = 0.45          # seconds between live detections
THUMB_W = 150


class Page:
    __slots__ = ("frame", "corners", "adjust", "thumb", "name", "ocr")

    def __init__(self, frame, corners, adjust, name):
        self.frame = frame           # full-resolution BGR, pristine
        self.corners = corners
        self.adjust = adjust
        self.thumb = None
        self.name = name
        self.ocr = None


class App:
    def __init__(self, root):
        self.root = root
        root.title("Overhead Scanner")
        root.configure(bg=BG)
        root.geometry("1400x880")
        root.minsize(1100, 700)

        self.cam = camera.Camera()
        self.pages = []
        self.current = -1
        self.mode = "live"           # live | edit
        self.corner_mode = False
        self.live_quad = None
        self._quad_smooth = None
        self._quad_miss = 0
        self._last_detect = 0.0
        self._preview_img = None
        self._drag_corner = -1
        self._view = None            # (x, y, w, h) of the image inside the canvas
        self._busy = False
        self._auto_last = 0.0
        self._scene_dirty = False
        self._still_since = 0.0
        self._prev_small = None
        self._render_token = 0
        self._edit_cache = None

        self.var = {}
        self._build_ui()
        self._tick()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── construction ─────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TScale", background=PANEL, troughcolor=LINE)
        style.configure("TCombobox", fieldbackground=LINE, background=LINE)

        top = tk.Frame(self.root, bg=PANEL, height=46)
        top.pack(side="top", fill="x")
        top.pack_propagate(False)

        tk.Label(top, text="Overhead Scanner", bg=PANEL, fg=FG,
                 font=("Helvetica", 13, "bold")).pack(side="left", padx=12)

        tk.Label(top, text="Camera", bg=PANEL, fg=FG2).pack(side="left", padx=(14, 4))
        self.device_box = ttk.Combobox(top, width=22, state="readonly", values=[])
        self.device_box.pack(side="left")

        self.btn_start = tk.Button(top, text="Start camera", command=self.toggle_camera,
                                   bg=ACCENT, fg="#04101f", relief="flat",
                                   activebackground="#7fc0ff", padx=12, pady=3,
                                   font=("Helvetica", 11, "bold"))
        self.btn_start.pack(side="left", padx=10)

        self.lbl_status = tk.Label(top, text="off", bg=PANEL, fg=FG3)
        self.lbl_status.pack(side="left")

        self.lbl_res = tk.Label(top, text="", bg=PANEL, fg=GOOD,
                                font=("Helvetica", 11, "bold"))
        self.lbl_res.pack(side="right", padx=14)

        body = tk.Frame(self.root, bg=BG)
        body.pack(side="top", fill="both", expand=True)

        # ── tray ──
        tray = tk.Frame(body, bg=PANEL, width=THUMB_W + 34)
        tray.pack(side="left", fill="y")
        tray.pack_propagate(False)
        head = tk.Frame(tray, bg=PANEL)
        head.pack(fill="x", pady=6, padx=8)
        self.lbl_pages = tk.Label(head, text="Pages 0", bg=PANEL, fg=FG2)
        self.lbl_pages.pack(side="left")
        self._small_btn(head, "Import", self.import_images).pack(side="right")

        wrap = tk.Frame(tray, bg=PANEL)
        wrap.pack(fill="both", expand=True)
        self.tray_canvas = tk.Canvas(wrap, bg=PANEL, highlightthickness=0,
                                     width=THUMB_W + 16)
        sb = tk.Scrollbar(wrap, orient="vertical", command=self.tray_canvas.yview)
        self.tray_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tray_canvas.pack(side="left", fill="both", expand=True)
        self.tray_inner = tk.Frame(self.tray_canvas, bg=PANEL)
        self.tray_canvas.create_window((0, 0), window=self.tray_inner, anchor="nw")
        self.tray_inner.bind("<Configure>", lambda e: self.tray_canvas.configure(
            scrollregion=self.tray_canvas.bbox("all")))

        # ── stage ──
        stage = tk.Frame(body, bg=BG)
        stage.pack(side="left", fill="both", expand=True)

        bar = tk.Frame(stage, bg=PANEL, height=42)
        bar.pack(side="top", fill="x")
        bar.pack_propagate(False)
        self.btn_live = self._small_btn(bar, "Live", lambda: self.set_mode("live"))
        self.btn_live.pack(side="left", padx=(10, 2), pady=6)
        self.btn_edit = self._small_btn(bar, "Edit", lambda: self.set_mode("edit"))
        self.btn_edit.pack(side="left", padx=2)

        self.btn_capture = tk.Button(bar, text="●  Capture", command=self.capture,
                                     bg="#c8443f", fg="white", relief="flat",
                                     activebackground="#e05a55", padx=16, pady=3,
                                     font=("Helvetica", 11, "bold"))
        self.btn_capture.pack(side="left", padx=16)

        self.var["autocap"] = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="Auto", variable=self.var["autocap"], bg=PANEL,
                       fg=FG2, selectcolor=LINE, activebackground=PANEL,
                       activeforeground=FG).pack(side="left")

        self._small_btn(bar, "Corners", self.toggle_corners).pack(side="right", padx=4)
        self._small_btn(bar, "Detect", self.redetect).pack(side="right", padx=4)
        self._small_btn(bar, "Delete", self.delete_page).pack(side="right", padx=4)

        self.canvas = tk.Canvas(stage, bg="#0a0d11", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        foot = tk.Frame(stage, bg=PANEL, height=26)
        foot.pack(side="bottom", fill="x")
        foot.pack_propagate(False)
        self.lbl_info = tk.Label(foot, text="", bg=PANEL, fg=FG3, anchor="w")
        self.lbl_info.pack(side="left", padx=10)
        self.lbl_hint = tk.Label(foot, text="", bg=PANEL, fg=ACCENT, anchor="e")
        self.lbl_hint.pack(side="right", padx=10)

        # ── inspector ──
        insp = tk.Frame(body, bg=PANEL, width=310)
        insp.pack(side="right", fill="y")
        insp.pack_propagate(False)
        self._build_inspector(insp)

        # Probing each camera's maximum takes a few seconds; let the window
        # appear first so it doesn't look like a hang, then start the camera
        # without waiting to be asked — the app has exactly one purpose.
        self.root.after(150, self._startup)

    def _startup(self):
        self.refresh_devices()
        if self.devices:
            self.root.after(50, self.toggle_camera)

    def _small_btn(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd, bg=LINE, fg=FG,
                         relief="flat", activebackground="#2c3542",
                         activeforeground=FG, padx=10, pady=2)

    def _section(self, parent, title):
        tk.Label(parent, text=title.upper(), bg=PANEL, fg=FG3, anchor="w",
                 font=("Helvetica", 9, "bold")).pack(fill="x", padx=12, pady=(12, 4))
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", padx=12)
        return f

    def _slider(self, parent, key, label, lo, hi, step=1, fmt="%.0f"):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=1)
        var = tk.DoubleVar(value=imaging.DEFAULTS.get(key, 0))
        self.var[key] = var
        lab = tk.Label(row, text=label, bg=PANEL, fg=FG2, anchor="w", width=17,
                       font=("Helvetica", 10))
        lab.pack(side="left")
        val = tk.Label(row, text=fmt % var.get(), bg=PANEL, fg=FG, width=5,
                       anchor="e", font=("Helvetica", 10))
        val.pack(side="right")

        def on_change(_v):
            val.configure(text=fmt % var.get())
            self._apply_adjust(key, var.get())

        s = ttk.Scale(row, from_=lo, to=hi, variable=var, command=on_change,
                      orient="horizontal")
        s.pack(side="right", fill="x", expand=True, padx=6)
        return var

    def _build_inspector(self, insp):
        f = self._section(insp, "Filter")
        grid = tk.Frame(f, bg=PANEL)
        grid.pack(fill="x")
        self.filter_buttons = {}
        names = [("original", "Original"), ("auto", "Auto"), ("color", "Colour doc"),
                 ("gray", "Greyscale"), ("bw", "Black & white"), ("whiteboard", "Whiteboard"),
                 ("ink", "Ink boost"), ("photo", "Photo")]
        for i, (key, label) in enumerate(names):
            b = tk.Button(grid, text=label, relief="flat", bg=LINE, fg=FG2,
                          activebackground="#2c3542", font=("Helvetica", 10),
                          command=lambda k=key: self.set_filter(k))
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
            self.filter_buttons[key] = b
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        f = self._section(insp, "Geometry")
        row = tk.Frame(f, bg=PANEL); row.pack(fill="x")
        self._small_btn(row, "↺ 90", lambda: self.rotate(-90)).pack(side="left", expand=True, fill="x", padx=1)
        self._small_btn(row, "↻ 90", lambda: self.rotate(90)).pack(side="left", expand=True, fill="x", padx=1)
        self._small_btn(row, "Flip H", lambda: self.flip("fliph")).pack(side="left", expand=True, fill="x", padx=1)
        self._small_btn(row, "Flip V", lambda: self.flip("flipv")).pack(side="left", expand=True, fill="x", padx=1)
        self._slider(f, "straighten", "Straighten", -15, 15, fmt="%.1f")

        f = self._section(insp, "Lighting")
        self._slider(f, "flatten", "Shadow removal", 0, 100)
        self._slider(f, "temp", "Temperature", -100, 100)
        self._slider(f, "tint", "Tint", -100, 100)

        f = self._section(insp, "Tone")
        self._slider(f, "exposure", "Exposure", -100, 100)
        self._slider(f, "contrast", "Contrast", -100, 100)
        self._slider(f, "gamma", "Gamma", 0.3, 3.0, fmt="%.2f")
        self._slider(f, "highlights", "Highlights", -100, 100)
        self._slider(f, "shadows", "Shadows", -100, 100)

        f = self._section(insp, "Colour & detail")
        self._slider(f, "saturation", "Saturation", -100, 100)
        self._slider(f, "vibrance", "Vibrance", -100, 100)
        self._slider(f, "denoise", "Denoise", 0, 100)
        self._slider(f, "sharpen", "Sharpen", 0, 150)

        f = self._section(insp, "Black & white")
        self._slider(f, "threshold", "Threshold bias", -50, 50)
        self._slider(f, "window", "Local window", 20, 300)

        f = self._section(insp, "OCR")
        row = tk.Frame(f, bg=PANEL); row.pack(fill="x", pady=2)
        self._small_btn(row, "Read page", self.run_ocr).pack(side="left", expand=True, fill="x", padx=1)
        self._small_btn(row, "Read all", self.run_ocr_all).pack(side="left", expand=True, fill="x", padx=1)
        self.lbl_ocr = tk.Label(f, text="", bg=PANEL, fg=FG3, anchor="w",
                                wraplength=280, justify="left", font=("Helvetica", 9))
        self.lbl_ocr.pack(fill="x", pady=2)
        self.txt_ocr = tk.Text(f, height=7, bg=LINE, fg=FG, insertbackground=FG,
                               relief="flat", wrap="word", font=("Menlo", 9))
        self.txt_ocr.pack(fill="x", pady=2)
        if not ocr.available():
            self.lbl_ocr.configure(text="Tesseract not installed — text export and "
                                        "searchable PDF are unavailable. brew install tesseract",
                                   fg="#ffb648")

        f = self._section(insp, "Export")
        row = tk.Frame(f, bg=PANEL); row.pack(fill="x", pady=2)
        self._small_btn(row, "Save image", self.export_image).pack(side="left", expand=True, fill="x", padx=1)
        self._small_btn(row, "Save PDF", self.export_pdf).pack(side="left", expand=True, fill="x", padx=1)
        row2 = tk.Frame(f, bg=PANEL); row2.pack(fill="x", pady=2)
        self._small_btn(row2, "Apply to all pages", self.apply_all).pack(side="left", expand=True, fill="x", padx=1)
        self._small_btn(row2, "Save text", self.export_text).pack(side="left", expand=True, fill="x", padx=1)
        self.lbl_export = tk.Label(f, text="", bg=PANEL, fg=FG3, anchor="w",
                                   wraplength=280, justify="left",
                                   font=("Helvetica", 9))
        self.lbl_export.pack(fill="x", pady=4)

    # ── camera ───────────────────────────────────────────────────

    def refresh_devices(self):
        self.lbl_status.configure(text="scanning…", fg=FG3)
        self.root.update_idletasks()
        devs = camera.Camera.list_devices()
        # Highest-resolution device first: on a laptop with a built-in webcam
        # the document camera is rarely index 0, and picking by index would
        # quietly hand you 1080p from the wrong camera.
        devs.sort(key=lambda d: d["max_width"] * d["max_height"], reverse=True)
        self.devices = devs
        labels = ["Camera %d — up to %d×%d" % (d["index"], d["max_width"], d["max_height"])
                  for d in devs]
        self.device_box.configure(values=labels)
        if labels:
            self.device_box.current(0)
            self.lbl_status.configure(text="ready", fg=FG3)
        else:
            self.lbl_status.configure(text="no camera", fg=BAD)

    def toggle_camera(self):
        if self.cam.is_open():
            self.cam.close()
            self.btn_start.configure(text="Start camera")
            self.lbl_status.configure(text="off", fg=FG3)
            self.lbl_res.configure(text="")
            return
        if not self.devices:
            self.refresh_devices()
            if not self.devices:
                return
        sel = self.devices[max(0, self.device_box.current())]
        idx = sel["index"]
        self.lbl_status.configure(text="opening…", fg=FG3)
        self.root.update_idletasks()
        detect.reset_sticky()
        if not self.cam.open(idx, prefer=(sel["max_width"], sel["max_height"])):
            self.lbl_status.configure(text="failed", fg=BAD)
            messagebox.showerror("Camera", self.cam.error or "could not open the camera")
            return
        self.btn_start.configure(text="Stop camera")
        self.lbl_status.configure(text="live", fg=GOOD)
        self.lbl_res.configure(text="%d × %d" % (self.cam.width, self.cam.height))
        self.set_mode("live")

    # ── the loop ─────────────────────────────────────────────────

    def _tick(self):
        try:
            if self.mode == "live" and self.cam.is_open():
                self._live_frame()
        except Exception as exc:                       # keep the UI alive
            self.lbl_hint.configure(text="preview error: %s" % exc)
        self.root.after(int(1000 / PREVIEW_FPS), self._tick)

    def _live_frame(self):
        frame = self.cam.latest()
        if frame is None:
            return
        now = time.time()
        if now - self._last_detect > DETECT_EVERY:
            self._last_detect = now
            self._update_quad(detect.detect(frame, sticky=True))
            self._auto_capture(frame)

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        small = self._fit_to_canvas(frame, cw, ch)
        self._show(small)
        if self.live_quad is not None:
            self._draw_quad(self.live_quad, handles=False)
        h, w = frame.shape[:2]
        out = detect.output_size(self.live_quad, w, h) if self.live_quad is not None else (w, h)
        dpi = int(round(max(out) / 11.69))
        self.lbl_info.configure(
            text="live  %d×%d   crop → %d×%d  ≈%d dpi" %
                 (w, h, out[0], out[1], dpi))
        self.lbl_hint.configure(
            text="page found" if self.live_quad is not None else "no page outline — whole frame")

    def _update_quad(self, q):
        """Ease towards each reading and hold through a few misses, so the
        outline stops flickering between frames."""
        if q is None:
            self._quad_miss += 1
            if self._quad_miss > 3:
                self._quad_smooth = None
        else:
            self._quad_miss = 0
            if self._quad_smooth is None:
                self._quad_smooth = q
            else:
                moved = float(np.abs(q - self._quad_smooth).max())
                a = 1.0 if moved > 0.05 else 0.4
                self._quad_smooth = self._quad_smooth + (q - self._quad_smooth) * a
        self.live_quad = self._quad_smooth

    def _auto_capture(self, frame):
        if not self.var["autocap"].get():
            return
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 48),
                           interpolation=cv2.INTER_AREA).astype(np.int16)
        if self._prev_small is not None:
            motion = float(np.abs(small - self._prev_small).mean()) / 255.0
            now = time.time()
            if motion >= 0.03:
                self._scene_dirty = True
                self._still_since = 0.0
            elif self._scene_dirty:
                if self._still_since == 0.0:
                    self._still_since = now
                elif now - self._still_since > 0.6 and now - self._auto_last > 1.5:
                    self._auto_last = now
                    self._scene_dirty = False
                    self._still_since = 0.0
                    self.capture()
        self._prev_small = small

    # ── display helpers ──────────────────────────────────────────

    def _fit_to_canvas(self, img, cw, ch):
        h, w = img.shape[:2]
        k = min((cw - 20) / float(w), (ch - 20) / float(h))
        k = max(0.02, min(k, 1.0))
        dw, dh = max(1, int(w * k)), max(1, int(h * k))
        # Straight INTER_AREA: measured at ~16 ms for a 16 MP frame, and
        # striding first to "save" work is slower, because a strided view is
        # non-contiguous and OpenCV copies it before resizing.
        small = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
        self._view = ((cw - dw) // 2, (ch - dh) // 2, dw, dh)
        return small

    def _show(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._preview_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        x, y, _, _ = self._view
        self.canvas.create_image(x, y, image=self._preview_img, anchor="nw")

    def _draw_quad(self, quad, handles):
        if self._view is None or quad is None:
            return
        x, y, w, h = self._view
        pts = [(x + p[0] * w, y + p[1] * h) for p in quad]
        colour = ACCENT if handles else GOOD
        for i in range(4):
            a, b = pts[i], pts[(i + 1) % 4]
            self.canvas.create_line(a[0], a[1], b[0], b[1], fill=colour, width=2)
        if handles:
            for i, (px, py) in enumerate(pts):
                r = 7
                fill = ACCENT if i == self._drag_corner else "white"
                self.canvas.create_oval(px - r, py - r, px + r, py + r,
                                        fill=fill, outline=BG, width=2)

    # ── pages ────────────────────────────────────────────────────

    def cur(self):
        return self.pages[self.current] if 0 <= self.current < len(self.pages) else None

    def capture(self):
        if not self.cam.is_open():
            messagebox.showinfo("Capture", "Start the camera first.")
            return
        frame = self.cam.grab()
        if frame is None:
            return
        adjust = dict(self.cur().adjust) if self.cur() else imaging.new_adjust()
        for k in ("rotate", "fliph", "flipv", "straighten"):
            adjust[k] = imaging.DEFAULTS[k]
        # The outline on screen is the crop: same pixels, so it transfers exactly.
        quad = None if self.live_quad is None else np.array(self.live_quad, copy=True)
        page = Page(frame, quad, adjust, "page %d" % (len(self.pages) + 1))
        self.pages.append(page)
        self.current = len(self.pages) - 1
        self._edit_cache = None
        self.set_mode("edit")
        self.refresh_tray()
        h, w = frame.shape[:2]
        out = detect.output_size(quad, w, h) if quad is not None else (w, h)
        cover = int(round(100 * (out[0] * out[1] / float(w * h)) ** 0.5))
        note = ""
        if cover < 55:
            note = "  — page fills only %d%% of the frame, move the camera closer" % cover
        self.lbl_export.configure(text="captured %d×%d → %d×%d%s"
                                       % (w, h, out[0], out[1], note))

    def import_images(self):
        paths = filedialog.askopenfilenames(
            title="Import images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All", "*.*")])
        for p in paths:
            data = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                continue
            page = Page(img, detect.detect(img), imaging.new_adjust(),
                        os.path.basename(p))
            self.pages.append(page)
            self.current = len(self.pages) - 1
        if paths:
            self._edit_cache = None
            self.set_mode("edit")
            self.refresh_tray()

    def delete_page(self):
        if self.cur() is None:
            return
        del self.pages[self.current]
        self.current = min(self.current, len(self.pages) - 1)
        self._edit_cache = None
        if not self.pages:
            self.set_mode("live")
        self.refresh_tray()
        self.render_edit()

    def select_page(self, i):
        self.current = i
        self._edit_cache = None
        self.corner_mode = False
        self.set_mode("edit")
        self._sync_sliders()
        self.refresh_tray()

    def refresh_tray(self):
        for w in self.tray_inner.winfo_children():
            w.destroy()
        self.lbl_pages.configure(text="Pages %d" % len(self.pages))
        for i, page in enumerate(self.pages):
            if page.thumb is None:
                small = imaging.fit(page.frame, 420)
                out = imaging.process(small, page.adjust, page.corners, THUMB_W)
                rgb = cv2.cvtColor(imaging.fit(out, THUMB_W), cv2.COLOR_BGR2RGB)
                page.thumb = ImageTk.PhotoImage(Image.fromarray(rgb))
            border = ACCENT if i == self.current else PANEL
            holder = tk.Frame(self.tray_inner, bg=border, padx=2, pady=2)
            holder.pack(fill="x", padx=6, pady=4)
            lbl = tk.Label(holder, image=page.thumb, bd=0, bg=PANEL)
            lbl.pack()
            lbl.bind("<Button-1>", lambda e, k=i: self.select_page(k))
            tk.Label(holder, text="%d" % (i + 1), bg=border, fg=FG,
                     font=("Helvetica", 9)).pack()

    # ── editing ──────────────────────────────────────────────────

    def set_mode(self, mode):
        if mode == "edit" and self.cur() is None:
            mode = "live"
        self.mode = mode
        self.btn_live.configure(bg=ACCENT if mode == "live" else LINE,
                                fg="#04101f" if mode == "live" else FG)
        self.btn_edit.configure(bg=ACCENT if mode == "edit" else LINE,
                                fg="#04101f" if mode == "edit" else FG)
        if mode == "edit":
            self._sync_sliders()
            self.render_edit()

    def _sync_sliders(self):
        page = self.cur()
        a = page.adjust if page else imaging.new_adjust()
        for key, var in self.var.items():
            if key in a and isinstance(var, tk.DoubleVar):
                var.set(float(a[key]))
        for key, btn in self.filter_buttons.items():
            on = a.get("filter") == key
            btn.configure(bg=ACCENT if on else LINE, fg="#04101f" if on else FG2)

    def _apply_adjust(self, key, value):
        page = self.cur()
        if page is None:
            return
        if page.adjust.get(key) == value:
            return
        page.adjust[key] = value
        if key not in imaging.GEOMETRY_KEYS:
            page.adjust["filter"] = "custom"
            for k, b in self.filter_buttons.items():
                b.configure(bg=LINE, fg=FG2)
        page.thumb = None
        self.render_edit(fast=True)

    def set_filter(self, name):
        page = self.cur()
        if page is None:
            return
        imaging.set_filter(page.adjust, name)
        page.thumb = None
        self._sync_sliders()
        self.render_edit()
        self.refresh_tray()

    def rotate(self, delta):
        page = self.cur()
        if page is None:
            return
        page.adjust["rotate"] = (int(page.adjust["rotate"]) + delta) % 360
        page.thumb = None
        self.render_edit()

    def flip(self, key):
        page = self.cur()
        if page is None:
            return
        page.adjust[key] = not page.adjust.get(key)
        page.thumb = None
        self.render_edit()

    def apply_all(self):
        page = self.cur()
        if page is None:
            return
        n = 0
        for other in self.pages:
            if other is page:
                continue
            keep = {k: other.adjust[k] for k in ("rotate", "fliph", "flipv", "straighten")}
            other.adjust = dict(page.adjust)
            other.adjust.update(keep)
            other.thumb = None
            n += 1
        self.refresh_tray()
        self.lbl_export.configure(text="applied to %d other page%s" % (n, "" if n == 1 else "s"))

    def toggle_corners(self):
        if self.cur() is None:
            return
        self.corner_mode = not self.corner_mode
        self.set_mode("edit")
        self.render_edit()

    def redetect(self):
        page = self.cur()
        if page is None:
            return
        q = detect.detect(page.frame)
        page.corners = q
        page.thumb = None
        self.lbl_hint.configure(text="page edges detected" if q is not None
                                else "no page found — using the whole frame")
        self.render_edit()
        self.refresh_tray()

    def render_edit(self, fast=False):
        page = self.cur()
        if page is None:
            self.canvas.delete("all")
            self.lbl_info.configure(text="")
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        if self.corner_mode:
            small = self._fit_to_canvas(page.frame, cw, ch)
            self._show(small)
            quad = page.corners if page.corners is not None else detect.full_frame()
            self._draw_quad(quad, handles=True)
            self.lbl_hint.configure(text="drag the corners — Corners again when done")
        else:
            cap = int(max(cw, ch) * (0.8 if fast else 1.4))
            cap = max(700, min(cap, 2600))
            src = imaging.fit(page.frame, cap)
            # corners are normalised, so they survive the downscale unchanged
            out = imaging.process(src, page.adjust, page.corners, cap)
            small = self._fit_to_canvas(out, cw, ch)
            self._show(small)
            self.lbl_hint.configure(text="")

        h, w = page.frame.shape[:2]
        fw, fh = imaging.target_size(page.adjust, page.corners, w, h)
        self.lbl_info.configure(
            text="page %d/%d   source %d×%d   output %d×%d%s"
                 % (self.current + 1, len(self.pages), w, h, fw, fh,
                    "   cropped" if page.corners is not None else ""))

    # ── corner dragging ──────────────────────────────────────────

    def _canvas_to_norm(self, ex, ey):
        if self._view is None:
            return None
        x, y, w, h = self._view
        return (float(np.clip((ex - x) / float(w), 0, 1)),
                float(np.clip((ey - y) / float(h), 0, 1)))

    def _on_press(self, ev):
        if self.mode != "edit" or not self.corner_mode or self.cur() is None:
            return
        page = self.cur()
        if page.corners is None:
            page.corners = detect.full_frame()
        p = self._canvas_to_norm(ev.x, ev.y)
        if p is None:
            return
        x, y, w, h = self._view
        best, bestd = -1, 22.0
        for i, c in enumerate(page.corners):
            d = np.hypot((c[0] - p[0]) * w, (c[1] - p[1]) * h)
            if d < bestd:
                best, bestd = i, d
        self._drag_corner = best
        if best >= 0:
            self.render_edit()

    def _on_drag(self, ev):
        if self._drag_corner < 0:
            return
        page = self.cur()
        p = self._canvas_to_norm(ev.x, ev.y)
        if p is None:
            return
        page.corners[self._drag_corner] = p
        self.render_edit()

    def _on_release(self, _ev):
        if self._drag_corner < 0:
            return
        self._drag_corner = -1
        page = self.cur()
        if page is not None and page.corners is not None:
            page.corners = detect._order(page.corners) / 1.0
            page.thumb = None
        self.render_edit()
        self.refresh_tray()

    # ── export ───────────────────────────────────────────────────

    def _render_full(self, page):
        """Full resolution, from the pristine frame. Never downscaled."""
        return imaging.process(page.frame, page.adjust, page.corners, None)

    def export_image(self):
        page = self.cur()
        if page is None:
            messagebox.showinfo("Export", "No page selected.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg", initialfile="scan-%s.jpg" % time.strftime("%Y%m%d-%H%M%S"),
            filetypes=[("JPEG (quality 98)", "*.jpg"), ("PNG (lossless)", "*.png"),
                       ("TIFF (lossless)", "*.tif")])
        if not path:
            return
        self._set_busy(True)
        try:
            out = self._render_full(page)
            ext = os.path.splitext(path)[1].lower()
            if ext in (".jpg", ".jpeg"):
                params = [int(cv2.IMWRITE_JPEG_QUALITY), 98]
            elif ext == ".png":
                params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
            else:
                params = []
            ok, buf = cv2.imencode(ext if ext else ".jpg", out, params)
            if not ok:
                raise RuntimeError("could not encode %s" % ext)
            buf.tofile(path)
            self.lbl_export.configure(
                text="saved %d×%d  %.1f MB" % (out.shape[1], out.shape[0],
                                                    os.path.getsize(path) / 1048576.0))
        except Exception as exc:
            messagebox.showerror("Export", str(exc))
        finally:
            self._set_busy(False)

    def export_pdf(self):
        if not self.pages:
            messagebox.showinfo("Export", "No pages to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", initialfile="scan-%s.pdf" % time.strftime("%Y%m%d-%H%M%S"),
            filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        self._set_busy(True)
        try:
            entries = []
            for i, page in enumerate(self.pages):
                self.lbl_export.configure(text="rendering %d/%d…" % (i + 1, len(self.pages)))
                self.root.update_idletasks()
                out = self._render_full(page)
                ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                if not ok:
                    continue
                entry = {"jpeg": buf.tobytes(),
                         "width": out.shape[1], "height": out.shape[0]}
                if page.ocr and page.ocr.get("words"):
                    # OCR ran on a render of the same page, so a single scale
                    # factor maps its boxes onto this one.
                    k = out.shape[1] / float(max(1, page.ocr.get("img_w", out.shape[1])))
                    entry["words"] = [
                        {"text": w["text"], "x0": w["x0"] * k, "y0": w["y0"] * k,
                         "x1": w["x1"] * k, "y1": w["y1"] * k}
                        for w in page.ocr["words"]]
                entries.append(entry)
            data = pdfwriter.build(entries, page_size="a4",
                                   title="Scan %s" % time.strftime("%Y-%m-%d"))
            with open(path, "wb") as fh:
                fh.write(data)
            self.lbl_export.configure(text="saved %d page%s  %.1f MB"
                                           % (len(entries), "" if len(entries) == 1 else "s",
                                              len(data) / 1048576.0))
        except Exception as exc:
            messagebox.showerror("Export", str(exc))
        finally:
            self._set_busy(False)

    # ── OCR ──────────────────────────────────────────────────────

    def run_ocr(self, page=None, quiet=False):
        page = page or self.cur()
        if page is None:
            messagebox.showinfo("OCR", "No page selected.")
            return False
        if not ocr.available():
            messagebox.showinfo("OCR", ocr.install_hint())
            return False
        self._set_busy(True)
        self.lbl_ocr.configure(text="reading…", fg=FG3)
        self.root.update_idletasks()
        try:
            # Read the processed page, not the raw frame: the filters are what
            # make the text legible in the first place.
            img = self._render_full(page)
            res = ocr.recognise(img)
            page.ocr = res
            conf = res["confidence"]
            self.lbl_ocr.configure(
                text="%d words%s" % (len(res["words"]),
                                     "" if conf is None else "   %d%% confidence" % round(conf)),
                fg=GOOD)
            if not quiet:
                self.txt_ocr.delete("1.0", "end")
                self.txt_ocr.insert("1.0", res["text"])
            return True
        except Exception as exc:
            self.lbl_ocr.configure(text=str(exc)[:160], fg=BAD)
            if not quiet:
                messagebox.showerror("OCR", str(exc))
            return False
        finally:
            self._set_busy(False)

    def run_ocr_all(self):
        if not self.pages:
            return
        for i, page in enumerate(self.pages):
            self.lbl_ocr.configure(text="reading page %d/%d…" % (i + 1, len(self.pages)))
            self.root.update_idletasks()
            if not self.run_ocr(page, quiet=True):
                return
        done = sum(1 for p in self.pages if p.ocr)
        self.lbl_ocr.configure(text="read %d page%s" % (done, "" if done == 1 else "s"), fg=GOOD)

    def export_text(self):
        texts = [(p.ocr or {}).get("text", "") for p in self.pages]
        if not any(t.strip() for t in texts):
            messagebox.showinfo("Text", "No recognised text yet — run Read all first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", initialfile="scan-%s.txt" % time.strftime("%Y%m%d-%H%M%S"),
            filetypes=[("Text", "*.txt")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            for i, t in enumerate(texts):
                fh.write("--- page %d ---\n%s\n\n" % (i + 1, t))
        self.lbl_export.configure(text="saved text for %d page(s)" % len(texts))

    def _set_busy(self, busy):
        self._busy = busy
        self.root.configure(cursor="watch" if busy else "")
        self.root.update_idletasks()

    def _on_close(self):
        self.cam.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    # Tk opens behind whatever is in front on macOS; ask for focus once.
    root.lift()
    root.attributes("-topmost", True)
    root.after(400, lambda: root.attributes("-topmost", False))
    try:
        root.createcommand("::tk::mac::ReopenApplication", root.lift)
    except tk.TclError:
        pass
    root.mainloop()


if __name__ == "__main__":
    main()
