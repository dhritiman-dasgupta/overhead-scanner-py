"""Theme and the custom widgets the app is built from.

Qt rather than Tkinter for a reason worth recording: the only Python on this
machine is Apple's, which ships Tk 8.5.9, and on macOS 26 that Tk does not
repaint — the window comes up white, `tk.Button` ignores every colour you give
it, and no amount of application code changes either. Qt draws everything
itself, so the app looks and behaves the same wherever it runs.

Dark on purpose: the operator judges paper whiteness against the screen, and a
bright UI reflects in glossy pages.
"""

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QColor, QFont, QIcon, QImage, QPainter, QPainterPath,
                           QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QSlider, QVBoxLayout, QWidget)

BG      = "#0e1116"
PANEL   = "#161a21"
PANEL2  = "#242b37"
PANEL3  = "#303a49"
LINE    = "#2b323d"
FG      = "#e9eef5"
FG2     = "#9aa7b8"
FG3     = "#66748a"
ACCENT  = "#4da3ff"
ACCENT2 = "#7ebcff"
GOOD    = "#35d39a"
WARN    = "#ffb648"
BAD     = "#ff6b6b"
REC     = "#e0504b"

QSS = """
QWidget { background: %(BG)s; color: %(FG)s;
          font-family: "SF Pro Text", "Helvetica Neue", Helvetica, Arial;
          font-size: 12px; }
/* Labels must not paint their own ground, or every caption sitting on a panel
   shows up as a dark rectangle against it. */
QLabel { background: transparent; }
QToolTip { background: %(PANEL3)s; color: %(FG)s; border: 1px solid %(LINE)s;
           padding: 4px 6px; }

#topbar, #panel, #bar { background: %(PANEL)s; }
QScrollArea, QScrollArea > QWidget > QWidget { background: %(PANEL)s; }
#hair  { background: %(LINE)s; }
#stage { background: #080b0e; }

#title    { font-size: 14px; font-weight: 600; }
#muted    { color: %(FG3)s; }
#sublabel { color: %(FG2)s; font-size: 11px; }
#heading  { color: %(FG3)s; font-size: 10px; font-weight: 700;
            letter-spacing: 1px; }
#badge    { color: %(GOOD)s; font-size: 12px; font-weight: 600; }
#note     { color: %(FG3)s; font-size: 11px; }

QPushButton {
    background: %(PANEL2)s; color: %(FG)s; border: none; border-radius: 6px;
    padding: 7px 12px; font-size: 12px;
}
QPushButton:hover    { background: %(PANEL3)s; }
QPushButton:pressed  { background: %(LINE)s; }
QPushButton:disabled { color: %(FG3)s; background: %(PANEL)s; }
QPushButton:checked  { background: %(ACCENT)s; color: #06121f; font-weight: 600; }

QPushButton[kind="primary"] { background: %(ACCENT)s; color: #06121f; font-weight: 600; }
QPushButton[kind="primary"]:hover   { background: %(ACCENT2)s; }
QPushButton[kind="primary"]:disabled { background: %(PANEL2)s; color: %(FG3)s; }
QPushButton[kind="record"]  { background: %(REC)s; color: white; font-weight: 600;
                              padding: 8px 22px; font-size: 13px; }
QPushButton[kind="record"]:hover    { background: #ee6560; }
QPushButton[kind="record"]:disabled { background: %(PANEL2)s; color: %(FG3)s; }
QPushButton[kind="ghost"]   { background: transparent; color: %(FG2)s; }
QPushButton[kind="ghost"]:hover { background: %(PANEL2)s; color: %(FG)s; }

QComboBox {
    background: %(PANEL2)s; border: 1px solid %(LINE)s; border-radius: 6px;
    padding: 5px 8px; min-height: 16px;
}
QComboBox:hover { border-color: %(FG3)s; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox::down-arrow { image: none; border-left: 4px solid transparent;
    border-right: 4px solid transparent; border-top: 5px solid %(FG2)s;
    margin-right: 6px; }
QComboBox QAbstractItemView {
    background: %(PANEL2)s; border: 1px solid %(LINE)s; outline: none;
    selection-background-color: %(ACCENT)s; selection-color: #06121f; padding: 3px;
}

QTabWidget::pane { border: none; background: %(PANEL)s; }
QTabBar { background: %(PANEL)s; }
QTabBar::tab {
    background: %(PANEL)s; color: %(FG3)s; padding: 9px 18px; border: none;
    border-bottom: 2px solid transparent; font-size: 12px;
}
QTabBar::tab:selected { color: %(FG)s; border-bottom: 2px solid %(ACCENT)s; }
QTabBar::tab:hover:!selected { color: %(FG2)s; }

QSlider::groove:horizontal { height: 3px; background: %(LINE)s; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(ACCENT)s; border-radius: 2px; }
QSlider::handle:horizontal {
    background: %(FG)s; width: 12px; height: 12px; margin: -5px 0; border-radius: 6px;
}
QSlider::handle:horizontal:hover { background: white; }

QScrollArea, QScrollArea > QWidget > QWidget { background: %(PANEL)s; border: none; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 0; }
QScrollBar::handle:vertical { background: %(LINE)s; border-radius: 4px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: %(FG3)s; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

QListWidget {
    background: %(PANEL)s; border: none; outline: none; padding: 6px;
}
QListWidget::item { border-radius: 6px; padding: 3px; margin: 2px 0; color: %(FG2)s; }
QListWidget::item:selected { background: %(PANEL3)s; color: %(FG)s; }

QTextEdit {
    background: %(PANEL2)s; border: 1px solid %(LINE)s; border-radius: 6px;
    padding: 6px; font-family: Menlo, monospace; font-size: 11px;
    selection-background-color: %(ACCENT)s;
}
QProgressBar { background: %(LINE)s; border: none; border-radius: 2px; height: 3px;
               text-align: center; color: transparent; }
QProgressBar::chunk { background: %(ACCENT)s; border-radius: 2px; }
QCheckBox { color: %(FG2)s; font-size: 11px; spacing: 6px; }
QCheckBox::indicator { width: 14px; height: 14px; border-radius: 4px;
                       border: 1px solid %(LINE)s; background: %(PANEL2)s; }
QCheckBox::indicator:checked { background: %(ACCENT)s; border-color: %(ACCENT)s; }
QMenu { background: %(PANEL2)s; border: 1px solid %(LINE)s; padding: 4px; }
QMenu::item { padding: 6px 20px; border-radius: 4px; }
QMenu::item:selected { background: %(ACCENT)s; color: #06121f; }
""" % dict(BG=BG, PANEL=PANEL, PANEL2=PANEL2, PANEL3=PANEL3, LINE=LINE, FG=FG,
           FG2=FG2, FG3=FG3, ACCENT=ACCENT, ACCENT2=ACCENT2, GOOD=GOOD, REC=REC)


# ── small helpers ────────────────────────────────────────────────

def button(text, on_click=None, kind="normal", tip="", checkable=False):
    # Qt reads a lone "&" as a mnemonic marker and swallows it, which is how
    # "Black & white" reached the screen as "Black  white".
    b = QPushButton(text.replace("&", "&&"))
    b.setProperty("kind", kind)
    b.setCursor(Qt.PointingHandCursor)
    b.setCheckable(checkable)
    if tip:
        b.setToolTip(tip)
    if on_click:
        b.clicked.connect(lambda *_: on_click())
    return b


def label(text, kind=""):
    lb = QLabel(text)
    if kind:
        lb.setObjectName(kind)
    return lb


def hair(vertical=False):
    f = QFrame()
    f.setObjectName("hair")
    if vertical:
        f.setFixedWidth(1)
    else:
        f.setFixedHeight(1)
    return f


def spacer():
    w = QWidget()
    w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    w.setStyleSheet("background: transparent;")
    return w


def wrap_rgb(rgb):
    """Wrap a contiguous RGB buffer as a QImage *without* copying it.

    Copying is what this avoids: a 2200 px frame cost 18 ms of the GUI thread,
    ten times a second.

    QImage does not own the memory it is handed. PySide6 does keep the buffer
    object alive for you — verified, not assumed — but the array is attached to
    the image as well, so the lifetime is visible in the code that depends on
    it rather than resting on a binding's internal behaviour.
    """
    h, w = rgb.shape[:2]
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    img._buffer = rgb                  # noqa: SLF001 - deliberate lifetime tie
    return img


def to_qimage(bgr):
    """BGR ndarray -> QImage that owns its pixels, by copying.

    The copy is not optional: QImage wraps the buffer it is given, and the
    numpy array behind a live camera frame is replaced by the next frame.
    """
    if bgr is None:
        return None
    if bgr.ndim == 2:
        bgr = np.repeat(bgr[:, :, None], 3, axis=2)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class Section(QWidget):
    """A titled block inside a panel."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        box = QVBoxLayout(self)
        box.setContentsMargins(14, 12, 14, 4)
        box.setSpacing(7)
        if title:
            box.addWidget(label(title.upper(), "heading"))
        self.body = QVBoxLayout()
        self.body.setSpacing(6)
        box.addLayout(self.body)

    def add(self, widget):
        self.body.addWidget(widget)
        return widget

    def add_layout(self, lay):
        self.body.addLayout(lay)
        return lay


class SliderRow(QWidget):
    """Label, live value and a reset affordance in one row.

    The value turns blue when it differs from the filter's own setting, so a
    glance says which of a dozen sliders you have touched — the thing that is
    genuinely hard to remember when a scan comes out wrong.
    """

    changed = Signal(str, float)

    def __init__(self, key, text, lo, hi, value, decimals=0, parent=None):
        super().__init__(parent)
        self.key, self.lo, self.hi = key, lo, hi
        self.decimals = decimals
        self.default = value
        self._mute = False
        self.setObjectName("panel")

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(1)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self.name = label(text, "sublabel")
        head.addWidget(self.name)
        head.addStretch(1)
        self.value_lbl = label(self._fmt(value), "note")
        head.addWidget(self.value_lbl)
        self.reset_btn = QPushButton("⟲")
        self.reset_btn.setFixedSize(18, 16)
        self.reset_btn.setProperty("kind", "ghost")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setToolTip("Reset")
        self.reset_btn.clicked.connect(self.reset)
        head.addWidget(self.reset_btn)
        box.addLayout(head)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(lo * self._scale), int(hi * self._scale))
        self.slider.setValue(int(value * self._scale))
        self.slider.valueChanged.connect(self._moved)
        box.addWidget(self.slider)

    @property
    def _scale(self):
        return 10 ** self.decimals

    def _fmt(self, v):
        return ("%%.%df" % self.decimals) % v

    def _moved(self, raw):
        v = raw / float(self._scale)
        off = abs(v - self.default) > 1e-9
        self.value_lbl.setText(self._fmt(v))
        self.value_lbl.setStyleSheet("color: %s;" % (ACCENT if off else FG2))
        if not self._mute:
            self.changed.emit(self.key, v)

    def set_value(self, v, default=None):
        """Push a value in without emitting — for loading a page's settings."""
        self._mute = True
        if default is not None:
            self.default = default
        self.slider.setValue(int(round(float(v) * self._scale)))
        self._moved(self.slider.value())
        self._mute = False

    def reset(self):
        self.slider.setValue(int(round(self.default * self._scale)))


class Toast(QLabel):
    """Transient status, floated over the stage. Never blocks, unlike a dialog."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.hide()
        self._timer = None

    def show_message(self, text, kind="", ms=3200):
        from PySide6.QtCore import QTimer
        colour = {"good": GOOD, "bad": BAD, "warn": WARN}.get(kind, FG)
        self.setStyleSheet(
            "background: rgba(20,24,31,235); color: %s; border: 1px solid %s;"
            "border-radius: 8px; padding: 8px 16px; font-size: 12px;" % (colour, LINE))
        self.setText(text)
        self.adjustSize()
        self.reposition()
        self.show()
        self.raise_()
        if self._timer:
            self._timer.stop()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self._timer.start(ms)

    def reposition(self):
        p = self.parentWidget()
        if p:
            self.move(max(0, (p.width() - self.width()) // 2), p.height() - self.height() - 26)


class PreviewView(QWidget):
    """The stage: an image, the page outline, draggable corners, zoom and pan.

    Corners are kept in normalised (0..1) image coordinates so they survive the
    window being resized, the view being zoomed, and mean the same thing on the
    full-resolution frame as on the preview they were dragged on.
    """

    corners_changed = Signal(object)
    zoom_changed = Signal(float)

    HANDLE = 9
    ZOOM_MAX = 16.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("stage")
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._img = None
        self._compare = None            # shown instead, while held
        self._quad = None
        self._editable = False
        self._drag = -1
        self._rect = QRectF()
        self._placeholder = "No camera running"
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._panning = None
        self.grid = False
        self.cross = False

    # -- content ---------------------------------------------------

    def set_image(self, qimg):
        self._img = qimg
        self.update()

    def set_compare(self, qimg):
        """A second image to show while the compare button is held."""
        self._compare = qimg
        self.update()

    def set_quad(self, quad):
        self._quad = None if quad is None else np.asarray(quad, dtype=np.float32)
        self.update()

    def set_editable(self, on):
        self._editable = bool(on)
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        self.update()

    def is_editable(self):
        return self._editable

    def set_placeholder(self, text):
        self._placeholder = text
        self.update()

    def set_guides(self, grid=None, cross=None):
        if grid is not None:
            self.grid = bool(grid)
        if cross is not None:
            self.cross = bool(cross)
        self.update()

    def clear(self):
        self._img = None
        self._compare = None
        self._quad = None
        self.update()

    # -- zoom ------------------------------------------------------

    def zoom(self):
        return self._zoom

    def fit(self):
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_by(self, factor, at=None):
        """Zoom about a point, so what is under the cursor stays under it."""
        base = self._fit_rect()
        if base.width() <= 0:
            return
        at = at or QPointF(self.width() / 2.0, self.height() / 2.0)
        r = self._rect if self._rect.width() > 0 else base
        u = (at.x() - r.x()) / r.width()
        v = (at.y() - r.y()) / r.height()
        self._zoom = max(1.0, min(self.ZOOM_MAX, self._zoom * factor))
        nw, nh = base.width() * self._zoom, base.height() * self._zoom
        nx, ny = at.x() - u * nw, at.y() - v * nh
        self._pan = QPointF(nx + nw / 2.0 - base.center().x(),
                            ny + nh / 2.0 - base.center().y())
        self._clamp_pan()
        self.zoom_changed.emit(self._zoom)
        self.update()

    def _clamp_pan(self):
        """Keep the picture from being dragged off the edge of the stage."""
        base = self._fit_rect()
        if base.width() <= 0:
            return
        w, h = base.width() * self._zoom, base.height() * self._zoom
        limit_x = max(0.0, (w - self.width()) / 2.0 + base.x())
        limit_y = max(0.0, (h - self.height()) / 2.0 + base.y())
        self._pan = QPointF(max(-limit_x, min(limit_x, self._pan.x())),
                            max(-limit_y, min(limit_y, self._pan.y())))

    # -- geometry --------------------------------------------------

    def _fit_rect(self):
        img = self._img
        if img is None:
            return QRectF()
        iw, ih = img.width(), img.height()
        if not iw or not ih:
            return QRectF()
        k = min(self.width() / float(iw), self.height() / float(ih))
        w, h = iw * k, ih * k
        return QRectF((self.width() - w) / 2.0, (self.height() - h) / 2.0, w, h)

    def _draw_rect(self):
        base = self._fit_rect()
        if base.width() <= 0 or (self._zoom == 1.0 and self._pan.isNull()):
            return base
        w, h = base.width() * self._zoom, base.height() * self._zoom
        cx = base.center().x() + self._pan.x()
        cy = base.center().y() + self._pan.y()
        return QRectF(cx - w / 2.0, cy - h / 2.0, w, h)

    def _to_widget(self, pt):
        r = self._rect
        return QPointF(r.x() + pt[0] * r.width(), r.y() + pt[1] * r.height())

    def _to_image(self, pos):
        r = self._rect
        if r.width() <= 0 or r.height() <= 0:
            return None
        return (min(1.0, max(0.0, (pos.x() - r.x()) / r.width())),
                min(1.0, max(0.0, (pos.y() - r.y()) / r.height())))

    # -- painting --------------------------------------------------

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#080b0e"))
        img = self._compare or self._img
        if img is None:
            p.setPen(QColor(FG3))
            font = p.font()
            font.setPointSize(13)
            p.setFont(font)
            p.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            return

        p.setRenderHint(QPainter.Antialiasing, True)
        self._rect = self._draw_rect()
        # Smooth while shrinking; at high zoom the honest thing is the pixels
        # themselves, since the whole reason to zoom in is to judge sharpness.
        p.setRenderHint(QPainter.SmoothPixmapTransform, self._zoom < 2.5)
        p.drawImage(self._rect, img)

        if self._compare is not None:
            self._badge(p, "BEFORE")
        elif self._zoom > 1.001:
            self._badge(p, "%d%%" % round(self._zoom * 100))

        self._draw_guides(p)

        if self._quad is None or len(self._quad) != 4 or self._compare is not None:
            return

        pts = [self._to_widget(c) for c in self._quad]
        poly = QPolygonF(pts)

        # Dim everything outside the page so the crop reads at a glance.
        outside = QPainterPath()
        outside.addRect(QRectF(self.rect()))
        inner = QPainterPath()
        inner.addPolygon(poly)
        p.fillPath(outside.subtracted(inner), QColor(0, 0, 0, 110))

        colour = QColor(ACCENT) if self._editable else QColor(GOOD)
        p.setPen(QPen(colour, 2))
        p.setBrush(Qt.NoBrush)
        p.drawPolygon(poly)

        if not self._editable:
            return
        r = self.HANDLE
        for i, pt in enumerate(pts):
            p.setBrush(QColor(ACCENT) if i == self._drag else QColor(BG))
            p.setPen(QPen(QColor(ACCENT), 2))
            p.drawEllipse(pt, r, r)

    def _badge(self, p, text):
        p.setPen(QColor(FG))
        font = p.font()
        font.setPointSize(10)
        font.setBold(True)
        p.setFont(font)
        box = QRectF(12, 12, 78, 22)
        p.fillRect(box, QColor(14, 17, 22, 220))
        p.drawText(box, Qt.AlignCenter, text)

    def _draw_guides(self, p):
        if not (self.grid or self.cross):
            return
        r = self._rect
        p.setPen(QPen(QColor(255, 255, 255, 46), 1))
        if self.grid:
            for i in (1, 2):
                x = r.x() + r.width() * i / 3.0
                y = r.y() + r.height() * i / 3.0
                p.drawLine(QPointF(x, r.y()), QPointF(x, r.bottom()))
                p.drawLine(QPointF(r.x(), y), QPointF(r.right(), y))
        if self.cross:
            c = r.center()
            p.setPen(QPen(QColor(255, 255, 255, 80), 1))
            p.drawLine(QPointF(c.x(), r.y()), QPointF(c.x(), r.bottom()))
            p.drawLine(QPointF(r.x(), c.y()), QPointF(r.right(), c.y()))

    # -- interaction -----------------------------------------------

    def _hit(self, pos):
        if self._quad is None:
            return -1
        best, best_d = -1, (self.HANDLE + 9) ** 2
        for i, c in enumerate(self._quad):
            w = self._to_widget(c)
            d = (w.x() - pos.x()) ** 2 + (w.y() - pos.y()) ** 2
            if d < best_d:
                best, best_d = i, d
        return best

    def wheelEvent(self, e):
        step = e.angleDelta().y()
        if not step:
            return
        self.zoom_by(1.0016 ** step, e.position())
        e.accept()

    def mouseDoubleClickEvent(self, _e):
        self.fit()

    def mousePressEvent(self, e):
        if self._editable:
            self._drag = self._hit(e.position())
            if self._drag >= 0:
                self.update()
                return
        if self._zoom > 1.001:
            self._panning = (e.position(), QPointF(self._pan))
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._panning is not None:
            start, origin = self._panning
            self._pan = QPointF(origin.x() + e.position().x() - start.x(),
                                origin.y() + e.position().y() - start.y())
            self._clamp_pan()
            self.update()
            return
        if not self._editable:
            self.setCursor(Qt.OpenHandCursor if self._zoom > 1.001 else Qt.ArrowCursor)
            return
        if self._drag < 0:
            self.setCursor(Qt.OpenHandCursor if self._hit(e.position()) >= 0
                           else Qt.CrossCursor)
            return
        pt = self._to_image(e.position())
        if pt is None:
            return
        self._quad[self._drag] = pt
        self.update()

    def mouseReleaseEvent(self, _e):
        if self._panning is not None:
            self._panning = None
            self.setCursor(Qt.OpenHandCursor if self._zoom > 1.001 else Qt.ArrowCursor)
            return
        if self._drag >= 0:
            self._drag = -1
            self.update()
            self.corners_changed.emit(self._quad.copy())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._clamp_pan()
        self._rect = self._draw_rect()
