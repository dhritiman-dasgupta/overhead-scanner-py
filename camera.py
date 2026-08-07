"""Camera capture, at the sensor's real maximum.

The point of this module is one measured fact. On this hardware the browser
route tops out awkwardly — Chrome's video stream will not run the 4656x3496
mode (66 seconds to a first frame), and macOS AVFoundation's own video path is
capped by the session preset at 1920x1080 whatever `activeFormat` says. OpenCV
asks the device directly and gets the full 4656x3496 at about 10 fps.

So there is no preview mode and no capture mode here. The camera runs at full
resolution the whole time and a capture is simply the frame you were already
looking at, downscaled only for display. That removes a whole class of bug:
the preview and the capture cannot disagree about resolution, field of view or
what the page looked like, because they are the same pixels.
"""

import platform
import threading
import time

import cv2


def _backend():
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


# Tried in order. The first that delivers real frames wins.
LADDER = [
    (4656, 3496),   # 16 MP, 4:3
    (4208, 3120),
    (3840, 2160),
    (2592, 1944),
    (1920, 1080),
    (1280, 720),
]


class Camera:
    """Background reader that always holds the most recent full-resolution frame."""

    def __init__(self):
        self.cap = None
        self.index = None
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self.error = None

    # ── discovery ────────────────────────────────────────────────

    @staticmethod
    def _quiet():
        """Silence OpenCV's console noise about indices that do not exist.

        Probing for cameras necessarily asks about indices that are not there,
        and each miss prints two lines to stderr. They are not errors and they
        bury real ones.
        """
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        except Exception:
            pass

    @staticmethod
    def list_devices(limit=4, probe_max=True):
        """Probe indices and report what each one can actually do.

        OpenCV cannot give device names on macOS, so the honest way to tell a
        16 MP document camera from a 1080p built-in webcam is to ask each one
        for a large mode and see what comes back. Costs a couple of seconds per
        device, and it is what lets the app default to the right camera instead
        of whichever happens to be index 0.
        """
        Camera._quiet()
        found = []
        misses = 0
        for i in range(limit):
            cap = cv2.VideoCapture(i, _backend())
            if not cap.isOpened():
                cap.release()
                misses += 1
                if misses >= 2:      # indices are contiguous; stop after a gap
                    break
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                continue
            default = (frame.shape[1], frame.shape[0])
            best = default
            if probe_max:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, LADDER[0][0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LADDER[0][1])
                time.sleep(0.35)
                deadline = time.time() + 3.0
                while time.time() < deadline:
                    got, f = cap.read()
                    if got and f is not None:
                        if f.shape[1] * f.shape[0] > best[0] * best[1]:
                            best = (f.shape[1], f.shape[0])
                        if best[0] >= LADDER[0][0] * 0.9:
                            break
            cap.release()
            time.sleep(0.25)          # let the device settle before the next open
            found.append({"index": i, "width": default[0], "height": default[1],
                          "max_width": best[0], "max_height": best[1]})
        return found

    # ── lifecycle ────────────────────────────────────────────────

    def open(self, index=0, prefer=None):
        """Open a device and settle it on the highest mode that really works.

        A device will happily *report* a resolution it cannot actually stream,
        so each rung of the ladder is only accepted once a frame of the
        expected size has been read back.
        """
        self.close()
        self.error = None
        cap = cv2.VideoCapture(index, _backend())
        if not cap.isOpened():
            self.error = "could not open camera %d" % index
            cap.release()
            return False

        # Pull one frame before asking for a mode. This device ignores a
        # resolution set on a stream that has never delivered anything, and
        # silently stays at its default — which is how a 16 MP camera ends up
        # handing back 1080p.
        cap.read()
        time.sleep(0.2)

        ladder = list(LADDER)
        if prefer:
            ladder = [tuple(prefer)] + [m for m in ladder if tuple(m) != tuple(prefer)]

        chosen = None
        for (w, h) in ladder:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            time.sleep(0.5)
            # Keep reading until a frame of the requested size arrives. The
            # first frames after a mode change are still the *old* size —
            # taking the first one back is how a 16 MP request quietly settles
            # for the 1080p that was already in flight.
            frame = None
            deadline = time.time() + 4.0
            while time.time() < deadline:
                got, f = cap.read()
                if not got or f is None:
                    continue
                frame = f
                fh, fw = f.shape[:2]
                if fw * fh >= w * h * 0.8:
                    break
            if frame is None:
                continue
            gh, gw = frame.shape[:2]
            # Accept a mode that returns something close to what was asked for;
            # some drivers round to their nearest supported size.
            if gw * gh >= w * h * 0.8:
                chosen = (gw, gh)
                break

        if chosen is None:
            self.error = "camera %d delivered no usable frames" % index
            cap.release()
            return False

        self.cap = cap
        self.index = index
        self.width, self.height = chosen
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        with self._lock:
            self._frame = None
        self.width = self.height = 0

    def is_open(self):
        return self.cap is not None and self._thread is not None and self._thread.is_alive()

    # ── frames ───────────────────────────────────────────────────

    def _run(self):
        misses = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                misses += 1
                if misses > 60:
                    self.error = "camera stopped delivering frames"
                    break
                time.sleep(0.02)
                continue
            misses = 0
            with self._lock:
                self._frame = frame
                self._seq += 1

    def latest(self):
        """The most recent full-resolution frame, or None. Not copied — treat
        as read-only; anything that mutates must copy first."""
        with self._lock:
            return self._frame

    def grab(self):
        """A private copy of the current frame, for keeping as a page."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def sequence(self):
        with self._lock:
            return self._seq
