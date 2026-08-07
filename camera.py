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

The second measured fact shapes everything below: **this camera must not be
touched before it is used.** Opening and releasing it — which is all a
resolution probe does — leaves it refusing its 16 MP mode for several seconds
afterwards. An earlier version enumerated devices by probing each one, and
that probe was itself the reason the camera that had just reported 4656x3496
handed back 1920x1080 when the app opened it. Devices are now identified by
name, through AVFoundation, without opening anything.
"""

import os
import platform
import shutil
import subprocess
import sys
import threading
import time

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))


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

# How long to let a freshly opened device settle before asking for a mode.
# Measured: at 0.6 s the 16 MP request fails about half the time; at 1.2 s it
# succeeds in 0.1 s. The device is not ready to switch modes the instant it
# opens, and asking too early does not merely fail — it wedges the session, so
# every later request in that session returns nothing either.
SETTLE = 1.2
MODE_WAIT = 3.5


class Camera:
    """Background reader that always holds the most recent full-resolution frame."""

    def __init__(self):
        self.cap = None
        self.index = None
        self.name = ""
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.at_max = False          # did the top ladder rung come up?
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        # Serialises every call into the VideoCapture itself. Releasing or
        # reconfiguring an AVFoundation session while another thread is inside
        # read() is a native crash, not an exception — and at 16 MP a read
        # takes ~100 ms, or seconds on a camera that is misbehaving, so the
        # reader is very often mid-read exactly when the app wants to stop it.
        self._cap_lock = threading.RLock()
        self._thread = None
        self._stop = threading.Event()
        self.error = None

    # ── discovery ────────────────────────────────────────────────

    @staticmethod
    def _quiet():
        """Silence OpenCV's console noise about indices that do not exist."""
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        except Exception:
            pass

    @staticmethod
    def _helper():
        """Path to the AVFoundation name helper, compiled on first use.

        Kept out of the repository deliberately: it is a Mach-O binary, and a
        one-off `swiftc` run is cheaper to trust than a committed executable.
        If Swift is not there the caller falls back to scanning indices.
        """
        exe = os.path.join(_HERE, "listcams")
        if os.path.exists(exe):
            return exe
        src = os.path.join(_HERE, "tools_listcams.swift")
        if (platform.system() != "Darwin" or not os.path.exists(src)
                or not shutil.which("swiftc")):
            return None
        try:
            subprocess.run(["swiftc", "-O", src, "-o", exe],
                           capture_output=True, timeout=240, check=True)
        except Exception:
            return None
        return exe if os.path.exists(exe) else None

    @staticmethod
    def _avfoundation_devices():
        """Names in AVFoundation's own order — the order OpenCV indexes from.

        `listcams` (tools_listcams.swift) calls `AVCaptureDevice.devices(for:)`,
        the same deprecated enumeration OpenCV's AVFoundation backend uses, so
        its position *is* the OpenCV index. A DiscoverySession would not do:
        it groups by the device types you ask for, so built-in cameras come
        back before external ones and the mapping silently shifts.

        Costs nothing and opens nothing — which is the whole point, since
        opening a device is what stops it delivering its top resolution.
        """
        exe = Camera._helper()
        if exe is None:
            return None
        try:
            out = subprocess.run([exe], capture_output=True, text=True, timeout=10)
        except Exception:
            return None
        devices = []
        for line in out.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            devices.append({"index": idx, "kind": parts[1], "name": parts[2]})
        return devices or None

    @staticmethod
    def _blind_scan(limit=3):
        """Fallback when there is no name list: which indices open at all.

        Opens and immediately releases, which is exactly the thing that upsets
        this camera — hence only a fallback, and why it is never run on macOS
        where the name list works.
        """
        Camera._quiet()
        found = []
        for i in range(limit):
            try:
                cap = cv2.VideoCapture(i, _backend())
                ok = cap.isOpened() and cap.read()[0]
                cap.release()
            except Exception:
                ok = False
            if ok:
                found.append({"index": i, "kind": "other", "name": "Camera %d" % i})
        return found

    @staticmethod
    def list_devices():
        """Cameras, best candidate first, without opening any of them.

        Ordering matters more than it looks. On a laptop the built-in webcam is
        often index 0, and taking index 0 quietly scans your documents through
        a 1080p camera pointing at your face. An external camera is what an
        overhead scanner *is*, so external wins; a phone's Continuity Camera is
        ranked last because it is almost never what you meant and waking it
        takes the session with it.
        """
        devices = Camera._avfoundation_devices()
        if devices is None:
            devices = Camera._blind_scan()
        rank = {"external": 0, "other": 1, "builtin": 2, "continuity": 3}
        return sorted(devices, key=lambda d: (rank.get(d["kind"], 1), d["index"]))

    # ── lifecycle ────────────────────────────────────────────────

    def open(self, index=0, name="", want=-1):
        """Open a device and bring up one requested mode.

        `want` is a single (w, h), or None to take whatever the device streams
        by default. Deliberately *one* request and no ladder: a mode change
        this camera declines does not merely fail, it wedges the session, and
        the version that walked six rungs looking for a smaller one turned a
        recoverable stumble into a camera that returned nothing at all. The
        caller retries, and a retry gets a clean device.

        A device will happily *report* a resolution it cannot stream, so the
        mode counts as reached only once a frame of that size has been read
        back.
        """
        if want == -1:
            want = LADDER[0]
        self.close()
        self.error = None
        self.at_max = False
        Camera._quiet()

        cap = cv2.VideoCapture(index, _backend())
        if not cap.isOpened():
            cap.release()
            self.error = "could not open camera %d" % index
            return False

        # Pull one frame before asking for a mode. The device ignores a
        # resolution set on a stream that has never delivered anything and
        # silently stays at its default — which is how a 16 MP camera ends up
        # handing back 1080p.
        ok, first = cap.read()
        if not ok or first is None:
            cap.release()
            self.error = "camera %d opened but delivered no frames" % index
            return False
        native = (first.shape[1], first.shape[0])

        chosen = None
        if want and (want[0] > native[0] or want[1] > native[1]):
            # Let it settle first. Measured: asking at 0.6 s fails about half
            # the time and takes the session down with it; at 1.2 s the same
            # request is answered in 0.1 s.
            time.sleep(SETTLE)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, want[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want[1])
            frame = self._await_size(cap, want[0], want[1], MODE_WAIT)
            if frame is None:
                cap.release()
                self.error = ("camera %d stopped responding after a %dx%d request"
                              % (index, want[0], want[1]))
                return False
            gh, gw = frame.shape[:2]
            # Compare per-dimension, not by area. An 80%-of-area tolerance
            # accepts 4208x3120 as though it satisfied a 4656x3496 request —
            # it is 80.6% of the area — and quietly costs a fifth of the
            # pixels. Both sides must be close to what was asked for.
            if gw >= want[0] * 0.95 and gh >= want[1] * 0.95:
                chosen = (gw, gh)
                self.at_max = (tuple(want) == tuple(LADDER[0]))
            else:
                chosen = (gw, gh)      # it moved somewhere, and it is streaming

        if chosen is None:
            frame = self._await_size(cap, native[0], native[1], 2.5)
            if frame is None:
                cap.release()
                self.error = "camera %d delivered no usable frames" % index
                return False
            chosen = (frame.shape[1], frame.shape[0])
            self.at_max = (tuple(chosen) == tuple(LADDER[0]))

        self.cap = cap
        self.index = index
        self.name = name
        self.width, self.height = chosen
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._start_reader()
        return True

    def upgrade(self, target=None):
        """Retry the top mode on a camera already running below it.

        Worth a button of its own: this device refuses 16 MP while it is still
        settling and accepts it seconds later, so one retry often turns a 1080p
        session into a 16 MP one without reopening anything.
        """
        target = tuple(target or LADDER[0])
        if self.cap is None or (self.width, self.height) == target:
            return False
        self._stop_reader()
        try:
            with self._cap_lock:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target[1])
                frame = self._await_size(self.cap, target[0], target[1], MODE_WAIT)
        except Exception:
            frame = None
        if frame is not None and frame.shape[1] >= target[0] * 0.95:
            self.width, self.height = frame.shape[1], frame.shape[0]
            self.at_max = True
            with self._lock:
                # Drop the frame from the old mode. `open()` gets this for free
                # because `close()` clears it, but an in-session upgrade does
                # not — and a capture taken in the seconds after would quietly
                # have been the 1080p frame that was still sitting here.
                self._frame = frame
                self._seq += 1
            self._start_reader()
            return True
        # The failed request may have wedged the session; a clean reopen is
        # the only reliable way back.
        index, name = self.index, self.name
        self.close()
        time.sleep(1.2)
        if self.open(index, name, LADDER[0]):
            return self.at_max
        # Never leave the operator staring at nothing because an optional
        # upgrade did not come off — come back at whatever it will stream.
        self.open(index, name, None)
        return False

    @staticmethod
    def _await_size(cap, w, h, secs):
        """Read until a frame of the requested size arrives, or time runs out.

        The first frames after a mode change are still the *old* size — taking
        the first one back is how a 16 MP request quietly settles for the
        1080p that was already in flight.
        """
        last = None
        deadline = time.time() + secs
        while time.time() < deadline:
            ok, f = cap.read()
            if not ok or f is None:
                continue
            last = f
            if f.shape[1] >= w * 0.95 and f.shape[0] >= h * 0.95:
                return f
        return last

    def close(self):
        self._stop_reader()
        cap, self.cap = self.cap, None
        if cap is not None:
            # If the reader is somehow still inside read(), wait for it rather
            # than release underneath it. Failing that, drop the object without
            # releasing: leaking one capture until the process exits is an
            # unpleasant outcome, and a segfault is a much worse one.
            if self._cap_lock.acquire(timeout=4.0):
                try:
                    cap.release()
                finally:
                    self._cap_lock.release()
            else:
                self.error = "camera did not stop cleanly; left it open"
        with self._lock:
            self._frame = None
        self.width = self.height = 0
        self.at_max = False

    def is_open(self):
        return self.cap is not None and self._thread is not None and self._thread.is_alive()

    # ── frames ───────────────────────────────────────────────────

    def _start_reader(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _stop_reader(self):
        self._stop.set()
        if self._thread is not None:
            # Generous: one 16 MP read plus the time a stalling camera takes to
            # give up. The lock below is what makes correctness independent of
            # this number, but waiting properly avoids ever needing it.
            self._thread.join(timeout=6.0)
            self._thread = None

    def _run(self):
        misses = 0
        while not self._stop.is_set():
            cap = self.cap
            if cap is None:
                break
            try:
                with self._cap_lock:
                    if self._stop.is_set() or self.cap is not cap:
                        break
                    ok, frame = cap.read()
            except Exception as exc:                # noqa: BLE001
                self.error = "camera read failed: %s" % exc
                break
            if not ok or frame is None:
                misses += 1
                if misses > 90:
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
