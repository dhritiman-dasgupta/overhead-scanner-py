"""OCR via the Tesseract command-line binary.

Deliberately shells out rather than depending on a Python binding: the binding
still needs the same binary, and this way the app has no install step at all
and simply reports the truth when Tesseract is absent.

Recognition runs on the *processed* page, so a good Black & white or Auto
filter lifts accuracy more than any OCR setting will.
"""

import os
import shutil
import subprocess
import tempfile

import cv2

# Where Homebrew and MacPorts put it, plus whatever is on PATH.
_CANDIDATES = [
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/local/bin/tesseract",
    "/usr/bin/tesseract",
]

OCR_MAX = 2400          # long edge fed to Tesseract; more is slower, not better


def binary():
    """Path to a usable tesseract, or None."""
    found = shutil.which("tesseract")
    if found:
        return found
    for path in _CANDIDATES:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    return None


def available():
    return binary() is not None


def languages():
    exe = binary()
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--list-langs"], capture_output=True, text=True,
                             timeout=20)
        return [l.strip() for l in out.stdout.splitlines()[1:] if l.strip()]
    except Exception:
        return []


def install_hint():
    return ("Tesseract is not installed. With Homebrew:\n"
            "    brew install tesseract\n"
            "Then reopen this app — it is found automatically.")


def _fit(img, max_dim):
    h, w = img.shape[:2]
    k = min(1.0, max_dim / float(max(w, h)))
    if k >= 1.0:
        return img
    return cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)


def recognise(image, lang="eng", psm=3, want_words=True):   # noqa: D401
    """Read a processed page.

    Returns {'text', 'words': [{text,conf,x0,y0,x1,y1}], 'confidence', 'scale'}.
    Word boxes are in the coordinates of the image handed in, so they can be
    scaled onto a full-resolution export for a searchable PDF.
    """
    exe = binary()
    if not exe:
        raise RuntimeError(install_hint())

    small = _fit(image, OCR_MAX)
    scale = image.shape[1] / float(small.shape[1])

    tmp = tempfile.mkdtemp(prefix="ohs-ocr-")
    png = os.path.join(tmp, "page.png")
    cv2.imwrite(png, small)
    base = os.path.join(tmp, "out")

    common = [exe, png, base, "-l", lang, "--psm", str(psm),
              "-c", "preserve_interword_spaces=1"]
    try:
        subprocess.run(common, capture_output=True, timeout=180, check=True)
        with open(base + ".txt", "r", errors="replace") as fh:
            text = fh.read().rstrip()

        words, confs = [], []
        if want_words:
            subprocess.run(common + ["tsv"], capture_output=True, timeout=180, check=True)
            with open(base + ".tsv", "r", errors="replace") as fh:
                for line in fh.read().splitlines()[1:]:
                    parts = line.split("\t")
                    if len(parts) < 12:
                        continue
                    word = parts[11].strip()
                    if not word:
                        continue
                    try:
                        conf = float(parts[10])
                        x, y, w, h = (int(parts[6]), int(parts[7]),
                                      int(parts[8]), int(parts[9]))
                    except ValueError:
                        continue
                    if conf < 0:
                        continue
                    confs.append(conf)
                    words.append({"text": word, "conf": conf,
                                  "x0": x * scale, "y0": y * scale,
                                  "x1": (x + w) * scale, "y1": (y + h) * scale})
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode("utf-8", "replace")[:300]
        raise RuntimeError("tesseract failed: %s" % (err or "unknown error"))
    except subprocess.TimeoutExpired:
        raise RuntimeError("tesseract timed out")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {"text": text, "words": words, "img_w": image.shape[1], "img_h": image.shape[0],
            "confidence": (sum(confs) / len(confs)) if confs else None,
            "scale": scale}
