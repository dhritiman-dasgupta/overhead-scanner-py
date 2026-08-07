"""The processing pipeline, in numpy/OpenCV.

Order:
  warp -> rotate/flip/straighten -> illumination flattening -> white balance ->
  temperature/tint -> tone curve -> saturation/vibrance -> denoise -> sharpen ->
  filter mode (grey / adaptive threshold) -> invert

Two things here were learned the hard way against real captures, and both are
about *not destroying detail*:

  Nothing clips. Highlights roll off above a knee rather than being clamped at
  255 — in the divide, in the tone curve and in the sharpener. Each of those
  was independently capable of turning printed detail on a bright label into
  blank white. An unsharp mask deliberately overshoots at an edge; clamping
  that overshoot is how sharpening ends up destroying what it was meant to
  reveal.

  Sharpening scales with the image. A fixed 3x3 kernel sharpens strokes at
  preview size and grain at 16 MP, so the export came out both noisier than the
  preview and different from it.
"""

import cv2
import numpy as np

KNEE = 232.0

DEFAULTS = {
    "filter": "auto", "mode": "color",
    "flatten": 95, "wb": "gray", "temp": 0, "tint": 0,
    "exposure": 0, "contrast": 0, "gamma": 1.0, "highlights": 0, "shadows": 0,
    "saturation": 0, "vibrance": 0,
    "denoise": 0, "sharpen": 16,
    "threshold": 0, "window": 100,
    "invert": False,
    "rotate": 0, "fliph": False, "flipv": False, "straighten": 0.0,
    "outsize": "detected",
}

FILTERS = {
    "original":   dict(mode="color", flatten=0,   wb="off",   contrast=0,  gamma=1.0,  saturation=0,  vibrance=0,  sharpen=0,  denoise=0, exposure=0, highlights=0, shadows=0, temp=0, tint=0),
    "auto":       dict(mode="color", flatten=95,  wb="gray",  contrast=18, gamma=1.0,  saturation=6,  vibrance=10, sharpen=16, denoise=0, exposure=0, highlights=0, shadows=0),
    "color":      dict(mode="color", flatten=95,  wb="white", contrast=26, gamma=1.05, saturation=14, vibrance=18, sharpen=20, denoise=4, exposure=2, highlights=-6, shadows=4),
    "gray":       dict(mode="gray",  flatten=95,  wb="gray",  contrast=22, gamma=1.0,  saturation=0,  vibrance=0,  sharpen=18, denoise=4, exposure=0, highlights=0, shadows=0),
    "bw":         dict(mode="bw",    flatten=100, wb="gray",  contrast=10, gamma=1.0,  saturation=0,  vibrance=0,  sharpen=12, denoise=8, exposure=0, highlights=0, shadows=0, threshold=0, window=100),
    "whiteboard": dict(mode="color", flatten=100, wb="white", contrast=42, gamma=1.1,  saturation=45, vibrance=25, sharpen=14, denoise=10, exposure=4, highlights=-10, shadows=0),
    "ink":        dict(mode="gray",  flatten=100, wb="gray",  contrast=56, gamma=0.82, saturation=0,  vibrance=0,  sharpen=26, denoise=6, exposure=-2, highlights=0, shadows=-12),
    "photo":      dict(mode="color", flatten=0,   wb="off",   contrast=8,  gamma=1.0,  saturation=8,  vibrance=12, sharpen=8,  denoise=0, exposure=0, highlights=0, shadows=0),
}

PAPER = {"a4": (2480, 3508), "letter": (2550, 3300)}

GEOMETRY_KEYS = ("rotate", "fliph", "flipv", "straighten", "outsize")


def new_adjust():
    a = dict(DEFAULTS)
    a.update(FILTERS["auto"])
    a["filter"] = "auto"
    return a


def set_filter(adjust, name):
    preset = FILTERS.get(name)
    if not preset:
        return adjust
    keep = {k: adjust[k] for k in GEOMETRY_KEYS if k in adjust}
    keep["invert"] = adjust.get("invert", False)
    adjust.update(DEFAULTS)
    adjust.update(preset)
    adjust.update(keep)
    adjust["filter"] = name
    return adjust


# ── helpers ──────────────────────────────────────────────────────

def _shoulder(x):
    """Compress above the knee instead of clipping, so bright detail survives."""
    out = x.copy()
    hi = x > KNEE
    if np.any(hi):
        out[hi] = KNEE + (255.0 - KNEE) * np.tanh((x[hi] - KNEE) / (255.0 - KNEE))
    return np.clip(out, 0, 255)


def _paper_level(img, pct=90):
    g = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return float(np.percentile(g[::4, ::4], pct))


# ── stages ───────────────────────────────────────────────────────

def _flatten(img, strength):
    """Divide the image by its own illumination map.

    The estimate is the larger of two readings: a heavily blurred one, which
    passes a lamp's falloff through and ignores content, and the barely
    smoothed block maxima, which is what paper actually reads *here*. The blur
    alone lets whatever surrounds the paper drag the estimate down — a page
    that does not fill the frame has dark mat on the other side of its edge —
    so the gain climbs and lifts the ink along with the paper, bleaching text
    near the page edges. Taking the maximum of the two fixes that without
    losing the blur's real job, which is not blowing a dark photo inside a page
    to white.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    bw = int(np.clip(round(w / 14.0), 10, 88))
    bh = int(np.clip(round(h / 14.0), 10, 88))
    kx = max(1, int(round(w / float(bw))) | 1)
    ky = max(1, int(round(h / float(bh))) | 1)
    # per-block maximum: dilate by a block, then sample. Text and ink are dark,
    # and the maximum reads straight through them to the paper underneath.
    peak = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    small = cv2.resize(peak, (bw, bh), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    broad_k = max(3, (int(min(bw, bh) / 3) * 2 + 1))
    broad = cv2.GaussianBlur(small, (broad_k, broad_k), 0)
    local = cv2.GaussianBlur(small, (3, 3), 0)
    bg = np.maximum(broad, local * 0.8)

    floor = max(16.0, float(bg.max()) * 0.35)
    np.maximum(bg, floor, out=bg)

    s = float(np.clip(strength, 0, 100)) / 100.0
    gain = np.clip(np.power(255.0 / bg, s), 0.2, 3.5)
    gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)

    out = img.astype(np.float32) * gain[:, :, None]
    return _shoulder(out)


def _white_balance(img, mode):
    """Grey-world over the *paper*, not the whole frame.

    Averaging everything is the classic assumption and it is badly wrong for a
    scanner, where most of the frame can be a dark mat whose own colour cast
    then gets corrected as if it were the document.
    """
    if mode == "off":
        return img
    f = img.astype(np.float32)
    if mode == "gray":
        cut = max(24.0, _paper_level(img) * 0.55)
        g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        m = g >= cut
        if m.sum() < 32:
            return img
        means = f[m].reshape(-1, 3).mean(axis=0)
    elif mode == "white":
        means = np.percentile(f.reshape(-1, 3), 97, axis=0) / 248.0 * 255.0
    else:
        return img
    means = np.maximum(means, 1.0)
    gains = np.clip(means.mean() / means, 0.55, 1.9)
    return f * gains[None, None, :]


def _tone_lut(a):
    x = np.arange(256, dtype=np.float32)
    c = float(np.clip(a["contrast"], -100, 100)) * 2.55
    cf = (259.0 * (c + 255.0)) / (255.0 * (259.0 - c))
    v = x + a["exposure"] * 1.28
    v = cf * (v - 128.0) + 128.0
    v = np.clip(v, 0, 255)
    v = 255.0 * np.power(v / 255.0, 1.0 / max(0.05, float(a["gamma"])))
    n = v / 255.0
    if a["shadows"]:
        v = v + a["shadows"] * 1.15 * np.power(1.0 - n, 2.2)
    if a["highlights"]:
        v = v + a["highlights"] * 1.15 * np.power(n, 2.2)
    return _shoulder(v).astype(np.uint8)


def _saturation(img, sat, vib):
    f = img.astype(np.float32)
    lum = (f[:, :, 2] * 0.2126 + f[:, :, 1] * 0.7152 + f[:, :, 0] * 0.0722)[:, :, None]
    amt = 1.0 + sat / 100.0
    if vib:
        mx = f.max(axis=2, keepdims=True)
        mn = f.min(axis=2, keepdims=True)
        amt = amt + (vib / 100.0) * (1.0 - (mx - mn) / 255.0)
    return lum + (f - lum) * amt


def _sharpen(img, amount):
    """Unsharp mask at a radius that scales with the image."""
    h, w = img.shape[:2]
    r = int(np.clip(round(max(w, h) / 1500.0), 1, 6))
    k = 2 * r + 1
    src = img.astype(np.float32)
    blur = cv2.GaussianBlur(src, (k, k), 0)
    return _shoulder(src + (src - blur) * (amount / 100.0))


def _adaptive_threshold(img, bias, window_pct):
    """Local threshold: handles the gradient a single desk lamp leaves behind,
    which a global threshold cannot."""
    g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    block = int(round((min(w, h) / 16.0) * (window_pct / 100.0)))
    block = max(3, block | 1)
    C = float(np.clip(-bias * 0.25, -40, 40)) + 8.0
    bwimg = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, block, C)
    return cv2.cvtColor(bwimg, cv2.COLOR_GRAY2BGR).astype(np.float32)


# ── geometry ─────────────────────────────────────────────────────

def transform(img, a):
    out = img
    if a.get("fliph"):
        out = cv2.flip(out, 1)
    if a.get("flipv"):
        out = cv2.flip(out, 0)
    rot = int(a.get("rotate", 0)) % 360
    if rot == 90:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif rot == 270:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ang = float(a.get("straighten", 0.0))
    if abs(ang) > 1e-3:
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -ang, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += nw / 2.0 - w / 2.0
        M[1, 2] += nh / 2.0 - h / 2.0
        out = cv2.warpAffine(out, M, (nw, nh), flags=cv2.INTER_LANCZOS4,
                             borderValue=(255, 255, 255))
    return out


def target_size(a, corners, w, h, max_dim=None):
    import detect as _d
    if corners is not None:
        bw, bh = _d.output_size(corners, w, h)
    else:
        bw, bh = w, h
    paper = PAPER.get(a.get("outsize"))
    if paper:
        portrait = bh >= bw
        bw, bh = (paper[0], paper[1]) if portrait else (paper[1], paper[0])
    if max_dim:
        k = max_dim / float(max(bw, bh))
        if k < 1:
            bw, bh = max(16, int(bw * k)), max(16, int(bh * k))
    return bw, bh


# ── the pipeline ─────────────────────────────────────────────────

def process(frame, a, corners=None, max_dim=None):
    """Full path from a captured frame to a finished page (BGR uint8)."""
    import detect as _d
    h, w = frame.shape[:2]
    size = target_size(a, corners, w, h, max_dim)

    if corners is not None:
        img = _d.warp(frame, corners, size)
    elif (size[0], size[1]) != (w, h):
        interp = cv2.INTER_AREA if size[0] < w else cv2.INTER_LANCZOS4
        img = cv2.resize(frame, size, interpolation=interp)
    else:
        img = frame

    img = transform(img, a)
    f = img.astype(np.float32)

    if a["flatten"] > 0:
        f = _flatten(f, a["flatten"])
    if a.get("wb", "off") != "off":
        f = _white_balance(f, a["wb"])

    if a["temp"] or a["tint"]:
        t, ti = a["temp"] / 300.0, a["tint"] / 300.0
        gains = np.array([(1 - t) * (1 - ti * 0.5), 1 + ti, (1 + t) * (1 - ti * 0.5)],
                         dtype=np.float32)   # BGR
        f = f * gains[None, None, :]

    if (a["contrast"] or a["exposure"] or a["highlights"] or a["shadows"]
            or abs(a["gamma"] - 1.0) > 1e-6):
        f = cv2.LUT(np.clip(f, 0, 255).astype(np.uint8), _tone_lut(a)).astype(np.float32)

    if a["mode"] == "color" and (a["saturation"] or a["vibrance"]):
        f = _saturation(f, a["saturation"], a["vibrance"])

    if a["denoise"] > 0:
        k = 3 if max(f.shape[:2]) < 2000 else 5
        med = cv2.medianBlur(np.clip(f, 0, 255).astype(np.uint8), k).astype(np.float32)
        t = a["denoise"] / 100.0
        f = f * (1 - t) + med * t

    if a["sharpen"] > 0:
        f = _sharpen(f, a["sharpen"])

    if a["mode"] == "gray":
        g = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        f = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32)
    elif a["mode"] == "bw":
        f = _adaptive_threshold(f, a["threshold"], a["window"])

    if a.get("invert"):
        f = 255.0 - f

    return np.clip(f, 0, 255).astype(np.uint8)


def fit(img, max_dim):
    h, w = img.shape[:2]
    k = min(1.0, max_dim / float(max(w, h)))
    if k >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))),
                      interpolation=cv2.INTER_AREA)
