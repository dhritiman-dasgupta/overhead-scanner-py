"""Page detection and perspective correction.

Corners are normalised (0..1 of width/height) so one detection on a small
thumbnail applies unchanged to the full-resolution frame.

The approach is the same one the browser version arrived at after testing
against real desks, carried over because the reasoning still holds:

  * Try both threshold polarities. Guessing which side of the split is paper
    from a centre-vs-border brightness comparison is wrong often enough in
    ordinary use — a page that nearly fills the frame leaves a border ring
    that is itself mostly paper, a hand resting in the middle drags the centre
    down — and guessing wrong meant no outline at all.
  * Close the mask before looking for the page. Lines of text cut the paper
    into stripes, and the largest stripe is not the page.
  * Demand that the region fill its own outline. On a desk close to paper
    colour the paper-side threshold takes in the desk too and gets rejected,
    so the ink-side attempt wins — and its hull is the *text block*, a
    convincing quadrilateral sitting well inside the real page.
  * Decline rather than guess. A wrong crop is worse than no crop.
"""

import cv2
import numpy as np

WORK = 320          # detection runs on a thumbnail this size


def _order(pts):
    """Sort four points into top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    pts = pts[order]
    # start from whichever point is furthest towards the top-left
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return np.roll(pts, -start, axis=0)


def _valid(quad, w, h):
    area = cv2.contourArea(quad.astype(np.float32))
    if area < 0.045 * w * h or area > 0.95 * w * h:
        return False
    shortest = min(w, h)
    for i in range(4):
        a, b, c = quad[(i + 3) % 4], quad[i], quad[(i + 1) % 4]
        if np.hypot(*(quad[(i + 1) % 4] - quad[i])) < 0.12 * shortest:
            return False
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return False
        ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
        if ang < 45 or ang > 135:
            return False
    return True


def _attempt(gray, thr, bright, w, h):
    mask = (gray > thr if bright else gray < thr).astype(np.uint8)

    # Bridge the gaps that lines of type cut through the paper. Big enough for
    # text, far too small to bridge paper to desk.
    r = max(2, int(round(min(w, h) * 0.025)))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    solid = cv2.dilate(mask, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
    if n < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    # Hull from the ORIGINAL mask inside that component, so the quad sits on
    # the real paper edge rather than r pixels outside it.
    inside = (labels == biggest) & (mask > 0)
    count = int(inside.sum())
    total = w * h
    if count < total * 0.04 or count > total * 0.99:
        return None

    ys, xs = np.nonzero(inside)
    pts = np.column_stack([xs, ys]).astype(np.int32)
    if len(pts) < 8:
        return None

    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        box = cv2.boxPoints(cv2.minAreaRect(hull))
        quad = np.asarray(box, dtype=np.float32)

    quad = _order(quad)
    if not _valid(quad, w, h):
        return None

    # The region has to fill the shape it claims to be. Ink covers a small
    # fraction of its own bounding box; paper fills nearly all of its outline.
    if count < cv2.contourArea(quad) * 0.5:
        return None

    # And it must be a crop. A quad covering nearly everything is an outline
    # drawn round the whole frame, which is not a crop at all.
    if cv2.contourArea(quad) > 0.95 * total:
        return None

    # Separation: the page has to stand out from its surroundings. Otsu always
    # returns a split, even on a blank desk.
    inner = gray[inside].mean()
    outer = gray[~inside].mean()
    if abs(float(inner) - float(outer)) < 12:
        return None

    return quad


_sticky = {"bright": None}


def reset_sticky():
    _sticky["bright"] = None


def detect(frame, sticky=False):
    """Find the page. Returns 4 normalised corners (TL,TR,BR,BL) or None."""
    if frame is None or frame.size == 0:
        return None
    H, W = frame.shape[:2]
    scale = WORK / float(max(W, H))
    w, h = max(40, int(round(W * scale))), max(40, int(round(H * scale)))
    small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    m = max(2, int(round(min(w, h) * 0.04)))
    ring = np.concatenate([gray[:m].ravel(), gray[-m:].ravel(),
                           gray[:, :m].ravel(), gray[:, -m:].ravel()])
    mid = gray[int(h * .35):int(h * .65), int(w * .35):int(w * .65)]
    prefer = bool(mid.mean() >= ring.mean())
    if sticky and _sticky["bright"] is not None:
        prefer = _sticky["bright"]

    for bright in (prefer, not prefer):
        quad = _attempt(gray, thr, bright, w, h)
        if quad is not None:
            if sticky:
                _sticky["bright"] = bright
            # Pull in a hair: the mask boundary straddles the soft ramp at the
            # paper's edge, and erring outward leaves a fringe of desk.
            c = quad.mean(axis=0)
            v = quad - c
            n = np.linalg.norm(v, axis=1, keepdims=True)
            n[n == 0] = 1
            quad = quad - v / n * 0.8
            out = quad / np.array([w, h], dtype=np.float32)
            return np.clip(out, 0, 1)
    return None


# ── perspective ──────────────────────────────────────────────────

def output_size(corners, width, height):
    """Pixel size the quad should unwarp to: the longer of each opposite pair."""
    p = np.asarray(corners, dtype=np.float32) * np.array([width, height], dtype=np.float32)
    w = max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[3]))
    h = max(np.linalg.norm(p[3] - p[0]), np.linalg.norm(p[2] - p[1]))
    return max(16, int(round(w))), max(16, int(round(h)))


def warp(frame, corners, size=None, interp=cv2.INTER_LANCZOS4):
    """Straighten the quad out of the frame into a rectangle."""
    H, W = frame.shape[:2]
    src = (np.asarray(corners, dtype=np.float32) *
           np.array([W, H], dtype=np.float32))
    ow, oh = size if size else output_size(corners, W, H)
    dst = np.array([[0, 0], [ow, 0], [ow, oh], [0, oh]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    # INTER_AREA is not valid for warpPerspective; LANCZOS4 keeps small print
    # crisp when the crop is being scaled down.
    return cv2.warpPerspective(frame, M, (ow, oh), flags=interp,
                               borderMode=cv2.BORDER_REPLICATE)


def full_frame():
    return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
