"""Draw the app icon and write icon.icns. Called by make-app.sh.

Drawn rather than shipped as a binary asset: it keeps the repository text, and
the shape — a page under an overhead lens — is three primitives.
"""

import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw

SIZES = [16, 32, 64, 128, 256, 512, 1024]
BG_TOP = (32, 44, 62)
BG_BOTTOM = (13, 17, 22)
PAPER = (238, 243, 249)
ACCENT = (77, 163, 255)


def draw(size):
    s = size * 4                       # supersample, then shrink: cheap antialiasing
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    for y in range(s):                 # vertical gradient body
        t = y / float(s - 1)
        d.line([(0, y), (s, y)],
               fill=tuple(int(a + (b - a) * t) for a, b in zip(BG_TOP, BG_BOTTOM)) + (255,))

    rounded = Image.new("L", (s, s), 0)
    ImageDraw.Draw(rounded).rounded_rectangle([0, 0, s - 1, s - 1],
                                              radius=int(s * 0.225), fill=255)
    img.putalpha(rounded)

    # the page
    pw, ph = s * 0.40, s * 0.52
    px, py = (s - pw) / 2, s * 0.40
    d.rounded_rectangle([px, py, px + pw, py + ph], radius=s * 0.02, fill=PAPER)
    for i in range(5):                 # lines of text
        ly = py + ph * (0.16 + i * 0.15)
        w = pw * (0.72 if i % 2 == 0 else 0.54)
        d.rounded_rectangle([px + pw * 0.14, ly, px + pw * 0.14 + w, ly + s * 0.012],
                            radius=s * 0.006, fill=(150, 165, 185))

    # the lens above it, on its arm
    d.rounded_rectangle([s * 0.475, s * 0.12, s * 0.525, s * 0.34],
                        radius=s * 0.02, fill=ACCENT)
    d.ellipse([s * 0.40, s * 0.08, s * 0.60, s * 0.28], fill=ACCENT)
    d.ellipse([s * 0.445, s * 0.125, s * 0.555, s * 0.235], fill=BG_BOTTOM + (255,))

    return img.resize((size, size), Image.LANCZOS)


def main(out_dir):
    tmp = tempfile.mkdtemp(suffix=".iconset")
    for size in SIZES:
        draw(size).save(os.path.join(tmp, "icon_%dx%d.png" % (size, size)))
        if size <= 512:
            draw(size * 2).save(os.path.join(tmp, "icon_%dx%d@2x.png" % (size, size)))
    icns = os.path.join(out_dir, "icon.icns")
    subprocess.run(["iconutil", "-c", "icns", tmp, "-o", icns], check=True)
    print("  icon: %s" % icns)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
