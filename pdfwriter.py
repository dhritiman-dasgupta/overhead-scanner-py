"""Minimal PDF 1.4 writer: JPEG pages, optionally with an invisible text layer.

Written by hand rather than pulled in as a dependency so export works with
nothing installed. Each page is a DCTDecode (JPEG) XObject; when OCR words are
supplied they are drawn in Helvetica with rendering mode 3 (invisible) over the
image, which is what makes a scan selectable and searchable.
"""

import datetime

PAGE_SIZES = {"a4": (595.28, 841.89), "letter": (612.0, 792.0)}

# Helvetica advance widths, /1000 em, ASCII 32..126. Anything else falls back
# to 500 — close enough, the text is invisible and only the selection
# rectangles are user-visible.
_HELV = [
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
]

_WINANSI = {0x2018: 0x91, 0x2019: 0x92, 0x201C: 0x93, 0x201D: 0x94, 0x2022: 0x95,
            0x2013: 0x96, 0x2014: 0x97, 0x2026: 0x85, 0x20AC: 0x80, 0x2122: 0x99}


def _width_of(text, size):
    total = 0
    for ch in text:
        c = ord(ch)
        total += _HELV[c - 32] if 32 <= c <= 126 else 500
    return total * size / 1000.0


def _pdf_string(text):
    out = ["("]
    for ch in text:
        c = ord(ch)
        c = _WINANSI.get(c, c)
        if c > 255:
            c = 63
        if c in (40, 41, 92):
            out.append("\\" + chr(c))
        elif c < 32:
            out.append(" ")
        else:
            out.append(chr(c))
    out.append(")")
    return "".join(out)


def _latin1(s):
    return s.encode("latin-1", "replace")


def build(pages, page_size="a4", title="Scan", searchable=True):
    """pages: [{'jpeg': bytes, 'width': int, 'height': int, 'words': [...]}]

    A word is {'text', 'x0', 'y0', 'x1', 'y1'} in page-image pixels.
    Returns the PDF as bytes.
    """
    chunks = []
    length = [0]

    def push(x):
        b = _latin1(x) if isinstance(x, str) else x
        chunks.append(b)
        length[0] += len(b)

    n_objs = 4 + len(pages) * 3
    offsets = [0] * (n_objs + 1)

    def begin(n):
        offsets[n] = length[0]
        push("%d 0 obj\n" % n)

    push("%PDF-1.4\n%\xE2\xE3\xCF\xD3\n")

    def page_obj(i):
        return 5 + i * 3

    begin(1); push("<< /Type /Catalog /Pages 2 0 R >>\n"); push("endobj\n")
    begin(2)
    push("<< /Type /Pages /Count %d /Kids [%s] >>\n"
         % (len(pages), " ".join("%d 0 R" % page_obj(i) for i in range(len(pages)))))
    push("endobj\n")
    begin(3)
    push("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\n")
    push("endobj\n")
    begin(4)
    now = datetime.datetime.now()
    push("<< /Producer (Overhead Scanner) /Title %s /CreationDate (D:%s) >>\n"
         % (_pdf_string(title), now.strftime("%Y%m%d%H%M%S")))
    push("endobj\n")

    for i, pg in enumerate(pages):
        iw, ih = pg["width"], pg["height"]
        if page_size in PAGE_SIZES:
            base = PAGE_SIZES[page_size]
            landscape = iw > ih
            pw, ph = (base[1], base[0]) if landscape else base
        else:                                   # match the image's own shape
            longest = 841.89
            if iw >= ih:
                pw, ph = longest, longest * ih / float(iw)
            else:
                ph, pw = longest, longest * iw / float(ih)

        k = min(pw / iw, ph / ih)
        dw, dh = iw * k, ih * k
        ox, oy = (pw - dw) / 2.0, (ph - dh) / 2.0

        content = "q\n%.2f 0 0 %.2f %.2f %.2f cm\n/Im0 Do\nQ\n" % (dw, dh, ox, oy)

        words = pg.get("words") if searchable else None
        if words:
            sx, sy = dw / float(iw), dh / float(ih)
            t = ["BT\n3 Tr\n"]
            for wd in words:
                text = (wd.get("text") or "").strip()
                if not text:
                    continue
                bw = (wd["x1"] - wd["x0"]) * sx
                bh = (wd["y1"] - wd["y0"]) * sy
                if bw <= 0.3 or bh <= 0.3:
                    continue
                fs = min(bh * 0.92, 200.0)
                natural = _width_of(text, fs)
                if natural <= 0:
                    continue
                tz = max(1.0, min(3000.0, bw / natural * 100.0))
                x = ox + wd["x0"] * sx
                y = oy + dh - wd["y1"] * sy + fs * 0.16
                t.append("/F1 %.2f Tf\n%.1f Tz\n1 0 0 1 %.2f %.2f Tm\n%s Tj\n"
                         % (fs, tz, x, y, _pdf_string(text)))
            t.append("ET\n")
            content += "".join(t)

        body = _latin1(content)
        n_page = page_obj(i)
        n_content, n_img = n_page + 1, n_page + 2

        begin(n_page)
        push("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f]"
             " /Resources << /XObject << /Im0 %d 0 R >> /Font << /F1 3 0 R >> >>"
             " /Contents %d 0 R >>\n" % (pw, ph, n_img, n_content))
        push("endobj\n")

        begin(n_content)
        push("<< /Length %d >>\nstream\n" % len(body))
        push(body)
        push("\nendstream\n")
        push("endobj\n")

        begin(n_img)
        push("<< /Type /XObject /Subtype /Image /Width %d /Height %d"
             " /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode"
             " /Length %d >>\nstream\n" % (iw, ih, len(pg["jpeg"])))
        push(pg["jpeg"])
        push("\nendstream\n")
        push("endobj\n")

    xref_at = length[0]
    xref = ["xref\n0 %d\n0000000000 65535 f \n" % (n_objs + 1)]
    for n in range(1, n_objs + 1):
        xref.append("%010d 00000 n \n" % offsets[n])
    push("".join(xref))
    push("trailer\n<< /Size %d /Root 1 0 R /Info 4 0 R >>\nstartxref\n%d\n%%%%EOF\n"
         % (n_objs + 1, xref_at))

    return b"".join(chunks)
