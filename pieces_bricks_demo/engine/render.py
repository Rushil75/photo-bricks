"""Images drawn from a matrix (list of rows of colour codes): preview, flat, grid, compare."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .palette import Palette, hex_to_rgb

GAP = (22, 22, 24)
SS = 4  # supersampling for anti-aliased bricks


def _font(size: int):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _mix(c, target, f):
    return tuple(int(round(a + (b - a) * f)) for a, b in zip(c, target))


@lru_cache(maxsize=256)
def brick_tile(hex_: str, cell: int) -> Image.Image:
    """One 1x1 brick seen from above: bevelled body, round stud with highlight and shadow."""
    c = hex_to_rgb(hex_)
    s = cell * SS
    im = Image.new("RGB", (s, s), GAP)
    d = ImageDraw.Draw(im)
    g = max(SS, round(s * 0.045))           # gap to the next brick
    r = round(s * 0.08)
    d.rounded_rectangle([g, g, s - g - 1, s - g - 1], r, fill=_mix(c, (0, 0, 0), 0.22))       # bottom/right edge
    d.rounded_rectangle([g, g, s - g - 1 - SS * 2, s - g - 1 - SS * 2], r, fill=_mix(c, (255, 255, 255), 0.18))
    b = g + SS * 2
    d.rounded_rectangle([b, b, s - b - 1, s - b - 1], r, fill=c)                               # top face

    cx = cy = s / 2
    rad = s * 0.29
    off = s * 0.045
    d.ellipse([cx - rad + off, cy - rad + off * 1.4, cx + rad + off, cy + rad + off * 1.4],
              fill=_mix(c, (0, 0, 0), 0.38))                                                   # stud shadow
    d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=_mix(c, (255, 255, 255), 0.10))   # stud rim
    ir = rad * 0.86
    d.ellipse([cx - ir + off * 0.4, cy - ir + off * 0.4, cx + ir + off * 0.4, cy + ir + off * 0.4], fill=c)
    hl = Image.new("L", (s, s), 0)                                                             # highlight arc
    hd = ImageDraw.Draw(hl)
    w = max(SS, round(rad * 0.16))
    hd.arc([cx - rad + w, cy - rad + w, cx + rad - w, cy + rad - w], 160, 290, fill=150, width=w)
    im = Image.composite(Image.new("RGB", (s, s), (255, 255, 255)), im, hl)
    return im.resize((cell, cell), Image.LANCZOS)


def _rgb_matrix(matrix: list[list[str]], palette: Palette) -> np.ndarray:
    lut = {code: palette.rgb[i] for i, code in enumerate(palette.codes)}
    return np.array([[lut[c] for c in row] for row in matrix], np.uint8)


def preview(matrix, palette: Palette, cell: int = 22) -> Image.Image:
    n = len(matrix)
    hexes = dict(zip(palette.codes, palette.hexes))
    im = Image.new("RGB", (n * cell, n * cell), GAP)
    for y, row in enumerate(matrix):
        for x, code in enumerate(row):
            im.paste(brick_tile(hexes[code], cell), (x * cell, y * cell))
    return im


def flat(matrix, palette: Palette, cell: int = 16) -> Image.Image:
    im = Image.fromarray(_rgb_matrix(matrix, palette))
    return im.resize((im.width * cell, im.height * cell), Image.NEAREST)


def grid(matrix, palette: Palette, cell: int = 30, block: int = 8) -> Image.Image:
    n = len(matrix)
    margin = cell
    W = n * cell + 2 * margin
    im = Image.new("RGB", (W, W), (255, 255, 255))
    d = ImageDraw.Draw(im)
    rgb = _rgb_matrix(matrix, palette)
    f = _font(round(cell * 0.42))
    fl = _font(round(cell * 0.36))
    for y in range(n):
        for x in range(n):
            x0, y0 = margin + x * cell, margin + y * cell
            c = tuple(int(v) for v in rgb[y, x])
            d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=c)
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            d.text((x0 + cell / 2, y0 + cell / 2), matrix[y][x][1:], fill=(0, 0, 0) if lum > 140 else (255, 255, 255),
                   font=f, anchor="mm")
    for i in range(n + 1):
        thick = 3 if i % block == 0 else 1
        col = (20, 20, 20) if thick > 1 else (90, 90, 90)
        p = margin + i * cell
        d.line([(p, margin), (p, margin + n * cell)], fill=col, width=thick)
        d.line([(margin, p), (margin + n * cell, p)], fill=col, width=thick)
    for i in range(n):
        p = margin + i * cell + cell / 2
        for pos in ((p, margin / 2), (p, W - margin / 2)):
            d.text(pos, str(i + 1), fill=(0, 0, 0), font=fl, anchor="mm")
        for pos in ((margin / 2, p), (W - margin / 2, p)):
            d.text(pos, str(i + 1), fill=(0, 0, 0), font=fl, anchor="mm")
    return im


def compare(panels: list[tuple[Image.Image, str]], title: str, height: int = 640) -> Image.Image:
    pad, cap, head = 24, 44, 64
    ims = [p.resize((round(p.width * height / p.height), height), Image.LANCZOS) for p, _ in panels]
    W = sum(i.width for i in ims) + pad * (len(ims) + 1)
    H = head + height + cap + pad
    out = Image.new("RGB", (W, H), (250, 248, 244))
    d = ImageDraw.Draw(out)
    d.text((pad, head / 2), title, fill=(25, 25, 25), font=_font(26), anchor="lm")
    x = pad
    for im, (_, label) in zip(ims, panels):
        out.paste(im, (x, head))
        d.text((x + im.width / 2, head + height + cap / 2), label, fill=(60, 60, 60), font=_font(18), anchor="mm")
        x += im.width + pad
    return out
