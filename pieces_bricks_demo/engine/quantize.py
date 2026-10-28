"""Steps 7-8: downscale to N x N and match every cell to the palette in CIELAB (CIEDE2000)."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from .palette import Palette
from .settings import Settings


def downscale(img: np.ndarray, size: int) -> np.ndarray:
    """Area-average resize; works for RGB images and float masks."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_lab(rgb: np.ndarray) -> np.ndarray:
    return rgb2lab(rgb.astype(np.float64) / 255.0)  # sRGB -> linear -> XYZ -> LAB, D65


def _allowed(palette: Palette, skin: np.ndarray, eyes: np.ndarray, st: Settings) -> np.ndarray:
    """(N, N, K) bool: which palette colours each cell may use."""
    n = skin.shape[0]
    allowed = np.ones((n, n, len(palette)), bool)
    if st.skin_lock and st.skin_palette:
        for cells, codes in ((skin & ~eyes, st.skin_palette), (skin & eyes, st.skin_palette + st.eye_extra)):
            forbid = [i for i, c in enumerate(palette.codes) if c not in codes]
            rows, cols = np.nonzero(cells)
            allowed[rows[:, None], cols[:, None], forbid] = False
    return allowed


def match(lab: np.ndarray, palette: Palette, allowed: np.ndarray, kL: float) -> np.ndarray:
    """Nearest palette index per cell, no dithering. Ties go to the lower index."""
    d = deltaE_ciede2000(lab[:, :, None, :], palette.lab[None, None, :, :], kL=kL)
    d = np.where(allowed, d, np.inf)
    return np.argmin(d, axis=2)


def dither_fs(lab: np.ndarray, palette: Palette, allowed: np.ndarray, kL: float,
              strength: np.ndarray, clamp: float) -> np.ndarray:
    """Floyd-Steinberg in LAB, left to right, top to bottom. strength is per cell (N, N)."""
    n = lab.shape[0]
    work = lab.copy()
    out = np.zeros((n, n), np.int64)
    pal = palette.lab
    for y in range(n):
        for x in range(n):
            px = work[y, x]
            d = deltaE_ciede2000(np.broadcast_to(px, pal.shape), pal, kL=kL)
            d[~allowed[y, x]] = np.inf
            k = int(np.argmin(d))
            out[y, x] = k
            err = np.clip(px - pal[k], -clamp, clamp)
            for dx, dy, w in ((1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16)):
                xx, yy = x + dx, y + dy
                if 0 <= xx < n and yy < n:
                    work[yy, xx] += err * w * strength[yy, xx]
    return out


def quantize(rgb_grid: np.ndarray, palette: Palette, st: Settings,
             face_grid: np.ndarray, skin_grid: np.ndarray, eye_grid: np.ndarray | None = None) -> np.ndarray:
    """rgb_grid: (N, N, 3) uint8. face_grid: (N, N) face-mask coverage 0..1.
    skin_grid / eye_grid: (N, N) bool, cells limited to skin_palette (+ eye_extra).
    Returns (N, N) palette indices."""
    lab = to_lab(rgb_grid)
    eye_grid = eye_grid if eye_grid is not None else np.zeros_like(skin_grid)
    allowed = _allowed(palette, skin_grid, eye_grid, st)
    kL = 1.0 / max(st.l_weight, 1e-6)
    if not st.dither:
        return match(lab, palette, allowed, kL)
    strength = st.dither_strength + (st.dither_face_strength - st.dither_strength) * np.clip(face_grid, 0, 1)
    return dither_fs(lab, palette, allowed, kL, strength, st.dither_clamp)
