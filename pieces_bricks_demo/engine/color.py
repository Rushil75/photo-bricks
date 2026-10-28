"""sRGB -> CIELAB (D65, 2°) and CIEDE2000, in plain NumPy.

Same formulas and constants as skimage.color.rgb2lab / deltaE_ciede2000, so results match,
without shipping scikit-image + SciPy (~140 MB) to the serverless deployment.
"""
from __future__ import annotations

import numpy as np

XYZ_FROM_RGB = np.array([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])
D65_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: float array in [0, 1], last axis = channels."""
    arr = np.asarray(rgb, dtype=np.float64).copy()
    mask = arr > 0.04045
    arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    xyz = arr @ XYZ_FROM_RGB.T / D65_WHITE
    mask = xyz > 0.008856
    xyz[mask] = np.cbrt(xyz[mask])
    xyz[~mask] = 7.787 * xyz[~mask] + 16.0 / 116.0
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return np.stack([116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)], axis=-1)


def _polar_2pi(x, y):
    r = np.hypot(x, y)
    t = np.arctan2(y, x)
    t += (t < 0) * 2 * np.pi
    return r, t


def deltaE_ciede2000(lab1, lab2, kL: float = 1, kC: float = 1, kH: float = 1) -> np.ndarray:
    lab1, lab2 = np.broadcast_arrays(np.asarray(lab1, np.float64), np.asarray(lab2, np.float64))
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    Cbar = 0.5 * (np.hypot(a1, b1) + np.hypot(a2, b2))
    c7 = Cbar ** 7
    G = 0.5 * (1 - np.sqrt(c7 / (c7 + 25 ** 7)))
    scale = 1 + G
    C1, h1 = _polar_2pi(a1 * scale, b1)
    C2, h2 = _polar_2pi(a2 * scale, b2)

    Lbar = 0.5 * (L1 + L2)
    tmp = (Lbar - 50) ** 2
    SL = 1 + 0.015 * tmp / np.sqrt(20 + tmp)
    L_term = (L2 - L1) / (kL * SL)

    Cbar = 0.5 * (C1 + C2)
    SC = 1 + 0.045 * Cbar
    C_term = (C2 - C1) / (kC * SC)

    h_diff = h2 - h1
    h_sum = h1 + h2
    CC = C1 * C2
    dH = h_diff.copy()
    dH[h_diff > np.pi] -= 2 * np.pi
    dH[h_diff < -np.pi] += 2 * np.pi
    dH[CC == 0.0] = 0.0
    dH_term = 2 * np.sqrt(CC) * np.sin(dH / 2)

    Hbar = h_sum.copy()
    mask = np.logical_and(CC != 0.0, np.abs(h_diff) > np.pi)
    Hbar[mask * (h_sum < 2 * np.pi)] += 2 * np.pi
    Hbar[mask * (h_sum >= 2 * np.pi)] -= 2 * np.pi
    Hbar[CC == 0.0] *= 2
    Hbar *= 0.5

    T = (1 - 0.17 * np.cos(Hbar - np.deg2rad(30)) + 0.24 * np.cos(2 * Hbar)
         + 0.32 * np.cos(3 * Hbar + np.deg2rad(6)) - 0.20 * np.cos(4 * Hbar - np.deg2rad(63)))
    SH = 1 + 0.015 * Cbar * T
    H_term = dH_term / (kH * SH)

    c7 = Cbar ** 7
    Rc = 2 * np.sqrt(c7 / (c7 + 25 ** 7))
    dtheta = np.deg2rad(30) * np.exp(-(((np.rad2deg(Hbar) - 275) / 25) ** 2))
    R_term = -np.sin(2 * dtheta) * Rc * C_term * H_term

    return np.sqrt(np.maximum(L_term ** 2 + C_term ** 2 + H_term ** 2 + R_term, 0))
