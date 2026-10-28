"""Palette loading: palette.json -> codes, names, sRGB and CIELAB (D65) arrays."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from skimage.color import rgb2lab

DEFAULT_PALETTE = Path(__file__).resolve().parents[1] / "config" / "palette.json"


def hex_to_rgb(hex_: str) -> tuple[int, int, int]:
    h = hex_.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@dataclass(frozen=True)
class Palette:
    name: str
    version: str
    codes: tuple[str, ...]
    names: tuple[str, ...]
    hexes: tuple[str, ...]
    rgb: np.ndarray  # (K, 3) uint8
    lab: np.ndarray  # (K, 3) float64

    def __len__(self) -> int:
        return len(self.codes)

    def index(self, code: str) -> int:
        return self.codes.index(code)

    def indices(self, codes) -> list[int]:
        return [self.index(c) for c in codes]

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "colors": [
                {"code": c, "name": n, "hex": h}
                for c, n, h in zip(self.codes, self.names, self.hexes)
            ],
        }


@lru_cache(maxsize=8)
def load_palette(path: str | Path = DEFAULT_PALETTE) -> Palette:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    colors = data["colors"]
    codes = tuple(c["code"] for c in colors)
    if len(set(codes)) != len(codes):
        raise ValueError("Duplicate colour codes in palette")
    hexes = tuple(c["hex"].upper() for c in colors)
    rgb = np.array([hex_to_rgb(h) for h in hexes], dtype=np.uint8)
    lab = rgb2lab(rgb.reshape(1, -1, 3).astype(np.float64) / 255.0).reshape(-1, 3)
    # Version = palette name + hash of the colour table, so any HEX edit changes it.
    digest = hashlib.sha256(json.dumps(colors, sort_keys=True).encode()).hexdigest()[:8]
    version = f"{data.get('name', 'palette').replace(' ', '-').lower()}-{digest}"
    return Palette(
        name=data.get("name", "palette"),
        version=version,
        codes=codes,
        names=tuple(c["name"] for c in colors),
        hexes=hexes,
        rgb=rgb,
        lab=lab,
    )
