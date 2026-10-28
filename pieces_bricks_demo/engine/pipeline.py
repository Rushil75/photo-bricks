"""run(photo, settings) -> Result, and write_outputs(result, dir) for the 5 deliverable files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import export, preprocess as pp, quantize as q, render
from .palette import Palette, load_palette
from .settings import Settings

ENGINE_VERSION = "0.1.0"


@dataclass
class Result:
    matrix: list[list[str]]
    settings: Settings
    palette: Palette
    source: dict              # file, crop, faces: everything needed to reproduce
    crop_rgb: np.ndarray      # original pixels inside the crop
    processed_rgb: np.ndarray  # after enhancement + background, before downscale

    @property
    def size(self) -> int:
        return len(self.matrix)


def run(photo, settings: Settings | dict | None = None, palette: Palette | None = None,
        image: np.ndarray | None = None) -> Result:
    """photo: path (or a label when `image` is given). Deterministic for a given photo + settings."""
    st = settings if isinstance(settings, Settings) else Settings.from_dict(settings)
    palette = palette or load_palette()

    rgb = image if image is not None else pp.load_image(photo)                    # 1
    faces = pp.detect_faces(rgb)                                                 # 2
    crop = list(st.crop) if st.crop else pp.auto_crop(rgb.shape, faces, st)     # 3
    crop_rgb = pp.apply_crop(rgb, crop)
    crop = [crop[0], crop[1], crop_rgb.shape[0]]
    work = pp.to_work_size(crop_rgb, st)                                         # 4
    scale = st.work_size / crop[2]

    in_crop = [f for f in faces if _inside(f, crop)]
    fmask = pp.face_mask(work.shape, in_crop, crop, scale)
    emask = pp.eye_mask(work.shape, in_crop, crop, scale)
    cats = pp.segment(work) if st.segmentation == "model" else None
    pmask = pp.person_mask(work, fmask, st, cats)
    if cats is not None:  # face skin (3) + body skin (2)
        skin = cv2.GaussianBlur(np.isin(cats, (2, 3)).astype(np.float32), (0, 0), work.shape[0] / st.size / 3)
    else:
        skin = (fmask > 0.5).astype(np.float32)

    img = pp.enhance(work, fmask, st, skin, emask)                               # 5
    img = pp.simplify_background(img, pmask, st)                                 # 6

    grid_rgb = q.downscale(img, st.size)                                          # 7
    face_grid = q.downscale(fmask, st.size)
    eye_grid = q.downscale(emask, st.size) > 0.25
    skin_grid = (q.downscale(skin, st.size) > 0.6) | (eye_grid & (face_grid > 0.5))
    idx = q.quantize(grid_rgb, palette, st, face_grid, skin_grid, eye_grid)     # 8
    matrix = [[palette.codes[k] for k in row] for row in idx.tolist()]

    source = {
        "file": Path(str(photo)).name,
        "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "crop": crop,
        "faces": [f.to_json() for f in faces],
        "segmentation": "selfie-multiclass" if cats is not None else "face-boxes",
        "engine_version": ENGINE_VERSION,
    }
    return Result(matrix, st, palette, source, crop_rgb, img)


def _inside(f: pp.Face, crop) -> bool:
    x, y, s = crop
    cx, cy = f.box[0] + f.box[2] / 2, f.box[1] + f.box[3] / 2
    return x <= cx <= x + s and y <= cy <= y + s


def render_from_matrix(out_dir: Path, palette: Palette | None = None, cell: int = 22) -> dict:
    """Regenerate preview / flat / grid / counts from matrix.json alone."""
    out_dir = Path(out_dir)
    palette = palette or load_palette()
    doc = export.read_matrix(out_dir / "matrix.json")
    export.validate(doc, palette)
    m = doc["grid"]
    render.preview(m, palette, cell).save(out_dir / "preview.png", optimize=True)
    render.flat(m, palette).save(out_dir / "flat.png", optimize=True)
    render.grid(m, palette).save(out_dir / "grid.png", optimize=True)
    rows = export.write_counts(out_dir / "counts.csv", m, palette)
    return {"doc": doc, "counts": rows}


def write_outputs(result: Result, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export.write_matrix(out_dir / "matrix.json", result.matrix, result.palette,
                        result.settings.to_dict(), result.source)
    render_from_matrix(out_dir, result.palette)
    # Reference images for review (not part of the 5 deliverables).
    Image.fromarray(result.crop_rgb).resize((768, 768), Image.LANCZOS).save(out_dir / "source.jpg", quality=92)
    Image.fromarray(result.processed_rgb).resize((768, 768), Image.LANCZOS).save(out_dir / "processed.jpg", quality=92)
    return out_dir
