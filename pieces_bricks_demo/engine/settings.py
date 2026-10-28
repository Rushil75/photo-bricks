"""Every tunable engine setting, with defaults. Per-photo overrides live in config/photos.yaml."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Optional


@dataclass
class Settings:
    size: int = 48

    # 3. Crop. crop=[x, y, side] in original pixels overrides auto-crop.
    crop: Optional[list[int]] = None
    crop_mode: str = "faces"          # faces | center
    face_pad: float = 1.6             # multi-face: padding around the union of faces, in face heights
    portrait_scale: float = 2.8       # single face: square side = face height * portrait_scale
    portrait_shift: float = 0.35      # single face: push the face up so shoulders fit (fraction of side)
    face_select: Optional[list[int]] = None  # crop around these face indices only (sorted left to right)

    # 4. Upscale / working resolution
    work_size: int = 1152             # crop is resized to this before enhancement
    upscale_min: int = 1000
    upscale_method: str = "fsrcnn"    # fsrcnn | lanczos

    # 5. Enhance
    denoise_h: float = 3.0            # 0 disables
    clahe_clip: float = 1.0           # 0 disables
    clahe_grid: int = 8
    sharpen_amount: float = 0.25      # unsharp mask outside faces
    sharpen_face_amount: float = 0.4  # unsharp mask inside the face mask
    sharpen_eye_amount: float = 1.2   # unsharp mask around the eye landmarks
    sharpen_sigma_cells: float = 0.6  # blur radius in brick units, so detail survives the downscale
    skin_shadow_lift: float = 0.6     # 0..1, pulls skin shadows towards the median skin tone
    saturation: float = 1.06

    # 6. Background
    bg_mode: str = "bilateral"        # bilateral | meanshift | none
    bg_strength: float = 1.0          # 0..2, scales filter radius and iterations
    segmentation: str = "model"       # model (selfie segmentation) | faces (padded face boxes)

    # 8. Colour matching
    l_weight: float = 1.15            # >1 weights lightness more (CIEDE2000 kL = 1 / l_weight)
    dither: bool = False
    dither_strength: float = 0.5
    dither_face_strength: float = 0.0
    dither_clamp: float = 10.0        # max LAB error carried to neighbours, per channel
    skin_lock: bool = True            # skin cells may only use skin_palette (+ eye_extra around eyes)
    skin_palette: list[str] = field(default_factory=lambda: [
        "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C38", "C39",  # skin tones
        "C37",                                                                 # lips
        "C01", "C10", "C40",                                                   # brows, lashes
    ])
    eye_extra: list[str] = field(default_factory=lambda: ["C21", "C17", "C18", "C36"])  # eye colours

    @classmethod
    def from_dict(cls, d: dict | None) -> "Settings":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        return cls(**d)

    def merged(self, overrides: dict | None) -> "Settings":
        return Settings.from_dict({**asdict(self), **(overrides or {})})

    def to_dict(self) -> dict:
        return asdict(self)
