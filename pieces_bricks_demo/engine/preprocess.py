"""Steps 1-6: load, face detection, square crop, upscale, enhance, simplify background.

Images are RGB uint8 numpy arrays unless noted. Masks are float32 in [0, 1].
"""
from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .settings import Settings

MODELS = Path(__file__).resolve().parents[1] / "models"
YUNET = MODELS / "face_detection_yunet_2023mar.onnx"
SEGMENTER = MODELS / "selfie_multiclass_256x256.tflite"

log = logging.getLogger(__name__)


@dataclass
class Face:
    box: tuple[float, float, float, float]   # x, y, w, h in original pixels
    landmarks: list[tuple[float, float]]     # right eye, left eye, nose, mouth right, mouth left
    score: float

    def to_json(self) -> dict:
        return {"box": [round(v, 1) for v in self.box],
                "landmarks": [[round(x, 1), round(y, 1)] for x, y in self.landmarks],
                "score": round(self.score, 3)}


# ---------------------------------------------------------------- 1. load

def load_image(src) -> np.ndarray:
    """Path, bytes or PIL image -> RGB array, EXIF rotation applied."""
    if isinstance(src, (str, Path)):
        im = Image.open(src)
    elif isinstance(src, (bytes, bytearray)):
        import io
        im = Image.open(io.BytesIO(src))
    else:
        im = src
    im = ImageOps.exif_transpose(im)
    return np.array(im.convert("RGB"))


# ---------------------------------------------------------------- 2. faces

@lru_cache(maxsize=1)
def _yunet():
    return cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.6, 0.3, 5000)


def detect_faces(rgb: np.ndarray, max_side: int = 1280, min_score: float = 0.6) -> list[Face]:
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    det = _yunet()
    det.setInputSize((small.shape[1], small.shape[0]))
    det.setScoreThreshold(min_score)
    _, raw = det.detect(np.ascontiguousarray(small[:, :, ::-1]))
    faces = []
    for r in (raw if raw is not None else []):
        r = r.astype(np.float64) / np.array([s] * 14 + [1])
        faces.append(Face(tuple(r[:4]), [(r[4 + 2 * i], r[5 + 2 * i]) for i in range(5)], float(r[14])))
    faces.sort(key=lambda f: f.box[0])  # stable left-to-right order for face_select
    return faces


# ---------------------------------------------------------------- 3. crop

def auto_crop(shape, faces: list[Face], st: Settings) -> list[int]:
    h, w = shape[:2]
    if st.face_select:
        faces = [faces[i] for i in st.face_select if i < len(faces)]
    if st.crop_mode == "center" or not faces:
        side = min(h, w)
        return [(w - side) // 2, (h - side) // 2, side]

    if len(faces) == 1:
        fx, fy, fw, fh = faces[0].box
        side = fh * st.portrait_scale
        cx = fx + fw / 2
        # face centre sits at portrait_shift from the top of the square
        cy = fy + fh / 2 - side * st.portrait_shift + side / 2
    else:
        x0 = min(f.box[0] for f in faces); y0 = min(f.box[1] for f in faces)
        x1 = max(f.box[0] + f.box[2] for f in faces); y1 = max(f.box[1] + f.box[3] for f in faces)
        fh = float(np.median([f.box[3] for f in faces]))
        pad = fh * st.face_pad
        side = max(x1 - x0, y1 - y0) + 2 * pad
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2 + pad * 0.35  # a little more body than headroom

    side = min(side, h, w)
    x = int(round(np.clip(cx - side / 2, 0, w - side)))
    y = int(round(np.clip(cy - side / 2, 0, h - side)))
    return [x, y, int(round(side))]


def apply_crop(rgb: np.ndarray, crop: list[int]) -> np.ndarray:
    x, y, side = crop
    h, w = rgb.shape[:2]
    side = min(side, w - x, h - y)
    return rgb[y:y + side, x:x + side]


# ---------------------------------------------------------------- 4. upscale

@lru_cache(maxsize=3)
def _superres(scale: int):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(MODELS / f"FSRCNN_x{scale}.pb"))
    sr.setModel("fsrcnn", scale)
    return sr


def to_work_size(crop: np.ndarray, st: Settings) -> np.ndarray:
    side = crop.shape[0]
    if side < st.upscale_min and st.upscale_method == "fsrcnn":
        scale = next((s for s in (2, 3, 4) if side * s >= st.upscale_min), 4)
        crop = _superres(scale).upsample(np.ascontiguousarray(crop[:, :, ::-1]))[:, :, ::-1]
    interp = cv2.INTER_AREA if crop.shape[0] >= st.work_size else cv2.INTER_LANCZOS4
    return cv2.resize(crop, (st.work_size, st.work_size), interpolation=interp)


# ---------------------------------------------------------------- masks

def face_mask(shape, faces: list[Face], crop: list[int], scale: float) -> np.ndarray:
    """Soft ellipses over each face, in work-image coordinates."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    cx0, cy0 = crop[0], crop[1]
    for f in faces:
        fx, fy, fw, fh = f.box
        c = ((fx + fw / 2 - cx0) * scale, (fy + fh / 2 - cy0) * scale)
        axes = (fw * 0.62 * scale, fh * 0.66 * scale)
        cv2.ellipse(m, (round(c[0]), round(c[1])), (max(1, round(axes[0])), max(1, round(axes[1]))),
                    0, 0, 360, 1.0, -1)
    k = max(3, int(h * 0.01) | 1)
    return cv2.GaussianBlur(m, (k * 2 + 1, k * 2 + 1), 0)


def eye_mask(shape, faces: list[Face], crop: list[int], scale: float) -> np.ndarray:
    """Soft discs around both eye landmarks, in work-image coordinates."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    for f in faces:
        r = max(2, round(f.box[2] * 0.09 * scale))
        for ex, ey in f.landmarks[:2]:
            cv2.circle(m, (round((ex - crop[0]) * scale), round((ey - crop[1]) * scale)), r, 1.0, -1)
    k = max(3, int(h * 0.006) | 1)
    return cv2.GaussianBlur(m, (k * 2 + 1, k * 2 + 1), 0)


@lru_cache(maxsize=1)
def _segmenter():
    import ctypes
    try:  # MediaPipe's native lib expects GL symbols to be loaded already on headless Linux
        ctypes.CDLL("libGL.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=str(SEGMENTER)),
                                        output_category_mask=True)
    seg = vision.ImageSegmenter.create_from_options(opts)
    # Lives for the whole process. MediaPipe's destructor runs after its own executor has shut
    # down and raises a noisy error at exit, so skip it and let the OS reclaim the memory.
    type(seg).__del__ = lambda self: None
    return seg


_PROBE = """
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from engine.preprocess import _segmenter
import mediapipe as mp
_segmenter().segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((64, 64, 3), np.uint8)))
"""


@lru_cache(maxsize=1)
def mediapipe_ok() -> bool:
    """MediaPipe's native code can abort the whole process, which try/except cannot catch
    (e.g. its tcmalloc can't read the CPU count inside some containers). Try it once in a
    child process; if that dies, use the face-box fallback instead of crashing the server."""
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE, str(Path(__file__).resolve().parents[1])],
                           capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        log.warning("MediaPipe probe timed out; using face-box masks")
        return False
    if r.returncode != 0:
        log.warning("MediaPipe unavailable (exit %s); using face-box masks. %s",
                    r.returncode, r.stderr.decode(errors="replace").strip()[-400:])
    return r.returncode == 0


def segment(rgb: np.ndarray) -> np.ndarray | None:
    """MediaPipe multiclass selfie segmentation -> category map
    (0 bg, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 other). None if unavailable."""
    if not mediapipe_ok():
        return None
    try:
        import mediapipe as mp
        out = _segmenter().segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        return out.category_mask.numpy_view().reshape(rgb.shape[:2]).copy()
    except Exception as e:
        log.warning("segmentation unavailable, falling back to face boxes: %s", e)
        return None


def person_mask(rgb: np.ndarray, fmask: np.ndarray, st: Settings, cats: np.ndarray | None) -> np.ndarray:
    if st.segmentation == "mediapipe" and cats is not None:
        m = (cats > 0).astype(np.float32)
    else:  # padded face boxes
        k = int(rgb.shape[0] * 0.08) | 1
        m = cv2.dilate((fmask > 0.2).astype(np.float32), np.ones((k, k), np.uint8))
    m = np.maximum(m, fmask)
    k = max(3, int(rgb.shape[0] * 0.012) | 1)
    return cv2.GaussianBlur(m, (k * 2 + 1, k * 2 + 1), 0)


# ---------------------------------------------------------------- 5. enhance

def enhance(rgb: np.ndarray, fmask: np.ndarray, st: Settings,
            skin: np.ndarray | None = None, eyes: np.ndarray | None = None) -> np.ndarray:
    """skin: soft mask of face + body skin. eyes: soft mask around eye landmarks."""
    eyes = eyes if eyes is not None else np.zeros_like(fmask)
    img = rgb
    if st.denoise_h > 0:
        img = cv2.fastNlMeansDenoisingColored(img, None, st.denoise_h, st.denoise_h, 7, 21)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    if st.clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=st.clahe_clip, tileGridSize=(st.clahe_grid, st.clahe_grid))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Unsharp on L only, radius measured in bricks so the boost survives the downscale.
    L = lab[:, :, 0].astype(np.float32)

    # Skin shadows have no matching mid-dark skin brick and fall onto dark browns,
    # which reads as a beard. Pull them towards the median skin tone (eyes excluded).
    if skin is not None and st.skin_shadow_lift > 0 and (skin > 0.5).sum() > 100:
        ref = float(np.median(L[skin > 0.5]))
        w = st.skin_shadow_lift * skin * (1 - eyes)
        L = L + w * np.maximum(0, ref - L)

    sigma = max(0.5, st.sharpen_sigma_cells * img.shape[0] / st.size)
    blur = cv2.GaussianBlur(L, (0, 0), sigma)
    amount = st.sharpen_amount + (st.sharpen_face_amount - st.sharpen_amount) * fmask
    amount = amount + (st.sharpen_eye_amount - amount) * eyes
    L = np.clip(L + amount * (L - blur), 0, 255)
    lab[:, :, 0] = L.round().astype(np.uint8)

    if st.saturation != 1.0:
        ab = lab[:, :, 1:].astype(np.float32) - 128
        lab[:, :, 1:] = np.clip(ab * st.saturation + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------- 6. background

def simplify_background(rgb: np.ndarray, pmask: np.ndarray, st: Settings) -> np.ndarray:
    if st.bg_mode == "none" or st.bg_strength <= 0:
        return rgb
    s = st.bg_strength
    # Filter at ~6 px per brick: same effect at brick scale, a fraction of the cost.
    small_side = min(rgb.shape[0], st.size * 6)
    small = cv2.resize(rgb, (small_side, small_side), interpolation=cv2.INTER_AREA)
    if st.bg_mode == "meanshift":
        flat = cv2.pyrMeanShiftFiltering(small, sp=int(4 * s + 3), sr=int(22 * s + 8), maxLevel=1)
    else:
        flat = small
        for _ in range(max(1, round(3 * s))):
            flat = cv2.bilateralFilter(flat, 0, sigmaColor=25 + 20 * s, sigmaSpace=2 + 3 * s)
    flat = cv2.resize(flat, rgb.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    m = pmask[:, :, None]
    return (rgb * m + flat * (1 - m)).round().astype(np.uint8)
