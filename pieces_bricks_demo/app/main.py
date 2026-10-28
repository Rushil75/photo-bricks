"""FastAPI demo: upload a photo, crop, convert with the engine, edit bricks, download the pack.

Run: .venv/bin/uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from engine import Settings, load_palette, render_from_matrix, run
from engine import export, preprocess as pp

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
UPLOADS = ROOT / "uploads"
STATIC = Path(__file__).resolve().parent / "static"
UPLOADS.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

DISPLAY_MAX = 1400       # longest side of the image sent to the browser for cropping
SIZES = [32, 48, 64, 96]

app = FastAPI(title="Pieces & Bricks demo")

# Private link: when DEMO_PASSWORD is set, every request needs it (browser login prompt, any username).
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "")


@app.middleware("http")
async def require_password(request: Request, call_next):
    if not DEMO_PASSWORD or request.url.path == "/healthz":
        return await call_next(request)
    try:
        scheme, _, encoded = request.headers.get("authorization", "").partition(" ")
        password = base64.b64decode(encoded).decode().partition(":")[2] if scheme.lower() == "basic" else ""
    except (binascii.Error, UnicodeDecodeError):
        password = ""
    if secrets.compare_digest(password.encode(), DEMO_PASSWORD.encode()):
        return await call_next(request)
    return Response("Password required", status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="Pieces & Bricks demo"'})


_status = {"segmentation": "checking", "results": "checking"}


def _warm_up() -> None:
    """Runs in the background so the server answers immediately after a deploy."""
    _status["segmentation"] = "mediapipe" if pp.mediapipe_ok() else "face-boxes (MediaPipe unavailable)"
    if (OUTPUT / "index.json").exists():
        _status["results"] = "ready"
        return
    # Demo results aren't in git; build them once per deploy, in a separate process.
    _status["results"] = "generating"
    r = subprocess.run([sys.executable, "run_demo.py"], cwd=ROOT, capture_output=True, text=True)
    _status["results"] = "ready" if r.returncode == 0 else f"failed (exit {r.returncode})"
    if r.returncode != 0:
        print("run_demo.py failed:\n" + r.stderr[-2000:], flush=True)


@app.on_event("startup")
def start_warm_up() -> None:
    threading.Thread(target=_warm_up, daemon=True).start()


@app.get("/healthz")
def healthz():
    return {"ok": True, **_status}
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT), name="output")

_engine_lock = threading.Lock()  # MediaPipe / OpenCV DNN objects are shared and not thread-safe
_images: "OrderedDict[str, np.ndarray]" = OrderedDict()


# ---------------------------------------------------------------- image store

def _store(data: bytes, name: str) -> dict:
    try:
        rgb = pp.load_image(data)
    except Exception:
        raise HTTPException(400, "Could not read that file as an image")
    image_id = uuid.uuid4().hex[:12]
    d = UPLOADS / image_id
    d.mkdir()
    (d / ("original" + (Path(name).suffix.lower() or ".jpg"))).write_bytes(data)
    h, w = rgb.shape[:2]
    s = min(1.0, DISPLAY_MAX / max(h, w))
    Image.fromarray(rgb).resize((round(w * s), round(h * s)), Image.LANCZOS).save(d / "display.jpg", quality=90)
    with _engine_lock:
        faces = pp.detect_faces(rgb)
    suggested = pp.auto_crop(rgb.shape, faces, Settings())
    meta = {"image_id": image_id, "name": name, "width": w, "height": h,
            "display_url": f"/api/images/{image_id}/display.jpg",
            "faces": [f.to_json() for f in faces], "suggested_crop": suggested}
    (d / "meta.json").write_text(json.dumps(meta))
    _cache(image_id, rgb)
    return meta


def _cache(image_id: str, rgb: np.ndarray) -> None:
    _images[image_id] = rgb
    _images.move_to_end(image_id)
    while len(_images) > 8:
        _images.popitem(last=False)


def _get_image(image_id: str) -> tuple[np.ndarray, dict]:
    d = UPLOADS / image_id
    if not image_id.isalnum() or not (d / "meta.json").exists():
        raise HTTPException(404, "Unknown image; upload it again")
    meta = json.loads((d / "meta.json").read_text())
    if image_id not in _images:
        _cache(image_id, pp.load_image(next(d.glob("original.*"))))
    return _images[image_id], meta


# ---------------------------------------------------------------- API

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/results")
def results_page():
    return FileResponse(STATIC / "results.html")


@app.get("/api/palette")
def palette():
    return load_palette().to_json()


@app.get("/api/samples")
def samples():
    return [p.name for p in sorted(INPUT.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]


def _sample_path(name: str) -> Path:
    path = INPUT / name
    if path.parent != INPUT or not path.exists():
        raise HTTPException(404, "No such sample")
    return path


@app.get("/api/samples/{name}/thumb.jpg")
def sample_thumb(name: str):
    im = Image.fromarray(pp.load_image(_sample_path(name)))
    im.thumbnail((240, 240))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.post("/api/samples/{name}")
def use_sample(name: str):
    return _store(_sample_path(name).read_bytes(), name)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    return _store(await file.read(), file.filename or "upload.jpg")


@app.get("/api/images/{image_id}/display.jpg")
def display_image(image_id: str):
    _get_image(image_id)
    return FileResponse(UPLOADS / image_id / "display.jpg")


class ConvertRequest(BaseModel):
    image_id: str
    size: int = Field(48, ge=8, le=160)
    crop: Optional[list[int]] = None   # [x, y, side] in original pixels; None = auto
    dither: bool = False
    settings: dict = Field(default_factory=dict)  # any other engine setting


@app.post("/api/convert")
def convert(req: ConvertRequest):
    rgb, meta = _get_image(req.image_id)
    try:
        st = Settings.from_dict({**req.settings, "size": req.size, "dither": req.dither, "crop": req.crop})
    except (TypeError, ValueError) as e:
        raise HTTPException(422, str(e))
    with _engine_lock:
        res = run(meta["name"], st, image=rgb)
    pal = res.palette
    crop_png = io.BytesIO()
    Image.fromarray(res.crop_rgb).resize((512, 512), Image.LANCZOS).save(crop_png, "JPEG", quality=88)
    return {
        "size": res.size,
        "palette_version": pal.version,
        "settings": st.to_dict(),
        "source": res.source,
        "grid": res.matrix,
        "counts": export.counts(res.matrix, pal),
        "crop_image": "data:image/jpeg;base64," + base64.b64encode(crop_png.getvalue()).decode(),
    }


class ExportRequest(BaseModel):
    grid: list[list[str]]
    settings: dict = Field(default_factory=dict)
    source: dict = Field(default_factory=dict)


@app.post("/api/export")
def export_pack(req: ExportRequest):
    """Zip of matrix.json + preview/flat/grid PNGs + counts.csv, all rendered from the (edited) matrix."""
    pal = load_palette()
    doc = {"size": len(req.grid), "palette_version": pal.version, "grid": req.grid}
    try:
        export.validate(doc, pal)
    except ValueError as e:
        raise HTTPException(422, str(e))
    tmp = Path(tempfile.mkdtemp())
    try:
        export.write_matrix(tmp / "matrix.json", req.grid, pal, req.settings, req.source)
        render_from_matrix(tmp, pal)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("matrix.json", "counts.csv", "preview.png", "flat.png", "grid.png"):
                z.write(tmp / name, name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n = len(req.grid)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="mosaic_{n}x{n}.zip"'})


@app.get("/api/results")
def batch_results():
    """The offline Phase 1 results produced by run_demo.py."""
    idx = OUTPUT / "index.json"
    items = json.loads(idx.read_text()) if idx.exists() else []
    for it in items:
        d = OUTPUT / it["id"]
        it["counts"] = export.counts(export.read_matrix(d / "matrix.json")["grid"], load_palette())
        compare = OUTPUT / f"{it['photo']}_{it['size']}_compare.png"
        it["compare"] = f"/output/{compare.name}" if compare.exists() else None
    return items
