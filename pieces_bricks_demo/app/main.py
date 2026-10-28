"""FastAPI demo: upload a photo, crop, convert with the engine, edit bricks, download the pack.

Run: .venv/bin/uvicorn app.main:app --reload --port 8000

Stateless, so it runs on serverless hosts (Vercel) as well as a normal server: nothing is
written to disk, and every /api/convert call carries its own photo (or a sample's name).
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import os
import secrets
import shutil
import tempfile
import threading
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from engine import Settings, load_palette, render_from_matrix, run
from engine import export, preprocess as pp

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
STATIC = Path(__file__).resolve().parent / "static"

DISPLAY_MAX = 1400       # longest side of a sample image sent to the browser for cropping

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


@app.get("/healthz")
def healthz():
    checked = pp.segmenter_ok.cache_info().currsize > 0
    seg = ("selfie-multiclass" if pp.segmenter_ok() else "face-boxes (segmenter unavailable)") if checked \
        else "not checked yet (runs on the first conversion)"
    return {"ok": True, "segmentation": seg, "results": (OUTPUT / "index.json").exists()}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
if OUTPUT.is_dir():
    app.mount("/output", StaticFiles(directory=OUTPUT), name="output")

_engine_lock = threading.Lock()  # the OpenCV DNN / LiteRT objects are shared and not thread-safe


# ---------------------------------------------------------------- images

def _read_image(data: bytes):
    try:
        return pp.load_image(data)
    except Exception:
        raise HTTPException(400, "Could not read that file as an image")


def _sample_path(name: str) -> Path:
    path = INPUT / name
    if path.parent != INPUT or not path.exists():
        raise HTTPException(404, "No such sample")
    return path


@lru_cache(maxsize=8)
def _sample_rgb(name: str):
    return pp.load_image(_sample_path(name))


def _describe(rgb) -> dict:
    """Size, faces and the suggested square crop, all in the image's own pixels."""
    with _engine_lock:
        faces = pp.detect_faces(rgb)
    h, w = rgb.shape[:2]
    return {"width": w, "height": h, "faces": [f.to_json() for f in faces],
            "suggested_crop": pp.auto_crop(rgb.shape, faces, Settings())}


def _jpeg(im: Image.Image, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------- pages + API

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


@app.get("/api/samples/{name}/thumb.jpg")
def sample_thumb(name: str):
    im = Image.fromarray(_sample_rgb(name))
    im.thumbnail((240, 240))
    return Response(_jpeg(im, 85), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/samples/{name}/display.jpg")
def sample_display(name: str):
    im = Image.fromarray(_sample_rgb(name))
    im.thumbnail((DISPLAY_MAX, DISPLAY_MAX))
    return Response(_jpeg(im, 90), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.post("/api/samples/{name}")
def describe_sample(name: str):
    return {**_describe(_sample_rgb(name)), "sample": name,
            "display_url": f"/api/samples/{name}/display.jpg"}


@app.post("/api/detect")
def describe_upload(file: UploadFile = File(...)):
    """Faces and suggested crop for an uploaded photo. Nothing is stored."""
    return _describe(_read_image(file.file.read()))


class ConvertParams(BaseModel):
    sample: Optional[str] = None       # a file in input/; otherwise the uploaded `file` is used
    size: int = Field(48, ge=8, le=160)
    crop: Optional[list[int]] = None   # [x, y, side] in the image's pixels; None = auto
    dither: bool = False
    settings: dict = Field(default_factory=dict)  # any other engine setting


@app.post("/api/convert")
def convert(params: str = Form(...), file: Optional[UploadFile] = File(None)):
    try:
        req = ConvertParams.model_validate_json(params)
    except ValidationError as e:
        raise HTTPException(422, str(e))
    if req.sample:
        name, rgb = req.sample, _sample_rgb(req.sample)
    elif file is not None:
        name, rgb = file.filename or "upload.jpg", _read_image(file.file.read())
    else:
        raise HTTPException(422, "Send a photo or a sample name")
    try:
        st = Settings.from_dict({**req.settings, "size": req.size, "dither": req.dither, "crop": req.crop})
    except (TypeError, ValueError) as e:
        raise HTTPException(422, str(e))
    with _engine_lock:
        res = run(name, st, image=rgb)
    pal = res.palette
    crop_jpg = _jpeg(Image.fromarray(res.crop_rgb).resize((512, 512), Image.LANCZOS))
    return {
        "size": res.size,
        "palette_version": pal.version,
        "settings": st.to_dict(),
        "source": res.source,
        "grid": res.matrix,
        "counts": export.counts(res.matrix, pal),
        "crop_image": "data:image/jpeg;base64," + base64.b64encode(crop_jpg).decode(),
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
    tmp = Path(tempfile.mkdtemp())  # /tmp is writable on serverless hosts too
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
    """The offline Phase 1 results produced by run_demo.py (committed under output/)."""
    idx = OUTPUT / "index.json"
    items = json.loads(idx.read_text()) if idx.exists() else []
    for it in items:
        d = OUTPUT / it["id"]
        it["counts"] = export.counts(export.read_matrix(d / "matrix.json")["grid"], load_palette())
        compare = OUTPUT / f"{it['photo']}_{it['size']}_compare.png"
        it["compare"] = f"/output/{compare.name}" if compare.exists() else None
    return items
