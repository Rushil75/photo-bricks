# Pieces & Bricks: mosaic demo

Turns a photo into an N×N brick mosaic using only the client's 40 colours (C01–C40), with exact brick counts.

## Setup

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements-dev.txt
```

The model files are in `models/`: YuNet face detector, MediaPipe selfie-multiclass segmenter (run with LiteRT), and FSRCNN x2/x3/x4 super-resolution.

## Phase 1: offline results

```bash
.venv/bin/python run_demo.py                                   # all photos, all sizes, both variants
.venv/bin/python run_demo.py --photo p5 --size 48 --dither off
.venv/bin/python run_demo.py --photo p1 --set portrait_scale=3.0 --set skin_shadow_lift=0.7
.venv/bin/python -m pytest                                     # the four automated quality checks
```

Each result goes to `output/<photo>_<size>_<nodither|dither>/` and contains `matrix.json`, `preview.png`, `flat.png`,
`grid.png` and `counts.csv`, plus `source.jpg` and `processed.jpg` for review. Comparison sheets are written to
`output/<photo>_<size>_compare.png`. Everything except `matrix.json` is rendered from `matrix.json`.

### Client pack

After internal review, set the chosen results, titles and notes in `config/pack.yaml`, then:

```bash
.venv/bin/python make_pack.py      # -> output/client_pack/PiecesBricks_Demo_Results.pdf
```

The PDF contains a cover, a 1-page method summary, two pages per photo (large preview; comparison sheet + brick list),
and the open questions. `output/client_pack/pack_manifest.json` records the exact settings of every result in it.

## Phase 2: web demo (FastAPI)

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- `/`: Studio. Upload or pick a photo, crop/zoom, choose a size, generate, click or drag to repaint bricks, see live
  counts, and download a pack (zip of the 5 files, re-rendered from the edited matrix).
  `/?sample=p5_man_colour_bg.png&size=48` opens straight into a result.
- `/results`: gallery of the batch results from `run_demo.py`.
- API: `POST /api/detect`, `POST /api/convert` (multipart: `params` JSON + `file`, or a `sample` name),
  `POST /api/export`, `GET /api/palette`, `GET /api/results`. The app is stateless: nothing is written to disk.

## Deploying

**Vercel (Hobby/free):** import the repo, Root Directory `pieces_bricks_demo`, preset FastAPI (finds `app/main.py`),
install command `pip install -r requirements.txt`, and add a `DEMO_PASSWORD` environment variable.
`.vercelignore` keeps the bundle small; the 18 demo results ship from the committed `output/` folder.

**Docker / Render:** the `Dockerfile` runs the same app; `render.yaml` (repo root) sets up a Render web service.

- With `DEMO_PASSWORD` set, the browser asks for a password once (any username). `/healthz` stays open.
- A conversion peaks at about 500 MB of memory. Uploads are shrunk to 2000 px in the browser first (Vercel limits
  request bodies to 4.5 MB).
- After changing the engine, run `python run_demo.py` and commit `output/` so the deployed gallery matches.

## Layout

```
engine/      palette, settings, preprocess (steps 1-6), quantize (7-8), render, export, pipeline
config/      palette.json (built from the client's xlsx by tools/build_palette.py), photos.yaml (per-photo crops)
input/       client originals, read-only. p2 and p5 were supplied as PNG and are kept as PNG.
app/         FastAPI app + static front end
tests/       palette-only, counts, determinism, rendered-from-matrix
```

## Engine notes

- The pipeline follows the brief's 8 steps. The face-specific additions below are all settings in `engine/settings.py`.
- **Skin shadow lift** (`skin_shadow_lift`): the palette has no mid-dark pinkish skin tone, so shadowed jaws and
  necks were matching C09/C10/C37 and looked like a beard. Shadows inside the segmenter's skin mask are lifted towards
  the median skin tone. Eye areas are excluded.
- **Skin lock** (`skin_lock`, `skin_palette`, `eye_extra`): skin cells may only use skin tones (C02–C09, C38, C39),
  C37 for lips and C01/C10/C40 for brows. Cells at the YuNet eye landmarks may also use blues for eye colour. This
  enforces the "no stray green or grey patches on faces" check.
- **Eye sharpening** (`sharpen_eye_amount`): the strongest unsharp mask is applied around the eye landmarks.
  Sharpening radii are measured in bricks, so the extra detail survives the downscale.
- The background filter runs at about 6 px per brick, which takes about 2–3 s per 48×48 result on CPU.
