"""The four automated quality checks from the brief:
palette-only codes, exact counts, determinism, and outputs rendered from matrix.json."""
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from engine import Settings, load_palette, render_from_matrix, run, write_outputs

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "input" / "p5_man_colour_bg.png"
PALETTE = load_palette()


@pytest.fixture(scope="module", params=[False, True], ids=["nodither", "dither"])
def out_dir(request, tmp_path_factory):
    d = tmp_path_factory.mktemp(f"p5_{request.param}")
    write_outputs(run(PHOTO, Settings(size=48, dither=request.param)), d)
    return d


def test_palette_loaded():
    assert len(PALETTE) == 40
    assert PALETTE.codes[0] == "C01" and PALETTE.codes[-1] == "C40"


def test_every_cell_is_a_palette_code(out_dir):
    doc = json.loads((out_dir / "matrix.json").read_text())
    assert doc["size"] == 48 and len(doc["grid"]) == 48
    assert all(len(row) == 48 for row in doc["grid"])
    assert {c for row in doc["grid"] for c in row} <= set(PALETTE.codes)


def test_preview_pixels_only_use_palette_colours(out_dir):
    # flat.png has one solid colour per cell, so every pixel must be a palette RGB.
    colours = {c for _, c in Image.open(out_dir / "flat.png").convert("RGB").getcolors(1 << 20)}
    assert colours <= {tuple(int(v) for v in rgb) for rgb in PALETTE.rgb}


def test_counts_sum_and_match_matrix(out_dir):
    doc = json.loads((out_dir / "matrix.json").read_text())
    recount = Counter(c for row in doc["grid"] for c in row)
    with open(out_dir / "counts.csv") as f:
        rows = list(csv.DictReader(f))
    total = rows.pop()
    assert total["code"] == "TOTAL" and int(total["quantity"]) == 48 * 48
    assert sum(int(r["quantity"]) for r in rows) == 48 * 48
    assert {r["code"]: int(r["quantity"]) for r in rows} == dict(recount)
    qty = [int(r["quantity"]) for r in rows]
    assert qty == sorted(qty, reverse=True)


@pytest.mark.parametrize("dither", [False, True])
def test_deterministic(dither, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_outputs(run(PHOTO, Settings(dither=dither)), a)
    write_outputs(run(PHOTO, Settings(dither=dither)), b)
    assert (a / "matrix.json").read_bytes() == (b / "matrix.json").read_bytes()


def test_outputs_come_from_matrix_json(out_dir, tmp_path):
    # Edit one cell in matrix.json only; regenerating must reflect it everywhere.
    d = tmp_path / "edited"
    shutil.copytree(out_dir, d)
    doc = json.loads((d / "matrix.json").read_text())
    old = doc["grid"][0][0]
    new = "C24" if old != "C24" else "C23"
    doc["grid"][0][0] = new
    (d / "matrix.json").write_text(json.dumps(doc))
    render_from_matrix(d)

    with open(d / "counts.csv") as f:
        counts = {r["code"]: int(r["quantity"]) for r in csv.DictReader(f)}
    with open(out_dir / "counts.csv") as f:
        before = {r["code"]: int(r["quantity"]) for r in csv.DictReader(f)}
    assert counts[new] == before.get(new, 0) + 1
    assert counts.get(old, 0) == before[old] - 1

    flat = Image.open(d / "flat.png").convert("RGB")
    assert flat.getpixel((2, 2)) == tuple(int(v) for v in PALETTE.rgb[PALETTE.index(new)])


def test_regenerating_is_byte_identical(out_dir, tmp_path):
    d = tmp_path / "same"
    shutil.copytree(out_dir, d)
    render_from_matrix(d)
    for name in ("preview.png", "grid.png", "flat.png", "counts.csv"):
        assert (d / name).read_bytes() == (out_dir / name).read_bytes(), name
