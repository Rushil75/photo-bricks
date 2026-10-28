"""matrix.json and counts.csv. matrix.json is the single source of truth for every other output."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .palette import Palette


def write_matrix(path: Path, matrix: list[list[str]], palette: Palette, settings: dict, source: dict) -> None:
    doc = {
        "size": len(matrix),
        "palette_version": palette.version,
        "settings": settings,
        "source": source,
        "grid": matrix,
    }
    # One row per line keeps the file diffable and readable.
    body = json.dumps({k: v for k, v in doc.items() if k != "grid"}, indent=2)[:-2]
    rows = ",\n".join("    " + json.dumps(r, separators=(",", ":")) for r in matrix)
    Path(path).write_text(f'{body},\n  "grid": [\n{rows}\n  ]\n}}\n')


def read_matrix(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def validate(doc: dict, palette: Palette) -> None:
    n = doc["size"]
    grid = doc["grid"]
    if len(grid) != n or any(len(r) != n for r in grid):
        raise ValueError(f"grid is not {n}x{n}")
    bad = {c for r in grid for c in r} - set(palette.codes)
    if bad:
        raise ValueError(f"codes not in palette: {sorted(bad)}")


def counts(matrix: list[list[str]], palette: Palette) -> list[dict]:
    """Every used colour with its quantity, quantity descending, then code."""
    c = Counter(code for row in matrix for code in row)
    rows = [{"code": code, "name": palette.names[i], "hex": palette.hexes[i], "quantity": c[code]}
            for i, code in enumerate(palette.codes) if c[code]]
    rows.sort(key=lambda r: (-r["quantity"], r["code"]))
    return rows


def write_counts(path: Path, matrix: list[list[str]], palette: Palette) -> list[dict]:
    rows = counts(matrix, palette)
    total = sum(r["quantity"] for r in rows)
    n = len(matrix)
    assert total == n * n, f"count total {total} != {n * n}"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "hex", "quantity"])
        for r in rows:
            w.writerow([r["code"], r["name"], r["hex"], r["quantity"]])
        w.writerow(["TOTAL", "", "", total])
    return rows
