"""Batch / single runs of the conversion engine.

  python run_demo.py                                  # every photo, every size, both variants
  python run_demo.py --photo p5 --size 48 --dither off
  python run_demo.py --photo p1 --set portrait_scale=3.2 --set l_weight=1.3
"""
import argparse
import json
import logging
import time
from pathlib import Path

import yaml
from PIL import Image

from engine import Settings, load_palette, run, write_outputs
from engine import render

ROOT = Path(__file__).resolve().parent
VARIANTS = {"off": [False], "on": [True], "both": [False, True]}


def load_plan(path=ROOT / "config" / "photos.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def parse_set(items) -> dict:
    out = {}
    for item in items or []:
        k, _, v = item.partition("=")
        out[k] = yaml.safe_load(v)
    return out


def variant_name(dither: bool) -> str:
    return "dither" if dither else "nodither"


def run_one(pid: str, spec: dict, size: int, dither: bool, extra: dict, out_root: Path) -> Path:
    st = Settings.from_dict({**spec.get("settings", {}), **extra, "size": size, "dither": dither})
    t = time.time()
    res = run(ROOT / "input" / spec["file"], st)
    out = write_outputs(res, out_root / f"{pid}_{size}_{variant_name(dither)}")
    print(f"  {out.name:28s} crop={res.source['crop']}  {time.time() - t:.1f}s")
    return out


def make_compare(pid: str, spec: dict, size: int, out_root: Path) -> None:
    a, b = out_root / f"{pid}_{size}_nodither", out_root / f"{pid}_{size}_dither"
    if not (a.exists() and b.exists()):
        return
    total = size * size
    panels = [(Image.open(a / "source.jpg"), "Original crop"),
              (Image.open(a / "preview.png"), "No dithering"),
              (Image.open(b / "preview.png"), "Dithering")]
    title = f"{pid.upper()} · {spec.get('label', '')} · {size}×{size} · {total:,} bricks"
    render.compare(panels, title).save(out_root / f"{pid}_{size}_compare.png", optimize=True)


def write_index(out_root: Path, plan: dict) -> None:
    """output/index.json: every result folder with its label, for the web gallery."""
    items = []
    for d in sorted(p for p in out_root.iterdir() if (p / "matrix.json").exists()):
        pid, size, variant = d.name.rsplit("_", 2)
        items.append({"id": d.name, "photo": pid, "size": int(size), "variant": variant,
                      "label": plan.get(pid, {}).get("label", pid)})
    (out_root / "index.json").write_text(json.dumps(items, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", nargs="*", help="photo ids from photos.yaml (default: all)")
    ap.add_argument("--size", type=int, nargs="*", help="override the sizes in photos.yaml")
    ap.add_argument("--dither", choices=VARIANTS, default="both")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any engine setting")
    ap.add_argument("--out", default=str(ROOT / "output"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    plan = load_plan()
    out_root = Path(args.out)
    extra = parse_set(args.set)
    print(f"palette {load_palette().version}")
    for pid in args.photo or list(plan):
        spec = plan[pid]
        print(f"{pid}: {spec['file']}")
        for size in args.size or spec.get("sizes", [48]):
            for dither in VARIANTS[args.dither]:
                run_one(pid, spec, size, dither, extra, out_root)
            make_compare(pid, spec, size, out_root)
    write_index(out_root, plan)


if __name__ == "__main__":
    main()
