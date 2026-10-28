"""Convert the client's palette spreadsheet into config/palette.json.

Usage: python tools/build_palette.py ../photo-bricks/PhotoBricks_40_Colors_Easy_Codes.xlsx
"""
import argparse
import json
import re
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--out", default=str(ROOT / "config" / "palette.json"))
    ap.add_argument("--name", default="PhotoBricks 40")
    args = ap.parse_args()

    ws = openpyxl.load_workbook(args.xlsx).active
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip().lower() if h else "" for h in rows[0]]
    i_code, i_name, i_hex = header.index("color code"), header.index("color name"), header.index("hex")

    colors = []
    for r in rows[1:]:
        if not r[i_code]:
            continue
        hex_ = str(r[i_hex]).strip().upper()
        if not re.fullmatch(r"#[0-9A-F]{6}", hex_):
            raise ValueError(f"Bad HEX for {r[i_code]}: {hex_}")
        colors.append({"code": str(r[i_code]).strip(), "name": str(r[i_name]).strip(), "hex": hex_})

    out = {"name": args.name, "source": Path(args.xlsx).name, "colors": colors}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {len(colors)} colours to {args.out}")


if __name__ == "__main__":
    main()
